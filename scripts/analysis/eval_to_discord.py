"""Eval → Discord trajectory collage.

LAYOUT (per IoU bucket, one collage per Discord attachment):
    rows  = N fixed anchor cases (deterministic, sorted by uid)
    col 0 = GT 4-view input image (the same render the model saw)
    col 1 = GT mesh (1 iso view, rendered from GT code) — fixed reference
    col 2 = pred mesh at step 1000   (1 iso view)
    col 3 = pred mesh at step 2000
    col 4 = pred mesh at step 3000
    ...   (one column per +1000 step, growing over time)

Each pred cell shows the IoU (small overlay, top-left corner of cell).
Render results are CACHED to predictions/render_cache/{bucket}/{step:06d}_{uid}.png
so a re-run only renders newly-arrived steps (cheap).

DUAL MODE:
  1. EVAL MODE  — orchestrate: parse log → render new step's preds → assemble
     trajectory collage covering ALL prior eval steps → POST to Discord.

         uv run python -m scripts.analysis.eval_to_discord \\
             --step 3000 \\
             --log /home/ubuntu/cadrille/logs/big_bench_shell_50k_*.log \\
             --output-dir /ephemeral/checkpoints/sft-...

  2. SEND MODE — generic Discord pipe for arbitrary analysis images:

         uv run python -m scripts.analysis.eval_to_discord \\
             --send --message "op heatmap" --file plot.png

Reusable building blocks importable from other analysis scripts:
    from scripts.analysis.eval_to_discord import (
        post_to_discord,         # Discord webhook POST (text + files)
        build_grid_collage,      # generic uniform-cell grid
        render_meshes_parallel,  # ProcessPool-backed cadquery → PNG
    )
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Curriculum baseline IoU per step (logs/curriculum_qwen3vl_2b_*.log) for delta.
CURR_BC = {1000:0.120, 2000:0.257, 3000:0.324, 4000:0.389, 5000:0.299,
           6000:0.397, 7000:0.338, 8000:0.409, 9000:0.433, 10000:0.490,
           11000:0.597, 12000:0.579, 13000:0.568, 14000:0.529, 15000:0.530,
           16000:0.523, 17000:0.509, 18000:0.546, 19000:0.546, 20000:0.546}
CURR_DC = {1000:0.212, 2000:0.291, 3000:0.305, 4000:0.363, 5000:0.390,
           6000:0.448, 7000:0.453, 8000:0.416, 9000:0.489, 10000:0.448,
           11000:0.444, 12000:0.458, 13000:0.461, 14000:0.440, 15000:0.480,
           16000:0.477, 17000:0.414, 18000:0.438, 19000:0.451, 20000:0.466}
CURR_FU = {1000:0.274, 2000:0.401, 3000:0.438, 4000:0.411, 5000:0.488,
           6000:0.490, 7000:0.535, 8000:0.545, 9000:0.517, 10000:0.548,
           11000:0.548, 12000:0.560, 13000:0.483, 14000:0.551, 15000:0.540,
           16000:0.556, 17000:0.542, 18000:0.541, 19000:0.553, 20000:0.565}

IOU_BUCKETS = ('BenchCAD val', 'DeepCAD test', 'Fusion360 test')


# ─── REUSABLE: Discord posting ──────────────────────────────────────────────

def post_to_discord(content: str = '',
                    files: list[tuple[str, bytes]] | None = None,
                    webhook: str | None = None) -> list[int]:
    """POST a message + optional attachments. Auto-splits if >10 files / >1900 chars."""
    webhook = webhook or os.environ.get('DISCORD_WEBHOOK_URL')
    if not webhook:
        raise RuntimeError('DISCORD_WEBHOOK_URL not set')
    files = files or []
    statuses = []

    text = content or ''
    text_chunks = [text[i:i + 1900] for i in range(0, len(text), 1900)] or ['']

    first_batch = files[:10]
    payload = {'content': text_chunks[0]}
    multipart = {
        f'files[{j}]': (name, data, 'image/png')
        for j, (name, data) in enumerate(first_batch)
    } or None
    if multipart:
        r = requests.post(webhook, data={'payload_json': json.dumps(payload)},
                          files=multipart, timeout=120)
    else:
        r = requests.post(webhook, json=payload, timeout=30)
    statuses.append(r.status_code)
    if r.status_code >= 400:
        print(f'discord error: {r.status_code} {r.text[:300]}', file=sys.stderr)

    for chunk in text_chunks[1:]:
        r = requests.post(webhook, json={'content': chunk}, timeout=30)
        statuses.append(r.status_code)

    for offset in range(10, len(files), 10):
        batch = files[offset:offset + 10]
        multipart = {
            f'files[{j}]': (name, data, 'image/png')
            for j, (name, data) in enumerate(batch)
        }
        payload = {'content': f'(attachments {offset + 1}-{offset + len(batch)} of {len(files)})'}
        r = requests.post(webhook, data={'payload_json': json.dumps(payload)},
                          files=multipart, timeout=120)
        statuses.append(r.status_code)
    return statuses


# ─── REUSABLE: generic grid collage ─────────────────────────────────────────

def build_grid_collage(rows: list[dict],
                       title: str,
                       col_titles: list[str] | None = None,
                       cell: int = 160,
                       label_w: int = 220) -> bytes:
    """Generic N-column collage. rows: [{'cells': [bytes_or_None, ...], 'label': str}]"""
    if not rows:
        img = Image.new('RGB', (640, 80), (245, 245, 245))
        ImageDraw.Draw(img).text((10, 30), f'{title}  (no rows)', fill=(20, 20, 20))
        buf = io.BytesIO(); img.save(buf, format='PNG'); return buf.getvalue()

    n_cols = max(len(r.get('cells', [])) for r in rows)
    n_rows = len(rows)
    header_h = 50 if col_titles else 28
    W = n_cols * cell + label_w
    H = n_rows * (cell + 4) + header_h

    img = Image.new('RGB', (W, H), (245, 245, 245))
    drw = ImageDraw.Draw(img)
    drw.text((10, 6), title, fill=(20, 20, 20))
    if col_titles:
        for c, ct in enumerate(col_titles[:n_cols]):
            drw.text((c * cell + 4, header_h - 18), ct[:cell // 7], fill=(60, 60, 60))

    for r, row in enumerate(rows):
        y = header_h + r * (cell + 4)
        cells = row.get('cells', [])
        for c in range(n_cols):
            png = cells[c] if c < len(cells) else None
            box = (c * cell + 1, y + 1, c * cell + cell - 1, y + cell - 1)
            if png:
                try:
                    cimg = Image.open(io.BytesIO(png)).convert('RGB').resize(
                        (cell - 2, cell - 2), Image.LANCZOS)
                    img.paste(cimg, (c * cell + 1, y + 1))
                except Exception:
                    drw.rectangle(list(box), fill=(220, 220, 220))
            else:
                drw.rectangle(list(box), fill=(232, 210, 210))
                drw.text((c * cell + cell // 2 - 5, y + cell // 2 - 8),
                         '—', fill=(150, 50, 50))
        drw.text((n_cols * cell + 6, y + 4), str(row.get('label', ''))[:300],
                 fill=(20, 20, 20))

    buf = io.BytesIO(); img.save(buf, format='PNG')
    return buf.getvalue()


# ─── REUSABLE: parallel CPU mesh rendering ──────────────────────────────────

def _render_mesh_to_png(args: tuple) -> tuple[str, bytes | None, str]:
    """Worker: exec cadquery code → mesh → 1-iso PNG bytes. (label, png, status)"""
    label, code, color_rgb, img_size = args
    if not code or not code.strip():
        return label, None, 'empty'

    def _on_alarm(signum, frame): raise TimeoutError('cadquery timeout')
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(15)
    try:
        import trimesh
        import open3d
        import cadquery as cq  # noqa: F401
        from common.datasets import mesh_to_image

        try:
            code_obj = compile(code, '<string>', 'exec')
        except SyntaxError:
            return label, None, 'syntax'
        captured = {}
        g = {'show_object': lambda obj, *a, **kw: captured.setdefault('r', obj)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exec(code_obj, g)
        res = g.get('r') or g.get('result') or captured.get('r')
        if res is None:
            return label, None, 'no_r'
        compound = res.val()
        verts, faces = compound.tessellate(0.001, 0.1)
        if len(faces) < 3:
            return label, None, 'empty_mesh'
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        ext = float(np.max(mesh.extents))
        if ext > 1e-7:
            mesh.apply_scale(1.0 / ext)
        v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
        o3d = open3d.geometry.TriangleMesh()
        o3d.vertices = open3d.utility.Vector3dVector(v)
        o3d.triangles = open3d.utility.Vector3iVector(f)
        o3d.paint_uniform_color(np.array(color_rgb) / 255.0)
        o3d.compute_vertex_normals()
        img = mesh_to_image(o3d, camera_distance=-1.6, front=[1, 1, 1], img_size=img_size)
        buf = io.BytesIO(); img.save(buf, format='PNG')
        return label, buf.getvalue(), 'ok'
    except TimeoutError:
        return label, None, 'timeout'
    except Exception as e:
        return label, None, f'err:{type(e).__name__}'
    finally:
        signal.alarm(0)


def render_meshes_parallel(tasks: list[tuple[str, str, tuple, int]],
                           max_workers: int = 6) -> dict[str, tuple[bytes | None, str]]:
    """tasks: [(label, code, rgb, img_size), ...]   →   {label: (png|None, status)}"""
    results: dict[str, tuple[bytes | None, str]] = {}
    if not tasks:
        return results
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for label, png, status in pool.map(_render_mesh_to_png, tasks):
            results[label] = (png, status)
    return results


def _render_stl_to_png(args: tuple) -> tuple[str, bytes | None, str]:
    """Worker: load STL → 1-iso PNG bytes. (label, png, status)"""
    label, stl_path, color_rgb, img_size = args
    if not stl_path or not Path(stl_path).exists():
        return label, None, 'nofile'

    def _on_alarm(signum, frame): raise TimeoutError('stl render timeout')
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(15)
    try:
        import trimesh
        import open3d
        from common.datasets import mesh_to_image
        mesh = trimesh.load(str(stl_path), force='mesh')
        if mesh.is_empty or len(mesh.faces) < 3:
            return label, None, 'empty_mesh'
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        ext = float(np.max(mesh.extents))
        if ext > 1e-7:
            mesh.apply_scale(1.0 / ext)
        v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
        o3d = open3d.geometry.TriangleMesh()
        o3d.vertices = open3d.utility.Vector3dVector(v)
        o3d.triangles = open3d.utility.Vector3iVector(f)
        o3d.paint_uniform_color(np.array(color_rgb) / 255.0)
        o3d.compute_vertex_normals()
        img = mesh_to_image(o3d, camera_distance=-1.6, front=[1, 1, 1], img_size=img_size)
        buf = io.BytesIO(); img.save(buf, format='PNG')
        return label, buf.getvalue(), 'ok'
    except TimeoutError:
        return label, None, 'timeout'
    except Exception as e:
        return label, None, f'err:{type(e).__name__}'
    finally:
        signal.alarm(0)


def render_stls_parallel(tasks: list[tuple[str, str, tuple, int]],
                         max_workers: int = 6) -> dict[str, tuple[bytes | None, str]]:
    """tasks: [(label, stl_path, rgb, img_size), ...]"""
    results: dict[str, tuple[bytes | None, str]] = {}
    if not tasks:
        return results
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for label, png, status in pool.map(_render_stl_to_png, tasks):
            results[label] = (png, status)
    return results


def find_gt_stl(uid: str, bucket: str) -> Path | None:
    if bucket == 'BenchCAD val':
        cand = REPO_ROOT / 'data/benchcad/val' / f'{uid}.stl'
    elif bucket == 'DeepCAD test':
        cand = REPO_ROOT / 'data/deepcad_test_mesh' / f'{uid}.stl'
    elif bucket == 'Fusion360 test':
        cand = REPO_ROOT / 'data/fusion360_test_mesh' / f'{uid}.stl'
    else:
        return None
    return cand if cand.exists() else None


# ─── EVAL-SPECIFIC: log parse, input lookup ─────────────────────────────────

def parse_eval_block(log_path: Path, step: int) -> dict:
    text = log_path.read_text()
    marker = f'step={step} running IoU eval'
    idx = text.find(marker)
    if idx < 0: return {}
    block = text[idx:idx + 6000]
    out = {}
    for bucket in ('BenchCAD val', 'recode20k train', 'text2cad train',
                   'DeepCAD test', 'Fusion360 test'):
        pat = re.compile(
            rf'\[(?:img|text)/{re.escape(bucket)}\]\s+'
            rf'(?:op_loss_w=(?P<op>-?[\d.]+)\s+)?'
            rf'(?:recall=(?P<rec>[\d.]+)\s+)?'
            rf'(?:rare_recall=(?P<rare>[\d.]+)\s+)?'
            rf'(?:IoU=(?P<iou>[\d.]+))?'
        )
        m = pat.search(block)
        if m:
            d = m.groupdict()
            out[bucket] = {
                'op_loss': float(d['op']) if d['op'] else None,
                'recall':  float(d['rec']) if d['rec'] else None,
                'rare':    float(d['rare']) if d['rare'] else None,
                'iou':     float(d['iou']) if d['iou'] else None,
            }
    return out


def find_input_image(uid: str, bucket: str) -> bytes | None:
    if bucket == 'BenchCAD val':
        cand = REPO_ROOT / 'data/benchcad/val' / f'{uid}_render.png'
    elif bucket == 'DeepCAD test':
        cand = REPO_ROOT / 'data/deepcad_test_mesh' / f'{uid}_render.png'
    elif bucket == 'Fusion360 test':
        cand = REPO_ROOT / 'data/fusion360_test_mesh' / f'{uid}_render.png'
    else:
        return None
    if cand.exists():
        try: return cand.read_bytes()
        except Exception: return None
    return None


def fmt_iou(v: float | None) -> str:
    return f'{v:.3f}' if isinstance(v, (int, float)) else 'N/A'


def fmt_delta(curr_val: float | None, baseline: dict, step: int) -> str:
    if curr_val is None or step not in baseline:
        return ''
    d = curr_val - baseline[step]
    sign = '+' if d > 0 else ''
    emoji = '🟢' if d > 0.05 else '🟡' if d > -0.05 else '🔴'
    return f' ({emoji}{sign}{d:.3f} vs curr {baseline[step]:.3f})'


# ─── TRAJECTORY COLLAGE ─────────────────────────────────────────────────────

def build_trajectory_collage(bucket: str,
                             anchors: list[dict],
                             trajectory: dict,
                             steps: list[int],
                             cell: int = 120,
                             gt_w: int = 200) -> bytes:
    """Build trajectory collage for a bucket.

    anchors:    [{'uid', 'input_png', 'gt_mesh_png'}]   in display order
    trajectory: {uid: {step: (pred_png_or_None, iou_or_None)}}
    steps:      sorted list of step ints to display as pred columns
    Layout:     [4-view input | GT mesh iso | pred@1k | pred@2k | ...]
    """
    n_rows = len(anchors)
    n_step_cols = len(steps)
    header_h = 40
    # col widths: gt_w (4-view) + cell (GT mesh) + n_step_cols * cell (preds)
    W = gt_w + cell + n_step_cols * cell
    H = n_rows * cell + header_h

    img = Image.new('RGB', (W, H), (245, 245, 245))
    drw = ImageDraw.Draw(img)
    drw.text((10, 6),
             f'{bucket}  ({n_rows} anchors × {n_step_cols} steps)  '
             f'pred-cell={cell}px  GT-4view={gt_w}px',
             fill=(20, 20, 20))

    # column headers
    drw.text((max(4, gt_w // 2 - 28), header_h - 16), '4-view input', fill=(60, 60, 60))
    drw.text((gt_w + 4, header_h - 16), 'GT mesh', fill=(40, 100, 40))
    for c, step in enumerate(steps):
        x = gt_w + cell + c * cell
        lbl = f'{step // 1000}k' if step % 1000 == 0 else str(step)
        drw.text((x + 4, header_h - 16), lbl, fill=(60, 60, 60))

    for r, anc in enumerate(anchors):
        y = header_h + r * cell

        # ── col 0: 4-view input ──────────────────────────
        if anc.get('input_png'):
            try:
                im = Image.open(io.BytesIO(anc['input_png'])).convert('RGB')
                im = im.resize((gt_w - 2, cell - 2), Image.LANCZOS)
                img.paste(im, (1, y + 1))
            except Exception:
                drw.rectangle([1, y + 1, gt_w - 1, y + cell - 1], fill=(220, 220, 220))
        else:
            drw.rectangle([1, y + 1, gt_w - 1, y + cell - 1], fill=(232, 210, 210))
        drw.text((4, y + 4), anc['uid'][:20], fill=(20, 20, 20))

        # ── col 1: GT mesh iso ───────────────────────────
        gt_x = gt_w
        if anc.get('gt_mesh_png'):
            try:
                im = Image.open(io.BytesIO(anc['gt_mesh_png'])).convert('RGB').resize(
                    (cell - 2, cell - 2), Image.LANCZOS)
                img.paste(im, (gt_x + 1, y + 1))
            except Exception:
                drw.rectangle([gt_x + 1, y + 1, gt_x + cell - 1, y + cell - 1],
                              fill=(220, 220, 220))
        else:
            drw.rectangle([gt_x + 1, y + 1, gt_x + cell - 1, y + cell - 1],
                          fill=(210, 230, 210))
            drw.text((gt_x + 8, y + cell // 2 - 6), 'GT?', fill=(80, 130, 80))

        # ── cols 2..N: per-step pred ─────────────────────
        traj = trajectory.get(anc['uid'], {})
        for c, step in enumerate(steps):
            x = gt_w + cell + c * cell
            png, iou = traj.get(step, (None, None))
            if png:
                try:
                    im = Image.open(io.BytesIO(png)).convert('RGB').resize(
                        (cell - 2, cell - 2), Image.LANCZOS)
                    img.paste(im, (x + 1, y + 1))
                except Exception:
                    drw.rectangle([x + 1, y + 1, x + cell - 1, y + cell - 1],
                                  fill=(220, 220, 220))
            else:
                drw.rectangle([x + 1, y + 1, x + cell - 1, y + cell - 1],
                              fill=(232, 210, 210))
            if iou is not None and iou >= 0:
                color = (40, 140, 40) if iou >= 0.5 \
                    else (180, 140, 30) if iou >= 0.3 \
                    else (200, 60, 60)
                drw.text((x + 3, y + 3), f'{iou:.2f}', fill=color)

    buf = io.BytesIO(); img.save(buf, format='PNG')
    return buf.getvalue()


# ─── EVAL MODE ──────────────────────────────────────────────────────────────

def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def discover_steps(pred_dir: Path, max_step: int) -> list[int]:
    """All step JSONLs available, excluding step 0 and any > max_step."""
    out = []
    for sf in sorted(pred_dir.glob('step-*.jsonl')):
        try:
            s = int(sf.stem.split('-')[1])
        except (IndexError, ValueError):
            continue
        if 0 < s <= max_step:
            out.append(s)
    return out


def pick_anchors(jsonl_path: Path, n_per_bucket: int) -> dict[str, list[dict]]:
    """Deterministic anchors: sort by uid, take first N per bucket."""
    rows = _read_jsonl(jsonl_path)
    by_bucket: dict[str, list[dict]] = {b: [] for b in IOU_BUCKETS}
    for r in rows:
        if r.get('bucket') in by_bucket:
            by_bucket[r['bucket']].append(r)
    for b, pool in by_bucket.items():
        pool.sort(key=lambda x: x['uid'])
        by_bucket[b] = pool[:n_per_bucket]
    return by_bucket


def eval_main(args):
    log_path = Path(args.log)
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / 'predictions'
    cache_root = pred_dir / 'render_cache'
    cache_root.mkdir(parents=True, exist_ok=True)

    cur_jsonl = pred_dir / f'step-{args.step:06d}.jsonl'
    if not cur_jsonl.exists():
        print(f'JSONL not found: {cur_jsonl}', file=sys.stderr); sys.exit(2)

    # 1. Anchors (deterministic, locked at the current step's JSONL uid order)
    n_anchor = args.anchors_per_bucket
    anchors_by_bucket = pick_anchors(cur_jsonl, n_anchor)
    print(f'anchors: '
          + ', '.join(f'{b}={len(anchors_by_bucket[b])}' for b in IOU_BUCKETS),
          flush=True)

    # 2. Discover step JSONLs (1k, 2k, ..., args.step)
    steps = discover_steps(pred_dir, args.step)
    print(f'trajectory steps: {steps}', flush=True)

    # 3. Build trajectory dict + collect tasks for missing cache files
    #    Cache layout:
    #      render_cache/{bucket}/gt_{uid}.png         (GT mesh, never expires)
    #      render_cache/{bucket}/{step:06d}_{uid}.png (pred, per-step)
    trajectory: dict[str, dict[str, dict[int, tuple[bytes | None, float | None]]]] = {
        b: {} for b in IOU_BUCKETS
    }
    gt_pngs: dict[str, dict[str, bytes | None]] = {b: {} for b in IOU_BUCKETS}

    # Two task lists: GT renders use STL files; pred renders use cadquery code.
    gt_tasks: list[tuple] = []      # (cache_path, label, stl_path, bucket, uid)
    pred_tasks: list[tuple] = []    # (cache_path, label, code, bucket, uid, step)

    for bucket in IOU_BUCKETS:
        bcache = cache_root / bucket.replace(' ', '_')
        bcache.mkdir(exist_ok=True)
        anchor_uids = {a['uid'] for a in anchors_by_bucket[bucket]}

        # GT meshes (one per anchor uid)
        # Strategy: try STL on disk first; if missing or render fails, fall
        # back to gt_code (cadquery exec) when the JSONL has it.
        for r in anchors_by_bucket[bucket]:
            uid = r['uid']
            gt_cache = bcache / f'gt_{uid}.png'
            if gt_cache.exists():
                gt_pngs[bucket][uid] = gt_cache.read_bytes()
                continue
            gt_pngs[bucket][uid] = None
            stl = find_gt_stl(uid, bucket)
            if stl is not None:
                label = f'GT_STL|{bucket}|{uid}'
                gt_tasks.append((gt_cache, label, str(stl), bucket, uid,
                                 r.get('gt_code') or ''))

        # Pred meshes per (step, uid)
        for step in steps:
            sjsonl = pred_dir / f'step-{step:06d}.jsonl'
            if not sjsonl.exists():
                continue
            for r in _read_jsonl(sjsonl):
                if r.get('bucket') != bucket: continue
                if r['uid'] not in anchor_uids: continue
                iou = r.get('iou')
                cache_file = bcache / f'{step:06d}_{r["uid"]}.png'
                if cache_file.exists():
                    trajectory[bucket].setdefault(r['uid'], {})[step] = (
                        cache_file.read_bytes(), iou)
                else:
                    label = f'PRED|{bucket}|{r["uid"]}|{step}'
                    pred_tasks.append(
                        (cache_file, label, r.get('pred_code') or '',
                         bucket, r['uid'], step))

    # 4a. GT meshes — try STL first, fall back to gt_code (cadquery) if STL fails
    if gt_tasks:
        print(f'rendering {len(gt_tasks)} GT meshes from STL ...', flush=True)
        worker_args = [(t[1], t[2], (136, 200, 255), 200) for t in gt_tasks]
        gt_res = render_stls_parallel(worker_args, max_workers=args.workers)
        n_stl_ok = 0
        fallback_tasks = []  # (cache_file, label, code, bucket, uid)
        for cache_file, label, _stl, bucket, uid, gt_code in gt_tasks:
            png, status = gt_res.get(label, (None, '?'))
            if png:
                cache_file.write_bytes(png); n_stl_ok += 1
                gt_pngs[bucket][uid] = png
            elif gt_code:
                fb_label = f'GT_CODE|{bucket}|{uid}'
                fallback_tasks.append((cache_file, fb_label, gt_code, bucket, uid))
            else:
                gt_pngs[bucket][uid] = None
        print(f'  GT-STL rendered: {n_stl_ok}/{len(gt_tasks)} ok', flush=True)

        if fallback_tasks:
            print(f'  fallback: rendering {len(fallback_tasks)} GT meshes from gt_code ...',
                  flush=True)
            worker_args = [(t[1], t[2], (136, 200, 255), 200) for t in fallback_tasks]
            fb_res = render_meshes_parallel(worker_args, max_workers=args.workers)
            n_fb_ok = 0
            for cache_file, label, _code, bucket, uid in fallback_tasks:
                png, _status = fb_res.get(label, (None, '?'))
                if png:
                    cache_file.write_bytes(png); n_fb_ok += 1
                gt_pngs[bucket][uid] = png
            print(f'  GT-code fallback: {n_fb_ok}/{len(fallback_tasks)} ok',
                  flush=True)

    # 4b. Render pred meshes (cadquery code → iso PNG) in parallel
    if pred_tasks:
        print(f'rendering {len(pred_tasks)} pred meshes from code ...', flush=True)
        worker_args = [(t[1], t[2], (255, 220, 100), 200) for t in pred_tasks]
        pred_res = render_meshes_parallel(worker_args, max_workers=args.workers)
        n_ok = 0
        for cache_file, label, _code, bucket, uid, step in pred_tasks:
            png, _status = pred_res.get(label, (None, '?'))
            if png:
                cache_file.write_bytes(png); n_ok += 1
            # IoU from the same JSONL we read above
            sjsonl = pred_dir / f'step-{step:06d}.jsonl'
            iou = None
            if sjsonl.exists():
                for r in _read_jsonl(sjsonl):
                    if r.get('bucket') == bucket and r['uid'] == uid:
                        iou = r.get('iou'); break
            trajectory[bucket].setdefault(uid, {})[step] = (png, iou)
        print(f'  pred rendered: {n_ok}/{len(pred_tasks)} ok', flush=True)

    if not gt_tasks and not pred_tasks:
        print('all renders cached, skipping render', flush=True)

    # 5. Build per-bucket trajectory collages
    attachments: list[tuple[str, bytes]] = []
    for bucket in IOU_BUCKETS:
        anchors_disp = [{'uid': a['uid'],
                         'input_png': find_input_image(a['uid'], bucket),
                         'gt_mesh_png': gt_pngs[bucket].get(a['uid'])}
                        for a in anchors_by_bucket[bucket]]
        png = build_trajectory_collage(
            bucket, anchors_disp, trajectory[bucket], steps,
            cell=args.cell, gt_w=args.gt_w,
        )
        bname = bucket.replace(' ', '_')
        out_png = pred_dir / f'trajectory_step{args.step:06d}_{bname}.png'
        out_png.write_bytes(png)
        attachments.append((f'traj_step{args.step:06d}_{bname}.png', png))
        print(f'wrote → {out_png}  ({len(png)//1024} KB)', flush=True)

    # 6. Text message
    evals = parse_eval_block(log_path, args.step)
    bc = evals.get('BenchCAD val', {})
    dc = evals.get('DeepCAD test', {})
    fu = evals.get('Fusion360 test', {})

    lines = [f'**🚀 Big-50k SFT — step {args.step}/50000**', '']
    lines.append('**IoU vs curriculum (same step):**')
    lines.append(f'• BenchCAD val:    `{fmt_iou(bc.get("iou"))}`'
                 + fmt_delta(bc.get('iou'), CURR_BC, args.step))
    lines.append(f'• DeepCAD test:    `{fmt_iou(dc.get("iou"))}`'
                 + fmt_delta(dc.get('iou'), CURR_DC, args.step))
    lines.append(f'• Fusion360 test:  `{fmt_iou(fu.get("iou"))}`'
                 + fmt_delta(fu.get('iou'), CURR_FU, args.step))
    lines.append('')
    lines.append('**Op metrics (BenchCAD val):**')
    lines.append(f'• rare_recall (fillet/chamfer/etc): `{fmt_iou(bc.get("rare"))}`')
    lines.append(f'• recall: `{fmt_iou(bc.get("recall"))}`,  '
                 f'op_loss: `{fmt_iou(bc.get("op_loss"))}`')
    lines.append('')
    lines.append(f'trajectory: {len(steps)} steps × {n_anchor} anchors per bucket'
                 f'  ({len(steps) * n_anchor * len(IOU_BUCKETS)} pred cells)')
    content = '\n'.join(lines)

    if args.no_discord:
        print('---no-discord set---'); print(content); return

    statuses = post_to_discord(content=content, files=attachments)
    print(f'discord POST: {statuses}', flush=True)


# ─── SEND MODE ──────────────────────────────────────────────────────────────

def send_main(args):
    files = []
    for fp in args.file or []:
        p = Path(fp)
        if not p.exists():
            print(f'file not found: {p}', file=sys.stderr); sys.exit(2)
        files.append((p.name, p.read_bytes()))
    statuses = post_to_discord(content=args.message or '', files=files)
    print(f'discord POST: {statuses}', flush=True)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    g_send = ap.add_argument_group('send mode')
    g_send.add_argument('--send', action='store_true',
                        help='Generic mode: just send --message + --file(s)')
    g_send.add_argument('--message', default='')
    g_send.add_argument('--file', action='append', default=[])

    g_eval = ap.add_argument_group('eval / trajectory mode')
    g_eval.add_argument('--step', type=int)
    g_eval.add_argument('--log')
    g_eval.add_argument('--output-dir')
    g_eval.add_argument('--anchors-per-bucket', type=int, default=8)
    g_eval.add_argument('--workers', type=int, default=6)
    g_eval.add_argument('--cell', type=int, default=120,
                        help='Per-cell px for pred renders (default: 120)')
    g_eval.add_argument('--gt-w', type=int, default=200,
                        help='Width of GT 4-view col (default: 200)')
    g_eval.add_argument('--no-discord', action='store_true')

    args = ap.parse_args()

    if args.send:
        return send_main(args)

    missing = [n for n in ('step', 'log', 'output_dir')
               if getattr(args, n) is None]
    if missing:
        ap.error(f'eval mode requires: {missing}. Or pass --send for generic mode.')
    eval_main(args)


if __name__ == '__main__':
    main()
