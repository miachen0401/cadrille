"""Per-case grid for cad_bench_722 with op annotations + essential-ops highlighting.

For all 720 cases, builds a multi-page grid with one row per case:
    [#case]  [GT]  [Cadrille-rl]  [CADEvolve v3]  [Q3VL ours]  [Qwen-zs]  [scores]

Each pred cell shows:
    • thumbnail of the rendered geometry
    • IoU + ESS verdict (✓ / ✗ / —)
    • ops actually used by the model, color-coded:
        green  = essential op for this family (AND used)
        red    = essential op for this family (AND **missing** from this pred)
        gray   = op used but not in family's essential spec

Posts every page to Discord.

Usage:
    set -a; source .env; set +a
    uv run python research/repro_official/build_full_grid_v3_ops.py --discord
"""
from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'
REPRO_DIR = REPO / 'eval_outputs/repro_official/cad_bench_722_full' / 'py'
RENDER_CACHE = Path('/tmp/cad_bench_722_renders')
OUT_DIR = EVAL_ROOT / 'full_case_grids_v3_ops'

MODELS = [
    ('cadrille_rl_repro',    'Cadrille-rl'),
    ('cadevolve_rl1',        'CADEvolve v3'),
    ('cadrille_qwen3vl_v3',  'Q3VL (ours)'),
    ('qwen25vl_3b_zs',       'Qwen-zs'),
]
PRED_DIR = {
    'cadrille_rl_repro':    REPRO_DIR,
    'cadevolve_rl1':        EVAL_ROOT / 'cadevolve_rl1',
    'cadrille_qwen3vl_v3':  EVAL_ROOT / 'cadrille_qwen3vl_v3',
    'qwen25vl_3b_zs':       EVAL_ROOT / 'qwen25vl_3b_zs',
}
META_PATH = {
    'cadrille_rl_repro':    REPO / 'eval_outputs/repro_official/cad_bench_722_full/metadata.jsonl',
    'cadevolve_rl1':        EVAL_ROOT / 'cadevolve_rl1' / 'metadata.jsonl',
    'cadrille_qwen3vl_v3':  EVAL_ROOT / 'cadrille_qwen3vl_v3' / 'metadata.jsonl',
    'qwen25vl_3b_zs':       EVAL_ROOT / 'qwen25vl_3b_zs' / 'metadata.jsonl',
}

SIDE     = 256
LABEL_H  = 22       # top label band per cell
ANNOT_H  = 86       # ops annotation band under cell (3 lines)
NUM_W    = 90
PAD      = 4
HEADER_H = 38

# ── render worker ──────────────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout('budget')


_EXEC_TMPL = textwrap.dedent('''\
    import sys
    import cadquery as cq
    import trimesh
    show_object = lambda *a, **kw: None

    {code}

    _r = locals().get("result") or locals().get("r")
    if _r is None: raise ValueError("no result")
    compound = _r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x,v.y,v.z) for v in verts], faces)
    if len(verts) < 4 or len(faces) < 4: raise ValueError("degenerate")
    mesh.export(sys.argv[1])
''')


def _render_4view_pyvista(stl_path, side=268):
    import numpy as np
    import pyvista as pv
    from PIL import Image
    mesh = pv.read(stl_path)
    b = mesh.bounds
    cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
    ext = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
    if ext < 1e-7: ext = 1.0
    mesh = mesh.translate([-cx, -cy, -cz]).scale(1.0 / ext)
    tile = side // 2
    color = (255/255, 255/255, 136/255)
    fronts = [(1, 1, 1), (-1, -1, -1), (-1, 1, -1), (1, -1, 1)]
    tiles = []
    for fx, fy, fz in fronts:
        pl = pv.Plotter(off_screen=True, window_size=(tile, tile))
        pl.background_color = (0.07, 0.07, 0.07)
        pl.add_mesh(mesh, color=color, lighting=True, smooth_shading=True)
        pl.camera_position = [(fx*1.6, fy*1.6, fz*1.6), (0, 0, 0), (0, 0, 1)]
        pl.enable_parallel_projection(); pl.camera.zoom(1.4)
        arr = pl.screenshot(None, return_img=True); pl.close()
        tiles.append(arr)
    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    canvas = np.vstack([top, bot])
    return Image.fromarray(canvas).convert('RGB').resize((side, side), Image.LANCZOS)


def _render_one(args):
    slug, stem, py_path, timeout_sec, cache_dir = args
    cache_key = f'{slug}__{stem}.png'
    if slug == 'cadrille_rl_repro':
        cache_key = f'repro_{cache_key}'
    cache_path = Path(cache_dir) / cache_key
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path), 'error': None}
    if timeout_sec > 0:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout_sec))
    try:
        code = Path(py_path).read_text()
        script = _EXEC_TMPL.format(code=code)
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            stl = f.name
        try:
            r = subprocess.run([sys.executable, '-c', script, stl],
                               capture_output=True, timeout=min(timeout_sec, 30))
            if r.returncode != 0 or Path(stl).stat().st_size < 100:
                return {'slug': slug, 'stem': stem, 'cache_path': None, 'error': 'exec_fail'}
            # Canonical renderer: common.meshio.render_img (see CLAUDE.md
            # "Mesh rendering"). PyPI open3d-cpu 0.18 segfaults on a small
            # fraction of CADEvolve-style preds, so run it in a subprocess
            # and fall back to pyvista on segfault. Source-built open3d
            # from scripts/setup.sh would not segfault.
            tmp_png = stl + '.png'
            render_script = textwrap.dedent(f'''
                import sys
                sys.path.insert(0, {str(REPO)!r})
                from common.meshio import render_img
                img = render_img(sys.argv[1])['video'][0]
                img.save(sys.argv[2])
            ''')
            r2 = subprocess.run([sys.executable, '-c', render_script, stl, tmp_png],
                                capture_output=True, timeout=20)
            if r2.returncode == 0 and Path(tmp_png).exists() and Path(tmp_png).stat().st_size > 0:
                from PIL import Image
                img = Image.open(tmp_png).convert('RGB')
                Path(tmp_png).unlink(missing_ok=True)
            else:
                Path(tmp_png).unlink(missing_ok=True)
                img = _render_4view_pyvista(stl, side=268)
            img.save(cache_path, format='PNG')
            return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path), 'error': None}
        finally:
            try: Path(stl).unlink()
            except Exception: pass
    except Exception as e:
        kind = 'timeout' if isinstance(e, _Timeout) else type(e).__name__
        return {'slug': slug, 'stem': stem, 'cache_path': None, 'error': f'{kind}: {str(e)[:60]}'}
    finally:
        if timeout_sec > 0:
            signal.alarm(0)


# ── drawing helpers ────────────────────────────────────────────────────────

def _font(size, bold=False):
    from PIL import ImageFont
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{name}', size)
    except Exception:
        return ImageFont.load_default()


def _fail_tile(side, msg):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (side, side), color=(28, 28, 28))
    d = ImageDraw.Draw(img)
    f = _font(18, bold=True)
    bbox = d.textbbox((0, 0), msg, font=f)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    d.text(((side - w) / 2, (side - h) / 2), msg, fill=(220, 90, 90), font=f)
    return img


def _draw_wrapped_ops(d, x, y, max_w, ops_color_pairs, font):
    """Draw a list of (op, color) pairs wrapped to max_w. Returns y after last line."""
    line_x = x
    line_y = y
    line_h = 13
    for op, color in ops_color_pairs:
        text = op + ' '
        bbox = d.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        if line_x + w > x + max_w:
            line_y += line_h
            line_x = x
        d.text((line_x, line_y), text, fill=color, font=font)
        line_x += w
    return line_y + line_h


def _annot_cell(img, label, iou, ep, gen_ops, essential_set, family_n_a, es=None):
    """One pred cell: thumbnail + 3-line annotation.

    Annotation:
      line 1: label + IoU + ESS verdict + fractional score (e.g. ESS 1/2)
      line 2-3: ops (green=matched essential, red=missing essential, gray=extra)
    """
    from PIL import Image, ImageDraw
    side = img.width
    canvas = Image.new('RGB', (side, side + LABEL_H + ANNOT_H), color=(18, 18, 18))
    canvas.paste(img, (0, LABEL_H))
    d = ImageDraw.Draw(canvas)
    f_lab = _font(11, bold=True)
    f_val = _font(10)

    # top label band
    d.rectangle([0, 0, side, LABEL_H], fill=(35, 35, 50))
    d.text((4, 3), label, fill=(225, 225, 225), font=f_lab)

    # status band under thumbnail
    y0 = side + LABEL_H
    if ep is True:
        ep_str, ep_color = 'ESS✓', (90, 220, 100)
    elif ep is False:
        ep_str, ep_color = 'ESS✗', (240, 90, 90)
    else:
        ep_str, ep_color = 'ESS—', (160, 160, 160)
    # Fractional ESS score in [0,1] — shown as ratio "k/N" of the
    # k satisfied AND-elements out of N total. None = N/A family.
    if es is not None:
        es_str = f'{es:.2f}'
    else:
        es_str = '—'
    iou_str = f'IoU={iou:.2f}' if iou is not None else 'IoU=—'
    d.text((4, y0 + 2), iou_str, fill=(225, 225, 225), font=f_lab)
    d.text((side - 78, y0 + 2), ep_str, fill=ep_color, font=f_lab)
    d.text((side - 38, y0 + 2), es_str, fill=ep_color, font=f_val)

    # ops list with color coding
    if gen_ops is None:
        d.text((4, y0 + 18), '(no pred)', fill=(160, 160, 160), font=f_val)
        return canvas

    # build (op, color) pairs
    color_used_essential = (90, 220, 100)    # green
    color_missing_essential = (240, 90, 90)  # red
    color_extra = (170, 170, 170)            # gray
    pairs = []
    used = set(gen_ops)
    for op in sorted(used):
        if op in essential_set:
            pairs.append((op, color_used_essential))
        else:
            pairs.append((op, color_extra))
    # missing essential ops at the end
    for op in sorted(essential_set - used):
        pairs.append(('!' + op, color_missing_essential))
    if not pairs and not family_n_a:
        d.text((4, y0 + 18), '(no ops)', fill=(160, 160, 160), font=f_val)
    else:
        _draw_wrapped_ops(d, 4, y0 + 18, side - 8, pairs, f_val)
    return canvas


def _number_cell(idx, family, diff, width, height):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(20, 20, 20))
    d = ImageDraw.Draw(img)
    fbig = _font(22, bold=True)
    fmid = _font(10, bold=True)
    fsm  = _font(9)
    s = f'#{idx}'
    bbox = d.textbbox((0, 0), s, font=fbig)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 6), s, fill=(240, 240, 240), font=fbig)
    fam = (family or '?')[:14]
    bbox = d.textbbox((0, 0), fam, font=fmid)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 38), fam, fill=(255, 200, 100), font=fmid)
    di = f'[{diff or "?"}]'
    bbox = d.textbbox((0, 0), di, font=fsm)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 58), di, fill=(150, 150, 150), font=fsm)
    return img


def _page_header(width, height, title):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(35, 35, 50))
    d = ImageDraw.Draw(img)
    f = _font(14, bold=True)
    d.text((10, 10), title, fill=(230, 230, 230), font=f)
    return img


# ── Discord ────────────────────────────────────────────────────────────────

def _post(content, attachment):
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        return False
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def w(s): body.write(s.encode())
    w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n')
    w('Content-Type: application/json\r\n\r\n')
    w(json.dumps({'content': content}) + '\r\n')
    w(f'--{boundary}\r\nContent-Disposition: form-data; '
      f'name="file"; filename="{attachment.name}"\r\n')
    w('Content-Type: image/png\r\n\r\n')
    body.write(attachment.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    req = urllib.request.Request(url, data=body.getvalue(), headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cad-fullgrid-v3-ops/1.0',
    })
    try:
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except Exception as e:
        print(f'  Discord failed: {e}')
        return False


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-chunks', type=int, default=15)
    ap.add_argument('--workers',  type=int, default=4)
    ap.add_argument('--task-timeout', type=int, default=30)
    ap.add_argument('--limit',    type=int, default=0,
                    help='Cap total cases for smoke testing')
    ap.add_argument('--per-family', type=int, default=0,
                    help='If >0, pick this many representative cases per family '
                         '(sorted by family sample count desc) instead of all 720')
    ap.add_argument('--split-by-iou', type=float, default=0,
                    help='If >0, split cases into 2 groups by Q3VL IoU '
                         '(threshold). Generates _high.png and _low.png suffixes.')
    ap.add_argument('--no-render-preds', action='store_true',
                    help='Skip rendering pred meshes; show only GT image + '
                         'per-model text annotations (ops, IoU, ESS).')
    ap.add_argument('--out-dir',  default=str(OUT_DIR))
    ap.add_argument('--discord',  action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Load metadata + essential_ops ──────────────────────────────────────
    print('Loading metadata + essential_ops …', flush=True)
    metas = {}
    for slug, _ in MODELS:
        d = {}
        with open(META_PATH[slug]) as f:
            for line in f:
                try: r = json.loads(line); d[r['stem']] = r
                except Exception: pass
        metas[slug] = d
        print(f'  {slug}: {len(d)} samples')

    ess_raw = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    ess_per_case = {slug: {c['stem']: c for c in m['per_case']}
                    for slug, m in ess_raw['models'].items()}

    # canonical_ops.ESSENTIAL_BY_FAMILY for the essential sets
    from common.essential_ops import ESSENTIAL_BY_FAMILY
    def family_essential_set(fam):
        spec = ESSENTIAL_BY_FAMILY.get(fam)
        if not spec:
            return set()
        out = set()
        for elem in spec:
            if isinstance(elem, str):
                out.add(elem)
            else:
                out.update(elem)
        return out

    # ── Sorted stems (or per-family representatives) ───────────────────────
    if args.per_family > 0:
        # Per-family canonical case = both-exec-ok, Q3VL IoU closest to
        # family median (alphabetical tiebreak). Same rule used by
        # build_cadevolve_weak_families.py and build_low_ess_families.py.
        from collections import defaultdict
        by_fam = defaultdict(list)
        for stem, q in metas['cadrille_qwen3vl_v3'].items():
            ce = metas['cadevolve_rl1'].get(stem) or {}
            if (q.get('error_type') != 'success' or ce.get('error_type') != 'success'
                    or q.get('iou') is None or ce.get('iou') is None
                    or not q.get('family')):
                continue
            by_fam[q['family']].append((stem, q['iou']))
        chosen = []
        for fam, items in by_fam.items():
            items.sort(key=lambda t: t[1])
            mid = items[len(items)//2][1]
            items.sort(key=lambda t: (abs(t[1] - mid), t[0]))
            chosen.append(items[0][0])
        all_stems = sorted(set(chosen),
                           key=lambda s: ((metas['cadrille_qwen3vl_v3'].get(s)
                                           or metas['cadevolve_rl1'].get(s)
                                           or {}).get('family', '~'), s))
        print(f'  per-family canonical: {len(all_stems)} stems '
              f'across {len(by_fam)} families', flush=True)
    else:
        all_stems = sorted(set().union(*[set(m.keys()) for m in metas.values()]))
    if args.limit:
        all_stems = all_stems[:args.limit]
    n_cases = len(all_stems)
    print(f'  total cases: {n_cases}', flush=True)

    # ── Optional split by Q3VL IoU into high / low groups ──────────────────
    iou_split_groups = None  # None = single group; otherwise [(label, [stems])]
    if args.split_by_iou > 0:
        thr = args.split_by_iou
        high, low = [], []
        for stem in all_stems:
            q = metas['cadrille_qwen3vl_v3'].get(stem) or {}
            iou = q.get('iou')
            # Sort failed-exec / no-iou into the LOW bucket so they're visible
            if iou is not None and iou >= thr:
                high.append(stem)
            else:
                low.append(stem)
        iou_split_groups = [(f'high_iou_ge{thr:.2f}', high),
                            (f'low_iou_lt{thr:.2f}',  low)]
        print(f'  split @ Q3VL IoU={thr}: high={len(high)}, low={len(low)}',
              flush=True)

    # ── GT composite_png ────────────────────────────────────────────────────
    print('Fetching GT composite_png …', flush=True)
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    gt_by_stem = {row['stem']: row['composite_png'] for row in ds
                  if row['stem'] in set(all_stems)}
    print(f'  loaded {len(gt_by_stem)} GT images', flush=True)

    # ── Render preds (with cache) ──────────────────────────────────────────
    # `--no-render-preds`: skip rendering entirely, just show GT image + per-model
    # text annotations. Much faster (no exec + no render); useful when the user
    # only cares about ops + scores, not the visual reconstruction.
    if args.no_render_preds:
        print('--no-render-preds: skipping pred rendering', flush=True)
    print('Rendering preds …' if not args.no_render_preds else '', flush=True)
    tasks = []
    if not args.no_render_preds:
        for slug, _ in MODELS:
            for stem in all_stems:
                rec = metas[slug].get(stem) or {}
                if rec.get('error_type') != 'success':
                    continue
                py_path = PRED_DIR[slug] / f'{stem}.py'
                if not py_path.exists():
                    continue
                tasks.append((slug, stem, str(py_path), args.task_timeout,
                              str(RENDER_CACHE)))
    pending = []
    for t in tasks:
        slug, stem = t[0], t[1]
        ck = (f'repro_{slug}__{stem}.png' if slug == 'cadrille_rl_repro'
              else f'{slug}__{stem}.png')
        if not (RENDER_CACHE / ck).exists():
            pending.append(t)
    print(f'  {len(tasks)} render tasks, {len(pending)} not yet cached', flush=True)
    if pending:
        t0 = time.time(); done = 0
        with ProcessPoolExecutor(max_workers=args.workers,
                                 max_tasks_per_child=80) as pool:
            futs = {pool.submit(_render_one, t): (t[0], t[1]) for t in pending}
            for fut in as_completed(futs):
                try:
                    res = fut.result(timeout=args.task_timeout * 2)
                except Exception as e:
                    res = {'slug': '?', 'stem': '?', 'error': str(e)}
                done += 1
                if done % 50 == 0:
                    rate = done / (time.time() - t0 + 1e-6)
                    eta = (len(pending) - done) / max(rate, 1e-6) / 60
                    print(f'    [{done}/{len(pending)}] {rate:.2f}/s ETA {eta:.1f}min',
                          flush=True)
        print(f'  render done in {(time.time()-t0)/60:.1f}min', flush=True)

    # ── Build pages ─────────────────────────────────────────────────────────
    from PIL import Image
    cell_h = LABEL_H + SIDE + ANNOT_H
    page_w = NUM_W + (1 + len(MODELS)) * (SIDE + PAD)
    out_paths = []

    # If --split-by-iou, build separate page sets for each group.
    # Otherwise treat the full all_stems as one group called 'all'.
    groups = iou_split_groups if iou_split_groups else [('all', all_stems)]

    for group_label, group_stems in groups:
        n_g = len(group_stems)
        if n_g == 0: continue
        chunk = max(1, (n_g + args.n_chunks - 1) // args.n_chunks)
        suffix = '' if group_label == 'all' else f'_{group_label}'
        print(f'\nBuilding {args.n_chunks} pages for group "{group_label}" '
              f'({n_g} cases) …', flush=True)
        for ci in range(args.n_chunks):
            lo = ci * chunk
            hi = min(lo + chunk, n_g)
            if lo >= hi: break
            chunk_stems = group_stems[lo:hi]
            page_h = HEADER_H + len(chunk_stems) * (cell_h + PAD)
            page = Image.new('RGB', (page_w, page_h), color=(10, 10, 10))
            page.paste(_page_header(page_w, HEADER_H,
                f'cad_bench_722 — group "{group_label}" — page {ci+1}/{args.n_chunks} '
                f'(cases {lo+1}..{hi} of {n_g})  '
                f'cols: # | GT | ' + ' | '.join(label for _, label in MODELS) +
                '   ops: green=essential matched, !red=essential missing, gray=extra'),
                (0, 0))
            for ri, stem in enumerate(chunk_stems):
                y = HEADER_H + ri * (cell_h + PAD)
                mref = next((metas[s].get(stem, {}) for s, _ in MODELS
                             if metas[s].get(stem) and metas[s][stem].get('family')), {})
                family = mref.get('family')
                diff = mref.get('difficulty')
                page.paste(_number_cell(lo + ri + 1, family, diff, NUM_W, cell_h), (0, y))
                x = NUM_W + PAD
                # GT
                gt = gt_by_stem.get(stem)
                if gt is not None:
                    gt_img = gt.convert('RGB').resize((SIDE, SIDE), Image.LANCZOS)
                    ess_set = family_essential_set(family) if family else set()
                    page.paste(_annot_cell(gt_img, 'GT', None, None,
                                           sorted(ess_set), ess_set, False),
                               (x, y))
                else:
                    page.paste(_annot_cell(_fail_tile(SIDE, 'NO GT'), 'GT',
                                           None, None, [], set(), False), (x, y))
                x += SIDE + PAD

                # 4 model columns
                ess_set = family_essential_set(family) if family else set()
                family_n_a = (family is None or not ess_set)
                for slug, label in MODELS:
                    rec = metas[slug].get(stem, {}) or {}
                    ec  = ess_per_case.get(slug, {}).get(stem, {}) or {}
                    iou = rec.get('iou')
                    ep  = ec.get('essential_pass')
                    es  = ec.get('essential_score')
                    gen_ops = ec.get('gen_ops')
                    if args.no_render_preds:
                        # Text-only cell — black thumbnail with model label, all
                        # info goes into the annotation band. Saves ~2.5s per cell.
                        img = Image.new('RGB', (SIDE, SIDE), color=(18, 18, 18))
                    else:
                        ck = (f'repro_{slug}__{stem}.png' if slug == 'cadrille_rl_repro'
                              else f'{slug}__{stem}.png')
                        cache_path = RENDER_CACHE / ck
                        if cache_path.exists() and cache_path.stat().st_size > 0:
                            img = Image.open(cache_path).convert('RGB').resize(
                                (SIDE, SIDE), Image.LANCZOS)
                        else:
                            et = rec.get('error_type', 'no pred')
                            img = _fail_tile(SIDE, et.upper())
                    page.paste(_annot_cell(img, label, iou, ep, gen_ops,
                                           ess_set, family_n_a, es=es), (x, y))
                    x += SIDE + PAD

            path = out_dir / f'cases{suffix}_{lo+1:04d}-{hi:04d}.png'
            page.save(path, optimize=True)
            sz = path.stat().st_size / 1024 / 1024
            print(f'  page {ci+1}/{args.n_chunks}: {hi-lo} cases  '
                  f'{page.size[0]}×{page.size[1]}  {sz:.1f}MB → {path.name}',
                  flush=True)
            out_paths.append((path, lo + 1, hi))

    # ── Discord ────────────────────────────────────────────────────────────
    if args.discord:
        print('\nPosting to Discord …', flush=True)
        for i, (p, lo, hi) in enumerate(out_paths):
            desc = (
                f'📦 **cad_bench_722 (CADEvolve v3) — case grid w/ ops**\n'
                f'`{p.name}` ({lo}–{hi})\n'
                f'cols: case# | GT | Cadrille-rl | CADEvolve v3 | Q3VL (ours) | Qwen-zs\n'
                f'each cell: IoU + ESS verdict + ops used. '
                f'**green**=essential matched, **!red**=missing, **gray**=extra'
            )
            ok = _post(desc, p)
            print(f'  page {i+1}/{len(out_paths)} ({p.name}) → '
                  f'{"sent" if ok else "FAILED"}', flush=True)
            time.sleep(2)

    print('\nAll done.')


if __name__ == '__main__':
    main()
