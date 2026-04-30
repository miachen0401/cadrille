"""IoU vs essential-ops comparison grid for cad_bench_722.

Picks 12 representative cases that highlight the divergence between
geometric IoU and op-level essential-pass / feature-F1, renders one
panel per (case × model), annotates each cell with IoU + essential
verdict + F1, and posts the grid + analysis markdown to Discord.

Models compared (4 cols):
  - Cadrille-rl (paper repro 4.50.3)   — col `cadrille_rl_repro`
  - CADEvolve-rl1 (kulibinai)          — col `cadevolve_rl1`
  - Cadrille-Q3VL-v3 (ours)            — col `cadrille_qwen3vl_v3`
  - Qwen2.5-VL-3B (zero-shot baseline) — col `qwen25vl_3b_zs`

Categories selected (3 each):
  A. Cadrille-Q3VL-v3 wins on both IoU and essential
  B. Cadrille-Q3VL-v3 passes essential but IoU still low
     (right ops, wrong geometry — diagnostic for SFT)
  C. CADEvolve has decent IoU but fails essential
     (visual match, ops mismatched — diagnostic for surface-level fits)
  D. Qwen-zs passes essential by accident
     (sample-size bias — only 14 applicable preds)

Usage:
    set -a; source .env; set +a
    uv run python research/3d_similarity/iou_vs_essential_grid.py --discord
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

EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'
REPRO_ROOT = REPO / 'eval_outputs' / 'repro_official' / 'cad_bench_722_full'
RENDER_CACHE = Path('/tmp/cad_bench_722_renders')
OUT_DIR = EVAL_ROOT / 'iou_vs_essential'

MODELS = [
    ('cadrille_rl_repro',    'Cadrille-rl (4.50.3 repro)'),
    ('cadevolve_rl1',        'CADEvolve-rl1'),
    ('cadrille_qwen3vl_v3',  'Cadrille-Q3VL-v3 (ours)'),
    ('qwen25vl_3b_zs',       'Qwen2.5-VL-3B (zero-shot)'),
]
PRED_DIR = {
    'cadrille_rl_repro':    REPRO_ROOT / 'py',
    'cadevolve_rl1':        EVAL_ROOT / 'cadevolve_rl1',
    'cadrille_qwen3vl_v3':  EVAL_ROOT / 'cadrille_qwen3vl_v3',
    'qwen25vl_3b_zs':       EVAL_ROOT / 'qwen25vl_3b_zs',
}
META_PATH = {
    'cadrille_rl_repro':    REPRO_ROOT / 'metadata.jsonl',
    'cadevolve_rl1':        EVAL_ROOT / 'cadevolve_rl1' / 'metadata.jsonl',
    'cadrille_qwen3vl_v3':  EVAL_ROOT / 'cadrille_qwen3vl_v3' / 'metadata.jsonl',
    'qwen25vl_3b_zs':       EVAL_ROOT / 'qwen25vl_3b_zs' / 'metadata.jsonl',
}

SIDE = 256
LABEL_H = 28
NUM_W = 110
ANNOT_H = 50            # per-cell IoU + essential + F1 line under cell
PAD = 4
HEADER_H = 60


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


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout('budget')


def _render_4view_pyvista(stl_path: str, side: int = 268):
    """Pyvista-based 4-view render — fallback when open3d isn't installed.

    Produces a 268×268 RGB PIL image: 2×2 grid of orthographic-ish views
    of the unit-normalized mesh in a yellow color, dark background.
    """
    import numpy as np
    import pyvista as pv
    from PIL import Image
    mesh = pv.read(stl_path)
    b = mesh.bounds
    cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
    ext = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
    if ext < 1e-7: ext = 1.0
    mesh = mesh.translate([-cx, -cy, -cz]).scale(1.0 / ext)
    tile = side // 2  # 134
    color = (255/255, 255/255, 136/255)
    fronts = [(1, 1, 1), (-1, -1, -1), (-1, 1, -1), (1, -1, 1)]
    tiles = []
    for fx, fy, fz in fronts:
        pl = pv.Plotter(off_screen=True, window_size=(tile, tile))
        pl.background_color = (0.07, 0.07, 0.07)
        pl.add_mesh(mesh, color=color, lighting=True, smooth_shading=True)
        pl.camera_position = [(fx*1.6, fy*1.6, fz*1.6), (0, 0, 0), (0, 0, 1)]
        pl.enable_parallel_projection()
        pl.camera.zoom(1.4)
        arr = pl.screenshot(None, return_img=True)
        pl.close()
        tiles.append(arr)
    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    canvas = np.vstack([top, bot])
    return Image.fromarray(canvas).convert('RGB').resize((side, side), Image.LANCZOS)


def _render_one(args) -> dict:
    slug, stem, py_path, timeout_sec, cache_dir = args
    # Use a `repro_` prefix in the cache to avoid clashing with the
    # broken cadrille_rl renders already in the shared cache.
    cache_key = f'{slug}__{stem}.png'
    if slug == 'cadrille_rl_repro':
        cache_key = f'repro_{cache_key}'
    cache_path = Path(cache_dir) / cache_key
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path),
                'error': None}
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
                               capture_output=True,
                               timeout=min(timeout_sec, 30))
            if r.returncode != 0 or Path(stl).stat().st_size < 100:
                return {'slug': slug, 'stem': stem, 'cache_path': None,
                        'error': 'exec_fail'}
            try:
                from common.meshio import render_img
                img = render_img(stl)['video'][0]
            except (ImportError, ModuleNotFoundError):
                # open3d not installed in this venv — pyvista fallback
                img = _render_4view_pyvista(stl, side=268)
            img.save(cache_path, format='PNG')
            return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path),
                    'error': None}
        finally:
            try: Path(stl).unlink()
            except Exception: pass
    except Exception as e:
        kind = 'timeout' if isinstance(e, _Timeout) else type(e).__name__
        return {'slug': slug, 'stem': stem, 'cache_path': None,
                'error': f'{kind}: {str(e)[:60]}'}
    finally:
        if timeout_sec > 0:
            signal.alarm(0)


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
    f = _font(16, bold=True)
    bbox = d.textbbox((0, 0), msg, font=f)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    d.text(((side - w) / 2, (side - h) / 2), msg, fill=(220, 90, 90), font=f)
    return img


def _annot_cell(img, top_label, iou, ep, f1):
    """Cell with thumbnail on top, IoU/essential/F1 annotation under."""
    from PIL import Image, ImageDraw
    side = img.width
    canvas = Image.new('RGB', (side, side + ANNOT_H), color=(18, 18, 18))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    f_lab = _font(11, bold=True)
    f_val = _font(11)

    d.text((4, side + 2), top_label, fill=(220, 220, 220), font=f_lab)

    # essential pass color: green=pass, red=fail, gray=N/A or no pred
    if ep is True:
        ep_str, ep_color = 'ESS:✓', (90, 220, 100)
    elif ep is False:
        ep_str, ep_color = 'ESS:✗', (240, 90, 90)
    else:
        ep_str, ep_color = 'ESS:—', (160, 160, 160)

    iou_str = f'IoU={iou:.3f}' if iou is not None else 'IoU=—'
    f1_str  = f'F1={f1:.2f}' if f1 is not None else 'F1=—'
    line2 = f'{iou_str}  {f1_str}'
    d.text((4, side + 18), line2, fill=(220, 220, 220), font=f_val)
    d.text((4, side + 34), ep_str, fill=ep_color, font=f_lab)
    return canvas


def _number_cell(idx, label, family, diff, width, height):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(20, 20, 20))
    d = ImageDraw.Draw(img)
    fbig = _font(22, bold=True)
    fmid = _font(11, bold=True)
    fsm  = _font(10)
    s = f'#{idx}'
    bbox = d.textbbox((0, 0), s, font=fbig)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 6), s, fill=(240, 240, 240), font=fbig)
    bbox = d.textbbox((0, 0), label, font=fmid)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 38), label, fill=(255, 200, 100), font=fmid)
    fam = family[:14]
    bbox = d.textbbox((0, 0), fam, font=fsm)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 56), fam, fill=(180, 180, 180), font=fsm)
    di = f'[{diff}]'
    bbox = d.textbbox((0, 0), di, font=fsm)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 72), di, fill=(150, 150, 150), font=fsm)
    return img


def _page_header(width, height, title, subtitle):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(35, 35, 50))
    d = ImageDraw.Draw(img)
    f1 = _font(17, bold=True)
    f2 = _font(12)
    d.text((10, 8), title, fill=(230, 230, 230), font=f1)
    d.text((10, 32), subtitle, fill=(180, 200, 230), font=f2)
    return img


def post_to_discord(content, attachments):
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  (no DISCORD_WEBHOOK_URL)'); return False
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def w(s): body.write(s.encode())
    w(f'--{boundary}\r\n'
      f'Content-Disposition: form-data; name="payload_json"\r\n'
      f'Content-Type: application/json\r\n\r\n')
    w(json.dumps({'content': content}) + '\r\n')
    for i, p in enumerate(attachments):
        ct = 'image/png' if p.suffix.lower() == '.png' else 'text/markdown'
        w(f'--{boundary}\r\n'
          f'Content-Disposition: form-data; name="files[{i}]"; filename="{p.name}"\r\n'
          f'Content-Type: {ct}\r\n\r\n')
        body.write(p.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    req = urllib.request.Request(url, data=body.getvalue(), headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cad-iou-vs-essential/1.0',
    })
    try:
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except Exception as e:
        print(f'  Discord post failed: {e}')
        return False


def _load_metas():
    metas = {}
    for slug, _ in MODELS:
        d = {}
        with open(META_PATH[slug]) as f:
            for line in f:
                try:
                    r = json.loads(line); d[r['stem']] = r
                except Exception:
                    pass
        metas[slug] = d
    return metas


def _load_essential():
    raw = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    out = {}
    for slug, m in raw['models'].items():
        out[slug] = {c['stem']: c for c in m['per_case']}
    return out, raw


def _select_cases(metas, ess, n_per_cat=3):
    """Pick N per category that demonstrate IoU vs essential divergence."""
    # All cases that have predictions from BOTH cadrille_qwen3vl_v3 and
    # at least one of {cadrille_rl_repro, cadevolve_rl1}.
    all_stems = set(ess.get('cadrille_qwen3vl_v3', {}).keys())
    selected = {'A_ours_wins': [], 'B_ess_pass_low_iou': [],
                'C_high_iou_ess_fail': [], 'D_qwen_zs_pass': []}

    for stem in all_stems:
        ours = ess['cadrille_qwen3vl_v3'].get(stem) or {}
        rl   = ess['cadrille_rl_repro'].get(stem) or {}
        ce   = ess['cadevolve_rl1'].get(stem) or {}
        qz   = ess['qwen25vl_3b_zs'].get(stem) or {}
        # IoU pulled from the per-model metadata.jsonl
        ours_iou = (metas['cadrille_qwen3vl_v3'].get(stem) or {}).get('iou')
        ce_iou   = (metas['cadevolve_rl1'].get(stem) or {}).get('iou')
        rl_iou   = (metas['cadrille_rl_repro'].get(stem) or {}).get('iou')

        # A: ours passes essential AND ours IoU >= 0.6 AND others (rl,ce) IoU < 0.4
        if (ours.get('essential_pass') is True
                and ours_iou is not None and ours_iou >= 0.6
                and (rl_iou is None or rl_iou < 0.4)
                and (ce_iou is None or ce_iou < 0.4)):
            selected['A_ours_wins'].append((stem, 'A', ours_iou or 0))
        # B: ours passes essential but IoU still low
        if (ours.get('essential_pass') is True
                and ours_iou is not None and ours_iou < 0.4):
            selected['B_ess_pass_low_iou'].append((stem, 'B', ours_iou))
        # C: cadevolve has decent IoU >= 0.4 but fails essential
        if (ce.get('essential_pass') is False
                and ce_iou is not None and ce_iou >= 0.4):
            selected['C_high_iou_ess_fail'].append((stem, 'C', ce_iou))
        # D: qwen-zs has applicable family AND essential_pass==True
        if qz.get('essential_pass') is True:
            selected['D_qwen_zs_pass'].append((stem, 'D', ours_iou or 0))

    # Sort + cap
    cases = []
    for cat, lst in selected.items():
        # Sort by primary descending IoU for A/C, ascending for B (most divergent), random for D
        if cat == 'B_ess_pass_low_iou':
            lst.sort(key=lambda t: t[2])    # lowest IoU first
        else:
            lst.sort(key=lambda t: -t[2])   # highest first
        cases.extend([(s, cat) for s, _, _ in lst[:n_per_cat]])
    return cases


def _category_label(cat):
    return {
        'A_ours_wins': 'A: ours wins',
        'B_ess_pass_low_iou': 'B: ESS✓ low IoU',
        'C_high_iou_ess_fail': 'C: high IoU ESS✗',
        'D_qwen_zs_pass': 'D: Qwen-zs ESS✓',
    }[cat]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-cat',  type=int, default=3)
    ap.add_argument('--workers',    type=int, default=4)
    ap.add_argument('--task-timeout', type=int, default=30)
    ap.add_argument('--out-dir',    default=str(OUT_DIR))
    ap.add_argument('--discord',    action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)

    print('Loading metadata + essential_ops …', flush=True)
    metas = _load_metas()
    for slug, _ in MODELS:
        print(f'  {slug}: {len(metas[slug])} samples', flush=True)
    ess, ess_raw = _load_essential()

    print('\nSelecting representative cases …', flush=True)
    cases = _select_cases(metas, ess, n_per_cat=args.n_per_cat)
    print(f'  {len(cases)} cases selected', flush=True)
    for stem, cat in cases:
        oi = (metas['cadrille_qwen3vl_v3'].get(stem) or {}).get('iou')
        print(f'    [{cat}] {stem:50}  ours_iou={oi}', flush=True)

    # Render each case for each model
    print('\nRendering predictions …', flush=True)
    tasks = []
    for stem, _ in cases:
        for slug, _ in MODELS:
            py = PRED_DIR[slug] / f'{stem}.py'
            if not py.exists():
                continue
            if (metas[slug].get(stem) or {}).get('error_type') != 'success':
                continue
            tasks.append((slug, stem, str(py), args.task_timeout,
                          str(RENDER_CACHE)))

    pending = []
    for t in tasks:
        slug, stem = t[0], t[1]
        cache_key = f'{slug}__{stem}.png'
        if slug == 'cadrille_rl_repro':
            cache_key = f'repro_{cache_key}'
        if not (RENDER_CACHE / cache_key).exists():
            pending.append(t)
    print(f'  {len(tasks)} render tasks, {len(pending)} not yet cached',
          flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 max_tasks_per_child=10) as pool:
            futs = {pool.submit(_render_one, t): (t[0], t[1]) for t in pending}
            for fut in as_completed(futs, timeout=600):
                try:
                    res = fut.result(timeout=args.task_timeout * 2)
                    print(f'    {res["slug"]:22} {res["stem"]:45} '
                          f'{"OK" if res["error"] is None else res["error"][:30]}',
                          flush=True)
                except Exception as e:
                    print(f'    !! {e}', flush=True)

    # Build the figure
    print('\nBuilding figure …', flush=True)
    from PIL import Image
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    gt_by_stem = {row['stem']: row['composite_png'] for row in ds
                  if row['stem'] in {s for s, _ in cases}}

    cell_h = SIDE + ANNOT_H
    page_w = NUM_W + (1 + len(MODELS)) * (SIDE + PAD)
    page_h = HEADER_H + len(cases) * (cell_h + PAD)
    page = Image.new('RGB', (page_w, page_h), color=(10, 10, 10))

    # header
    title = ('cad_bench_722  —  IoU vs essential-ops  —  '
             '12 representative cases (4 categories × 3)')
    sub = ('cols: # | GT | Cadrille-rl(4.50.3) | CADEvolve-rl1 | '
           'Cadrille-Q3VL-v3 (ours) | Qwen2.5-VL-3B (zs)   |   '
           'ESS:✓=essential ops match, ✗=miss, —=N/A or no pred')
    page.paste(_page_header(page_w, HEADER_H, title, sub), (0, 0))

    rows_md = []
    for ri, (stem, cat) in enumerate(cases):
        y = HEADER_H + ri * (cell_h + PAD)
        # case meta
        mref = next((metas[s].get(stem, {}) for s, _ in MODELS
                     if metas[s].get(stem) and metas[s][stem].get('family')), {})
        family = mref.get('family') or '?'
        diff   = mref.get('difficulty') or '?'
        page.paste(_number_cell(ri + 1, _category_label(cat), family, diff,
                                NUM_W, cell_h), (0, y))
        # GT
        x = NUM_W + PAD
        gt = gt_by_stem.get(stem)
        if gt is not None:
            gt_img = gt.convert('RGB').resize((SIDE, SIDE), Image.LANCZOS)
            page.paste(_annot_cell(gt_img, 'GT', None, None, None), (x, y))
        else:
            page.paste(_annot_cell(_fail_tile(SIDE, 'NO GT'), 'GT',
                                   None, None, None), (x, y))
        x += SIDE + PAD
        # 4 model columns
        for slug, label in MODELS:
            cache_key = (f'repro_{slug}__{stem}.png'
                         if slug == 'cadrille_rl_repro'
                         else f'{slug}__{stem}.png')
            cache_path = RENDER_CACHE / cache_key
            iou = (metas[slug].get(stem) or {}).get('iou')
            ec  = (ess[slug].get(stem) or {})
            ep  = ec.get('essential_pass')
            f1  = ec.get('feature_f1')
            top_label = label[:24]
            if cache_path.exists() and cache_path.stat().st_size > 0:
                img = Image.open(cache_path).convert('RGB').resize(
                    (SIDE, SIDE), Image.LANCZOS)
                page.paste(_annot_cell(img, top_label, iou, ep, f1), (x, y))
            else:
                et = (metas[slug].get(stem) or {}).get('error_type', 'no pred')
                page.paste(_annot_cell(_fail_tile(SIDE, et.upper()),
                                       top_label, iou, ep, f1), (x, y))
            x += SIDE + PAD
        rows_md.append({'idx': ri + 1, 'stem': stem, 'cat': cat,
                        'family': family, 'diff': diff,
                        'records': {s: {'iou': (metas[s].get(stem) or {}).get('iou'),
                                        'ep':  (ess[s].get(stem) or {}).get('essential_pass'),
                                        'f1':  (ess[s].get(stem) or {}).get('feature_f1'),
                                        'gen_ops': (ess[s].get(stem) or {}).get('gen_ops'),
                                        'gt_ops':  (ess[s].get(stem) or {}).get('gt_ops')}
                                    for s, _ in MODELS}})

    grid_path = out_dir / 'iou_vs_essential_grid.png'
    page.save(grid_path, optimize=True)
    sz = grid_path.stat().st_size / 1024 / 1024
    print(f'  → {grid_path.name}  {page.size[0]}×{page.size[1]}  {sz:.2f} MB',
          flush=True)

    # Markdown analysis
    print('\nWriting analysis markdown …', flush=True)
    md = ['# IoU vs essential-ops — 12-case comparison',
          '',
          'Source: `eval_outputs/cad_bench_722/essential_ops.json` + per-model '
          '`metadata.jsonl`. Predictions: `cadrille_rl_repro` from '
          '`eval_outputs/repro_official/cad_bench_722_full/py/`; the other '
          'three from `eval_outputs/cad_bench_722/<model>/`.',
          '',
          '## Overall numbers (refresher)',
          '',
          '| model                      | exec | mean IoU | ess_pass | mean F1 |',
          '|----------------------------|------|----------|----------|---------|']
    for slug, label in MODELS:
        m = ess_raw['models'][slug]
        rec_iou = [r.get('iou') for r in metas[slug].values()
                   if r.get('error_type') == 'success' and r.get('iou') is not None]
        miou = sum(rec_iou) / len(rec_iou) if rec_iou else float('nan')
        n_app = m['n_pass'] + m['n_fail']
        ep = (f'{m["pct_essential_pass"]*100:.1f}% ({m["n_pass"]}/{n_app})'
              if n_app else '—')
        ff = f'{m["mean_feature_f1"]:.3f}' if m['mean_feature_f1'] is not None else '—'
        md.append(f'| {label:<26} | {m["n_with_pred"]:>4} | '
                  f'{miou:.3f}    | {ep:<8} | {ff:>7} |')

    md += ['',
           '## Why the metric numbers diverge',
           '',
           '**Sample-size bias on Qwen2.5-VL-3B-zs.** The model only produces '
           f'{ess_raw["models"]["qwen25vl_3b_zs"]["n_with_pred"]} executable '
           'predictions out of 720 (≈2 %), of which only '
           f'{ess_raw["models"]["qwen25vl_3b_zs"]["n_pass"] + ess_raw["models"]["qwen25vl_3b_zs"]["n_fail"]}'
           ' fall in a family that has an essential spec. Reporting '
           f'{ess_raw["models"]["qwen25vl_3b_zs"]["n_pass"]}/'
           f'{ess_raw["models"]["qwen25vl_3b_zs"]["n_pass"] + ess_raw["models"]["qwen25vl_3b_zs"]["n_fail"]}'
           ' as a percentage hides the fact that the denominator is two '
           'orders of magnitude smaller than for the trained models. The '
           'fairer headline is the *absolute* pass count or '
           '`n_pass / 720` (coverage-weighted): '
           f'{ess_raw["models"]["qwen25vl_3b_zs"]["n_pass"]}/720 = '
           f'{ess_raw["models"]["qwen25vl_3b_zs"]["n_pass"]/720*100:.2f}%, '
           f'vs Cadrille-Q3VL-v3 = '
           f'{ess_raw["models"]["cadrille_qwen3vl_v3"]["n_pass"]}/720 = '
           f'{ess_raw["models"]["cadrille_qwen3vl_v3"]["n_pass"]/720*100:.1f}%.',
           '',
           '**Cadrille-rl (paper repro) shows 0 % essential pass.** That is *not* '
           'noise — the public 4.50.3 RL checkpoint is trained on '
           'point-cloud-conditioned compact-style code (`r = (cq.Workplane()'
           '.box(...)... .extrude(...))`), almost never using the '
           'BenchCAD-style ops (`sweep / revolve / loft / shell / polarArray '
           '/ rarray / makeTorus / sphere / polyline / spline / Sketch / '
           'polygon`). It can still match BenchCAD geometry by piecewise '
           'extrusion, hence non-zero IoU on cad_bench_722, but the op-level '
           'metric correctly reports that the *vocabulary* is wrong.',
           '',
           '**CADEvolve-rl1 — see the “open setup bug” section below**; until '
           'we re-run with the correct prompt/processor/max_new_tokens, the '
           '0.367 IoU here is an under-estimate, not a true ceiling.',
           '',
           '## CADEvolve setup bug (the 0.367 IoU is suspect)',
           '',
           'The official inference script '
           '(`zhemdi/CADEvolve/train/inference.py`, vendored as '
           '`research/repro_official/cadevolve_inference.py`) differs from '
           'our `research/repro_official/run_cadevolve.py` in three ways '
           'that all hurt IoU:',
           '',
           '1. **Prompt.** Official feeds *image only* '
           '(`{role:user, content:[{image:img}]}`). Our code adds a text '
           'prompt `"Generate CadQuery Python code for this 3D CAD model '
           'shown in multiple views."` — out-of-distribution for an SFT/RL '
           'checkpoint that never saw a textual prompt during training.',
           '2. **Processor config.** Official uses '
           '`AutoProcessor.from_pretrained(model_path, '
           'resized_width=14*17*2, resized_height=14*17*4)`. Ours uses '
           '`AutoProcessor.from_pretrained(base_model, '
           'min_pixels=200704, max_pixels=1003520*4)` — a different target '
           'resolution that drives the vision encoder into a different '
           'token-grid shape than training time.',
           '3. **`max_new_tokens=768`** vs official `4000`. Long programs '
           '(complex CAD) are silently truncated under our setting.',
           '',
           'Fix: align all three to official, then re-run cad_bench_722 + '
           'DeepCAD-300 + Fusion360-300 in a sandbox venv pinned to '
           'transformers==4.50.3.',
           '',
           '## Selected cases',
           '']
    md.append('| # | category | family | diff | stem | '
              + ' | '.join(label.split()[0] for _, label in MODELS) + ' |')
    md.append('|---|----------|--------|------|------|'
              + '|'.join(['---'] * len(MODELS)) + '|')
    for r in rows_md:
        cells = []
        for slug, _ in MODELS:
            x = r['records'][slug]
            iou_s = f'{x["iou"]:.2f}' if x['iou'] is not None else '—'
            if x['ep'] is True: ep_s = '✓'
            elif x['ep'] is False: ep_s = '✗'
            else: ep_s = '—'
            f1_s = f'{x["f1"]:.2f}' if x['f1'] is not None else '—'
            cells.append(f'IoU{iou_s}/ESS{ep_s}/F1{f1_s}')
        md.append(f'| {r["idx"]} | {_category_label(r["cat"])} | '
                  f'{r["family"][:14]} | {r["diff"]} | `{r["stem"][:30]}` | '
                  + ' | '.join(cells) + ' |')

    md.append('')
    md.append('### Per-case op breakdown (top-3 per category)')
    for cat in ('A_ours_wins', 'B_ess_pass_low_iou',
                'C_high_iou_ess_fail', 'D_qwen_zs_pass'):
        md.append('')
        md.append(f'#### {_category_label(cat)}')
        md.append('')
        for r in rows_md:
            if r['cat'] != cat:
                continue
            md.append(f'**#{r["idx"]} — {r["stem"]}** (family={r["family"]}, '
                      f'diff={r["diff"]})')
            for slug, label in MODELS:
                x = r['records'][slug]
                gen = x.get('gen_ops') or []
                gt  = x.get('gt_ops')  or []
                missing = sorted(set(gt) - set(gen))[:6]
                spurious = sorted(set(gen) - set(gt))[:6]
                line = (f'  - {label}: IoU={x["iou"]} ESS={x["ep"]} '
                        f'F1={x["f1"]}; '
                        f'missing={missing} spurious={spurious}')
                md.append(line)
            md.append('')

    md_path = out_dir / 'iou_vs_essential_analysis.md'
    md_path.write_text('\n'.join(md))
    print(f'  → {md_path}', flush=True)

    # Discord
    if args.discord:
        print('\nPosting to Discord …', flush=True)
        msg = ('📊 **cad_bench_722 — IoU vs essential-ops, 12 representative cases**\n'
               'Layout: # | GT | Cadrille-rl(4.50.3 repro) | CADEvolve-rl1 | '
               'Cadrille-Q3VL-v3 (ours) | Qwen2.5-VL-3B (zs)\n'
               'Each pred cell shows IoU + ESS verdict (✓/✗/—) + feature F1.\n'
               'Categories: **A** ours wins, **B** ESS✓ but IoU low, '
               '**C** IoU high but ESS✗, **D** Qwen-zs ESS✓ '
               '(small-denominator bias).\n'
               'Full analysis + per-case op diffs in the attached `.md`.\n')
        ok = post_to_discord(msg, [grid_path, md_path])
        print(f'  {"sent" if ok else "FAILED"}')

    print('\nDone.')


if __name__ == '__main__':
    main()
