"""CADEvolve weak-family deep-dive: families where CADEvolve underperforms,
with GT image + CADEvolve render + Q3VL render side-by-side, full op
annotations + per-family stats.

Selection: families where mean CADEvolve IoU < 0.50 AND ≥ 4 cases. Within
each family, pick the case where Q3VL most clearly beats CADEvolve (largest
positive Q3VL−CADEvolve IoU delta among both-exec_ok cases).

Posts a single grid PNG + per-family stats markdown to Discord.

Usage:
    set -a; source .env; set +a
    uv run python research/repro_official/build_cadevolve_weak_families.py --discord
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'research/essential_ops'))

EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'
RENDER_CACHE = Path('/tmp/cad_bench_722_renders')
OUT_DIR = EVAL_ROOT / 'cadevolve_weak'

MODELS = [
    ('cadevolve_rl1',        'CADEvolve v3'),
    ('cadrille_qwen3vl_v3',  'Q3VL (ours)'),
]
PRED_DIR = {s: EVAL_ROOT / s for s, _ in MODELS}
META_PATH = {s: EVAL_ROOT / s / 'metadata.jsonl' for s, _ in MODELS}

SIDE     = 256
LABEL_H  = 22
ANNOT_H  = 90
NUM_W    = 110
PAD      = 4
HEADER_H = 110

WEAK_THR = 0.50
MIN_N    = 4
N_FAMS   = 10


# ── render worker (reuses logic from build_full_grid_v3_ops.py) ────────────

class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout('budget')

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

# Out-of-process render_img — protects us from open3d-cpu's segfaults on
# certain meshes. PyPI open3d-cpu (the only easy install) crashes on a
# small fraction of preds; the source-built open3d from scripts/setup.sh
# does not. Until source build is done, run in subprocess + fall back
# to pyvista on crash.
_RENDER_TMPL = textwrap.dedent('''\
    import sys
    sys.path.insert(0, {repo!r})
    from common.meshio import render_img
    img = render_img(sys.argv[1])['video'][0]
    img.save(sys.argv[2])
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
    cache_path = Path(cache_dir) / f'{slug}__{stem}.png'
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
            # Try canonical render_img in subprocess (segfault-safe).
            tmp_png = str(cache_path) + '.tmp'
            render_script = _RENDER_TMPL.format(repo=str(REPO))
            r2 = subprocess.run([sys.executable, '-c', render_script, stl, tmp_png],
                                capture_output=True, timeout=20)
            if r2.returncode == 0 and Path(tmp_png).exists() and Path(tmp_png).stat().st_size > 0:
                Path(tmp_png).rename(cache_path)
                return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path), 'error': None}
            # Fallback: pyvista in-process (no segfault).
            try: Path(tmp_png).unlink()
            except Exception: pass
            img = _render_4view_pyvista(stl, side=268)
            img.save(cache_path, format='PNG')
            return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path),
                    'error': 'render_img_segfault_pyvista_fallback'}
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


def _annot_cell(img, label, iou, ep, f1, gen_ops, essential_set):
    from PIL import Image, ImageDraw
    side = img.width
    canvas = Image.new('RGB', (side, side + LABEL_H + ANNOT_H), color=(18, 18, 18))
    canvas.paste(img, (0, LABEL_H))
    d = ImageDraw.Draw(canvas)
    f_lab = _font(11, bold=True)
    f_val = _font(10)
    d.rectangle([0, 0, side, LABEL_H], fill=(35, 35, 50))
    d.text((4, 3), label, fill=(225, 225, 225), font=f_lab)
    y0 = side + LABEL_H

    if ep is True:
        ep_str, ep_color = 'ESS✓', (90, 220, 100)
    elif ep is False:
        ep_str, ep_color = 'ESS✗', (240, 90, 90)
    else:
        ep_str, ep_color = 'ESS—', (160, 160, 160)
    iou_str = f'IoU={iou:.2f}' if iou is not None else 'IoU=—'
    f1_str  = f'F1={f1:.2f}' if f1 is not None else ''
    d.text((4, y0 + 2), iou_str, fill=(225, 225, 225), font=f_lab)
    d.text((side - 80, y0 + 2), ep_str, fill=ep_color, font=f_lab)
    d.text((side - 36, y0 + 2), f1_str, fill=(220, 220, 220), font=f_val)

    if gen_ops is None:
        d.text((4, y0 + 18), '(no pred)', fill=(160, 160, 160), font=f_val)
        return canvas

    color_used = (90, 220, 100); color_extra = (170, 170, 170); color_missing = (240, 90, 90)
    pairs = []
    used = set(gen_ops)
    for op in sorted(used):
        if op in essential_set: pairs.append((op, color_used))
        else: pairs.append((op, color_extra))
    for op in sorted(essential_set - used):
        pairs.append(('!' + op, color_missing))
    line_x, line_y, line_h = 4, y0 + 18, 13
    for op, color in pairs:
        text = op + ' '
        bbox = d.textbbox((0, 0), text, font=f_val)
        w = bbox[2] - bbox[0]
        if line_x + w > side - 4:
            line_y += line_h; line_x = 4
        d.text((line_x, line_y), text, fill=color, font=f_val)
        line_x += w
    return canvas


def _number_cell(idx, family, fam_stat, width, height):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(20, 20, 20))
    d = ImageDraw.Draw(img)
    fbig = _font(20, bold=True)
    fmid = _font(11, bold=True)
    fsm  = _font(9)
    s = f'#{idx}'
    bbox = d.textbbox((0, 0), s, font=fbig); w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 4), s, fill=(240, 240, 240), font=fbig)
    fam = (family or '?')[:18]
    bbox = d.textbbox((0, 0), fam, font=fmid); w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 32), fam, fill=(255, 200, 100), font=fmid)
    if fam_stat:
        l1 = f'CE μ={fam_stat["ce_iou"]:.2f}'
        l2 = f'Q3VL μ={fam_stat["q_iou"]:.2f}'
        bbox = d.textbbox((0, 0), l1, font=fsm); w = bbox[2]-bbox[0]
        d.text(((width-w)/2, 56), l1, fill=(220,140,140), font=fsm)
        bbox = d.textbbox((0, 0), l2, font=fsm); w = bbox[2]-bbox[0]
        d.text(((width-w)/2, 72), l2, fill=(140,220,140), font=fsm)
        l3 = f'n={fam_stat["n"]}'
        bbox = d.textbbox((0, 0), l3, font=fsm); w = bbox[2]-bbox[0]
        d.text(((width-w)/2, 90), l3, fill=(160,160,160), font=fsm)
    return img


def _page_header(width, height):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(35, 35, 50))
    d = ImageDraw.Draw(img)
    f1 = _font(15, bold=True)
    f2 = _font(11)
    d.text((10, 8), 'cad_bench_722 — CADEvolve\'s weakest families',
           fill=(230, 230, 230), font=f1)
    d.text((10, 32), f'Families with mean CADEvolve IoU < {WEAK_THR:.2f} '
           f'(n ≥ {MIN_N}). Showing one representative case per family — '
           f'the one where Q3VL−CADEvolve IoU delta is largest.',
           fill=(180, 200, 230), font=f2)
    d.text((10, 52), 'cols: # | family stats | GT | CADEvolve v3 | Q3VL (ours).  '
           'ops: green=essential matched, !red=missing, gray=extra non-canonical',
           fill=(180, 180, 200), font=f2)
    d.text((10, 75), 'CE μ = mean CADEvolve IoU on family   ·   '
           'Q3VL μ = mean Q3VL IoU on family   ·   '
           'each cell IoU/ESS/F1 + ops list',
           fill=(150, 150, 180), font=f2)
    return img


def _post(content, attachment):
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url: return False
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def w(s): body.write(s.encode())
    w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n')
    w('Content-Type: application/json\r\n\r\n')
    w(json.dumps({'content': content}) + '\r\n')
    w(f'--{boundary}\r\nContent-Disposition: form-data; '
      f'name="file"; filename="{attachment.name}"\r\n')
    ct = 'image/png' if attachment.suffix == '.png' else 'text/markdown'
    w(f'Content-Type: {ct}\r\n\r\n')
    body.write(attachment.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    req = urllib.request.Request(url, data=body.getvalue(), headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cadevolve-weak/1.0',
    })
    try:
        urllib.request.urlopen(req, timeout=30).read(); return True
    except Exception as e:
        print(f'Discord failed: {e}'); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-fams',  type=int, default=N_FAMS)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--task-timeout', type=int, default=30)
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)

    print('Loading metadata + essential_ops …', flush=True)
    metas = {s: {} for s, _ in MODELS}
    for s, _ in MODELS:
        for line in open(META_PATH[s]):
            try: r = json.loads(line); metas[s][r['stem']] = r
            except Exception: pass

    ess_raw = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    ess_per_case = {slug: {c['stem']: c for c in m['per_case']}
                    for slug, m in ess_raw['models'].items()}

    from canonical_ops import ESSENTIAL_BY_FAMILY
    def family_essential_set(fam):
        spec = ESSENTIAL_BY_FAMILY.get(fam) or []
        out = set()
        for elem in spec:
            if isinstance(elem, str): out.add(elem)
            else: out.update(elem)
        return out

    # Per-family stats for both models
    fam_stats = defaultdict(lambda: {'ce_iou':[], 'q_iou':[], 'n':0,
                                     'ce_ess':[], 'q_ess':[], 'ce_f1':[], 'q_f1':[]})
    ce_per_case = ess_per_case.get('cadevolve_rl1', {})
    q_per_case  = ess_per_case.get('cadrille_qwen3vl_v3', {})
    for stem, c in ce_per_case.items():
        fam = c['family']
        ci = metas['cadevolve_rl1'].get(stem,{}).get('iou')
        qi = metas['cadrille_qwen3vl_v3'].get(stem,{}).get('iou')
        if ci is not None: fam_stats[fam]['ce_iou'].append(ci)
        if qi is not None: fam_stats[fam]['q_iou'].append(qi)
        if c.get('essential_pass') is not None:
            fam_stats[fam]['ce_ess'].append(c['essential_pass'])
        fam_stats[fam]['ce_f1'].append(c.get('feature_f1', 0))
        qrec = q_per_case.get(stem, {})
        if qrec.get('essential_pass') is not None:
            fam_stats[fam]['q_ess'].append(qrec['essential_pass'])
        fam_stats[fam]['q_f1'].append(qrec.get('feature_f1', 0))

    def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
    weak_fams = []
    for fam, s in fam_stats.items():
        if len(s['ce_iou']) < MIN_N: continue
        ci_mean = mean(s['ce_iou'])
        if ci_mean >= WEAK_THR: continue
        weak_fams.append({
            'family': fam, 'n': len(s['ce_iou']),
            'ce_iou': ci_mean, 'q_iou': mean(s['q_iou']),
            'ce_ess': mean(s['ce_ess']) if s['ce_ess'] else float('nan'),
            'q_ess':  mean(s['q_ess'])  if s['q_ess']  else float('nan'),
            'ce_f1':  mean(s['ce_f1']), 'q_f1': mean(s['q_f1']),
        })
    weak_fams.sort(key=lambda r: r['ce_iou'])
    weak_fams = weak_fams[:args.n_fams]
    print(f'  {len(weak_fams)} weak families:', flush=True)
    for f in weak_fams:
        print(f'    {f["family"]:<26} n={f["n"]} ce={f["ce_iou"]:.3f} q={f["q_iou"]:.3f}',
              flush=True)

    # Pick representative case per family: largest q − ce IoU delta
    chosen = []
    for fam_info in weak_fams:
        fam = fam_info['family']
        cands = []
        for stem, c in ce_per_case.items():
            if c['family'] != fam: continue
            ce_rec = metas['cadevolve_rl1'].get(stem) or {}
            q_rec  = metas['cadrille_qwen3vl_v3'].get(stem) or {}
            if (ce_rec.get('error_type') == 'success'
                    and q_rec.get('error_type') == 'success'
                    and ce_rec.get('iou') is not None and q_rec.get('iou') is not None):
                cands.append((stem, q_rec['iou'] - ce_rec['iou']))
        if not cands:
            for stem, c in ce_per_case.items():
                if c['family'] != fam: continue
                cands.append((stem, 0))
        if not cands: continue
        cands.sort(key=lambda t: -t[1])
        chosen.append({'stem': cands[0][0], **fam_info})

    print(f'  selected {len(chosen)} cases', flush=True)

    # GT
    print('Fetching GT composite_png …', flush=True)
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    stems = {c['stem'] for c in chosen}
    gt_by_stem = {row['stem']: row['composite_png'] for row in ds if row['stem'] in stems}

    # Render
    tasks = []
    for c in chosen:
        for slug, _ in MODELS:
            rec = metas[slug].get(c['stem']) or {}
            if rec.get('error_type') != 'success': continue
            py = PRED_DIR[slug] / f'{c["stem"]}.py'
            if not py.exists(): continue
            tasks.append((slug, c['stem'], str(py), args.task_timeout, str(RENDER_CACHE)))
    pending = [t for t in tasks if not (RENDER_CACHE / f'{t[0]}__{t[1]}.png').exists()]
    print(f'  {len(tasks)} render tasks, {len(pending)} pending', flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=20) as pool:
            futs = {pool.submit(_render_one, t): t for t in pending}
            for fut in as_completed(futs):
                try: fut.result(timeout=args.task_timeout * 2)
                except Exception as e: print(f'  ! {e}', flush=True)

    # Build figure
    print('Building figure …', flush=True)
    from PIL import Image
    cell_h = LABEL_H + SIDE + ANNOT_H
    page_w = NUM_W + (1 + len(MODELS)) * (SIDE + PAD)
    page_h = HEADER_H + len(chosen) * (cell_h + PAD)
    page = Image.new('RGB', (page_w, page_h), color=(10, 10, 10))
    page.paste(_page_header(page_w, HEADER_H), (0, 0))

    for ri, c in enumerate(chosen):
        y = HEADER_H + ri * (cell_h + PAD)
        page.paste(_number_cell(ri + 1, c['family'], c, NUM_W, cell_h), (0, y))
        x = NUM_W + PAD

        ess_set = family_essential_set(c['family'])

        # GT
        gt = gt_by_stem.get(c['stem'])
        if gt is not None:
            gt_img = gt.convert('RGB').resize((SIDE, SIDE), Image.LANCZOS)
            page.paste(_annot_cell(gt_img, 'GT', None, None, None,
                                   sorted(ess_set), ess_set), (x, y))
        else:
            page.paste(_annot_cell(_fail_tile(SIDE, 'NO GT'), 'GT',
                                   None, None, None, [], set()), (x, y))
        x += SIDE + PAD

        # 2 model cols
        for slug, label in MODELS:
            rec = metas[slug].get(c['stem']) or {}
            ec  = ess_per_case.get(slug, {}).get(c['stem'], {})
            iou = rec.get('iou'); ep = ec.get('essential_pass'); f1 = ec.get('feature_f1')
            gen_ops = ec.get('gen_ops')
            cache_path = RENDER_CACHE / f'{slug}__{c["stem"]}.png'
            if cache_path.exists() and cache_path.stat().st_size > 0:
                img = Image.open(cache_path).convert('RGB').resize(
                    (SIDE, SIDE), Image.LANCZOS)
            else:
                et = rec.get('error_type', 'no pred')
                img = _fail_tile(SIDE, et.upper())
            page.paste(_annot_cell(img, label, iou, ep, f1, gen_ops, ess_set), (x, y))
            x += SIDE + PAD

    out_grid = out_dir / 'cadevolve_weak_families.png'
    page.save(out_grid, optimize=True)
    sz = out_grid.stat().st_size / 1024 / 1024
    print(f'  → {out_grid.name}  {page.size[0]}×{page.size[1]}  {sz:.2f}MB',
          flush=True)

    # Stats markdown
    L = ['# CADEvolve\'s weak families on cad_bench_722',
         '',
         f'Top {len(weak_fams)} families with mean CADEvolve IoU < {WEAK_THR} '
         f'(n ≥ {MIN_N} cases per family). For each family, picked the '
         f'representative case where Q3VL most clearly outperforms.',
         '',
         '| family | n | CE IoU | Q3VL IoU | Δ | CE ESS | Q3VL ESS | CE F1 | Q3VL F1 |',
         '|--------|----|--------|----------|---|--------|----------|-------|---------|']
    for f in weak_fams:
        delta = f['q_iou'] - f['ce_iou']
        L.append(f'| {f["family"]} | {f["n"]} | {f["ce_iou"]:.3f} | '
                 f'{f["q_iou"]:.3f} | **{delta:+.3f}** | '
                 f'{f["ce_ess"]:.2f} | {f["q_ess"]:.2f} | '
                 f'{f["ce_f1"]:.2f} | {f["q_f1"]:.2f} |')
    L += ['',
          '## Pattern',
          '',
          'CADEvolve\'s weakest families share a structural theme: **rotational '
          'symmetry, swept profiles, or smooth curved bodies**. Specifically:',
          '',
          '- `torsion_spring`, `bellows`, `twisted_drill`, `propeller`: helical / '
          'twisted geometry (need `sweep+helix` or twistExtrude)',
          '- `ball_knob`, `capsule`, `knob`, `wing_nut`: spherical / curved '
          '(need `sphere` or `revolve`)',
          '- `venturi_tube`, `duct_elbow`, `pipe_elbow`, `t_pipe_fitting`, '
          '`j_hook`: swept tube cross-sections (need `sweep` or `loft`)',
          '- `impeller`, `double_simplex_sprocket`: radial periodic patterns '
          '(need `polarArray`)',
          '',
          'CADEvolve\'s vocabulary is **prismatic-heavy**: cylinder + box + cut + '
          'union dominates. It almost never uses `revolve / sweep / loft / sphere '
          '/ polarArray`. Where the part *requires* one of those primitives, '
          'CADEvolve falls back to discrete approximation (cylinder stack ≈ '
          'sphere; many small boxes ≈ polarArray) which gives low IoU.',
          '',
          'Q3VL (ours) wins on these because our SFT mix taught the canonical '
          'BenchCAD vocabulary including `.sphere()`, `.revolve()`, '
          '`.sweep()`, `.polarArray()`.']
    out_md = out_dir / 'cadevolve_weak_families.md'
    out_md.write_text('\n'.join(L))
    print(f'  → {out_md.name}', flush=True)

    if args.discord:
        msg = ('🔍 **CADEvolve\'s weak families on cad_bench_722**\n'
               '\n'
               f'Top {len(weak_fams)} families with mean CADEvolve IoU '
               f'< {WEAK_THR}. For each family one representative case is '
               'shown with GT + CADEvolve render + Q3VL render, full ops list, '
               'IoU/ESS/F1 per cell.\n'
               '\n'
               '**Pattern**: CADEvolve\'s weakness is concentrated in '
               '*rotational/curved/swept* shapes — `torsion_spring`, '
               '`bellows`, `ball_knob`, `capsule`, `venturi_tube`, '
               '`pipe_elbow`, `propeller`. Its prismatic-heavy vocabulary '
               '(cylinder/box/cut/union) doesn\'t cover `sphere` / `revolve` '
               '/ `sweep` / `polarArray`, so it approximates them with '
               'discrete stacks → low IoU.\n'
               '\n'
               'Per-family stats table in the attached `.md`.')
        ok = _post(msg, out_grid)
        if ok:
            time.sleep(2)
            _post('📊 Per-family stats table.', out_md)


if __name__ == '__main__':
    main()
