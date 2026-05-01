"""All families where CADEvolve fails the strict ESS metric most often.

Criteria: family has ≥ 4 cases with essential spec, and CADEvolve's mean
essential_pass rate across that family is < 0.5. Picks the canonical
representative case per family (`_per_family_canonical.pick_canonical_case`)
and renders GT + CADEvolve + Q3VL side-by-side with full op annotations.

Posts as 2 pages to Discord (16 + 15 families typically).

Usage:
    set -a; source .env; set +a
    uv run python research/repro_official/build_low_ess_families.py --discord
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
sys.path.insert(0, str(REPO / 'research/repro_official'))

EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'
RENDER_CACHE = Path('/tmp/cad_bench_722_renders')
OUT_DIR = EVAL_ROOT / 'low_ess_families'

CE_ESS_THR = 0.5
MIN_N = 4

PRED_DIR = {
    'CADEvolve v3': EVAL_ROOT / 'cadevolve_rl1',
    'Q3VL (ours)':  EVAL_ROOT / 'cadrille_qwen3vl_v3',
}
META_PATH = {
    'CADEvolve v3': EVAL_ROOT / 'cadevolve_rl1' / 'metadata.jsonl',
    'Q3VL (ours)':  EVAL_ROOT / 'cadrille_qwen3vl_v3' / 'metadata.jsonl',
}
SLUG = {'CADEvolve v3': 'cadevolve_rl1', 'Q3VL (ours)': 'cadrille_qwen3vl_v3'}

SIDE = 240; LABEL_H = 22; ANNOT_H = 86; NUM_W = 130; PAD = 4; HEADER_H = 70


# ── render worker ──────────────────────────────────────────────────────────

class _Timeout(Exception): pass
def _alarm(s, f): raise _Timeout('budget')

_EXEC_TMPL = textwrap.dedent('''\
    import sys, cadquery as cq, trimesh
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
_RENDER_TMPL = textwrap.dedent('''\
    import sys
    sys.path.insert(0, {repo!r})
    from common.meshio import render_img
    img = render_img(sys.argv[1])['video'][0]
    img.save(sys.argv[2])
''')


def _render_4view_pyvista(stl_path, side=268):
    import numpy as np, pyvista as pv
    from PIL import Image
    mesh = pv.read(stl_path)
    b = mesh.bounds
    cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
    ext = max(b[1]-b[0], b[3]-b[2], b[5]-b[4]) or 1.0
    mesh = mesh.translate([-cx,-cy,-cz]).scale(1.0/ext)
    tile = side // 2; color = (1.0, 1.0, 0.534)
    fronts = [(1,1,1),(-1,-1,-1),(-1,1,-1),(1,-1,1)]
    tiles = []
    for fx,fy,fz in fronts:
        pl = pv.Plotter(off_screen=True, window_size=(tile, tile))
        pl.background_color = (0.07,0.07,0.07)
        pl.add_mesh(mesh, color=color, lighting=True, smooth_shading=True)
        pl.camera_position = [(fx*1.6,fy*1.6,fz*1.6),(0,0,0),(0,0,1)]
        pl.enable_parallel_projection(); pl.camera.zoom(1.4)
        arr = pl.screenshot(None, return_img=True); pl.close()
        tiles.append(arr)
    canvas = np.vstack([np.hstack([tiles[0],tiles[1]]), np.hstack([tiles[2],tiles[3]])])
    return Image.fromarray(canvas).convert('RGB').resize((side,side), Image.LANCZOS)


def _render_one(args):
    slug, stem, py_path, timeout_sec, cache_dir = args
    cp = Path(cache_dir) / f'{slug}__{stem}.png'
    if cp.exists() and cp.stat().st_size > 0:
        return {'slug': slug, 'stem': stem, 'cache_path': str(cp), 'error': None}
    if timeout_sec > 0:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(timeout_sec))
    try:
        code = Path(py_path).read_text()
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            stl = f.name
        try:
            r = subprocess.run([sys.executable, '-c', _EXEC_TMPL.format(code=code), stl],
                               capture_output=True, timeout=min(timeout_sec, 30))
            if r.returncode != 0 or Path(stl).stat().st_size < 100:
                return {'slug': slug, 'stem': stem, 'cache_path': None, 'error': 'exec_fail'}
            tmp_png = str(cp) + '.tmp'
            r2 = subprocess.run([sys.executable, '-c', _RENDER_TMPL.format(repo=str(REPO)), stl, tmp_png],
                                capture_output=True, timeout=20)
            if r2.returncode == 0 and Path(tmp_png).exists() and Path(tmp_png).stat().st_size > 0:
                Path(tmp_png).rename(cp)
                return {'slug': slug, 'stem': stem, 'cache_path': str(cp), 'error': None}
            try: Path(tmp_png).unlink(missing_ok=True)
            except Exception: pass
            img = _render_4view_pyvista(stl, side=268); img.save(cp, format='PNG')
            return {'slug': slug, 'stem': stem, 'cache_path': str(cp), 'error': 'pyvista_fallback'}
        finally:
            try: Path(stl).unlink()
            except Exception: pass
    except Exception as e:
        return {'slug': slug, 'stem': stem, 'cache_path': None,
                'error': f'{type(e).__name__}: {str(e)[:60]}'}
    finally:
        if timeout_sec > 0: signal.alarm(0)


# ── drawing ────────────────────────────────────────────────────────────────

def _font(size, bold=False):
    from PIL import ImageFont
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try: return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{name}', size)
    except Exception: return ImageFont.load_default()

def _fail_tile(side, msg):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (side, side), color=(28,28,28))
    d = ImageDraw.Draw(img); f = _font(16, bold=True)
    bbox = d.textbbox((0,0), msg, font=f); w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
    d.text(((side-w)/2,(side-h)/2), msg, fill=(220,90,90), font=f)
    return img

def _annot_cell(img, label, iou, ep, f1, gen_ops, ess_set):
    from PIL import Image, ImageDraw
    side = img.width
    canvas = Image.new('RGB', (side, side+LABEL_H+ANNOT_H), color=(18,18,18))
    canvas.paste(img, (0, LABEL_H))
    d = ImageDraw.Draw(canvas); f_lab = _font(11, bold=True); f_val = _font(10)
    d.rectangle([0,0,side,LABEL_H], fill=(35,35,50))
    d.text((4,3), label, fill=(225,225,225), font=f_lab)
    y0 = side + LABEL_H
    if ep is True: ep_str, ep_color = 'ESS✓', (90,220,100)
    elif ep is False: ep_str, ep_color = 'ESS✗', (240,90,90)
    else: ep_str, ep_color = 'ESS—', (160,160,160)
    iou_str = f'IoU={iou:.2f}' if iou is not None else 'IoU=—'
    f1_str = f'F1={f1:.2f}' if f1 is not None else ''
    d.text((4, y0+2), iou_str, fill=(225,225,225), font=f_lab)
    d.text((side-78, y0+2), ep_str, fill=ep_color, font=f_lab)
    d.text((side-36, y0+2), f1_str, fill=(220,220,220), font=f_val)
    if gen_ops is None:
        d.text((4, y0+18), '(no pred)', fill=(160,160,160), font=f_val)
        return canvas
    used = set(gen_ops); pairs = []
    for op in sorted(used):
        pairs.append((op, (90,220,100) if op in ess_set else (170,170,170)))
    for op in sorted(ess_set - used):
        pairs.append(('!' + op, (240,90,90)))
    line_x, line_y, line_h = 4, y0+18, 13
    for op, col in pairs:
        text = op + ' '
        bbox = d.textbbox((0,0), text, font=f_val); w = bbox[2]-bbox[0]
        if line_x + w > side - 4: line_y += line_h; line_x = 4
        d.text((line_x, line_y), text, fill=col, font=f_val)
        line_x += w
    return canvas

def _info_cell(idx, family, n, ce_ess, q_ess, ce_iou, q_iou, width, height):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(20,20,20))
    d = ImageDraw.Draw(img)
    fbig = _font(18, bold=True); fmid = _font(10, bold=True); fsm = _font(9)
    s = f'#{idx}'
    bbox = d.textbbox((0,0), s, font=fbig); w = bbox[2]-bbox[0]
    d.text(((width-w)/2, 4), s, fill=(240,240,240), font=fbig)
    fam = (family or '?')[:18]
    bbox = d.textbbox((0,0), fam, font=fmid); w = bbox[2]-bbox[0]
    d.text(((width-w)/2, 26), fam, fill=(255,200,100), font=fmid)
    n_str = f'n={n}'
    d.text((6, 44), n_str, fill=(160,160,160), font=fsm)
    d.text((6, 58), f'CE_ess={ce_ess*100:.0f}%', fill=(220,140,140), font=fsm)
    d.text((6, 72), f'Q_ess ={q_ess*100:.0f}%', fill=(140,220,140), font=fsm)
    d.text((6, 90), f'CE_IoU={ce_iou:.2f}', fill=(220,140,140), font=fsm)
    d.text((6, 104), f'Q_IoU ={q_iou:.2f}', fill=(140,220,140), font=fsm)
    return img

def _page_header(width, height, page_idx, n_pages, lo, hi, n_total):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(35,35,50))
    d = ImageDraw.Draw(img); f1 = _font(15, bold=True); f2 = _font(11)
    d.text((10, 8), f'CADEvolve low-ESS families (mean CE_ess < 50%)  '
           f'— page {page_idx+1}/{n_pages} (families {lo}..{hi} of {n_total})',
           fill=(230,230,230), font=f1)
    d.text((10, 32), 'cols: # + family stats | GT | CADEvolve v3 | Q3VL (ours)   '
                    '· canonical case per family',
           fill=(180,200,230), font=f2)
    d.text((10, 50), 'green=essential matched · !red=essential missing · gray=extra non-canonical',
           fill=(180,180,200), font=f2)
    return img


def _post(content, attachment):
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        try: url = open(REPO/'.env').read().split('DISCORD_WEBHOOK_URL=')[1].split()[0]
        except Exception: return False
    if not url: return False
    boundary = uuid.uuid4().hex; body = io.BytesIO()
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
        'User-Agent': 'cad-low-ess/1.0',
    })
    try: urllib.request.urlopen(req, timeout=30).read(); return True
    except Exception as e: print(f'Discord failed: {e}'); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ce-ess-thr', type=float, default=CE_ESS_THR)
    ap.add_argument('--min-n', type=int, default=MIN_N)
    ap.add_argument('--n-chunks', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--task-timeout', type=int, default=30)
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)

    # Load metadata + ess
    metas = {}
    for label, p in META_PATH.items():
        metas[label] = {}
        for line in open(p):
            try: r = json.loads(line); metas[label][r['stem']] = r
            except Exception: pass

    ess = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    ess_pc = {slug: {c['stem']: c for c in m['per_case']}
              for slug, m in ess['models'].items()}

    from canonical_ops import ESSENTIAL_BY_FAMILY
    from _per_family_canonical import pick_canonical_case

    def ess_set(fam):
        spec = ESSENTIAL_BY_FAMILY.get(fam) or []
        out = set()
        for el in spec:
            if isinstance(el, str): out.add(el)
            else: out.update(el)
        return out

    # Aggregate per-family stats
    fam = defaultdict(lambda: {'ce_ess':[], 'q_ess':[], 'ce_iou':[], 'q_iou':[]})
    for c in ess['models']['cadevolve_rl1']['per_case']:
        f = c['family']
        if f not in ESSENTIAL_BY_FAMILY: continue
        if c['essential_pass'] is not None:
            fam[f]['ce_ess'].append(int(c['essential_pass']))
        qc = ess_pc['cadrille_qwen3vl_v3'].get(c['stem'], {})
        if qc.get('essential_pass') is not None:
            fam[f]['q_ess'].append(int(qc['essential_pass']))
        ci = (metas['CADEvolve v3'].get(c['stem']) or {}).get('iou')
        qi = (metas['Q3VL (ours)'].get(c['stem']) or {}).get('iou')
        if ci is not None: fam[f]['ce_iou'].append(ci)
        if qi is not None: fam[f]['q_iou'].append(qi)
    def m(xs): return sum(xs)/len(xs) if xs else 0.0

    low = []
    for f, s in fam.items():
        if len(s['ce_ess']) < args.min_n: continue
        ce_ess_pct = m(s['ce_ess'])
        if ce_ess_pct >= args.ce_ess_thr: continue
        low.append({'family': f, 'n': len(s['ce_ess']),
                    'ce_ess': ce_ess_pct, 'q_ess': m(s['q_ess']),
                    'ce_iou': m(s['ce_iou']), 'q_iou': m(s['q_iou'])})
    low.sort(key=lambda r: (r['ce_ess'], -r['q_ess']))
    print(f'Found {len(low)} families with CE_ess < {args.ce_ess_thr*100:.0f}%', flush=True)

    # Pick canonical case per family
    chosen = []
    for f in low:
        stem = pick_canonical_case(f['family'])
        if stem is None: continue
        chosen.append({**f, 'stem': stem})
    print(f'  selected {len(chosen)} canonical cases', flush=True)

    # GT
    print('Fetching GT …', flush=True)
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    stems = {c['stem'] for c in chosen}
    gt_by_stem = {row['stem']: row['composite_png']
                  for row in ds if row['stem'] in stems}

    # Render
    tasks = []
    for c in chosen:
        for label in ('CADEvolve v3', 'Q3VL (ours)'):
            slug = SLUG[label]
            rec = metas[label].get(c['stem']) or {}
            if rec.get('error_type') != 'success': continue
            py = PRED_DIR[label] / f'{c["stem"]}.py'
            if not py.exists(): continue
            tasks.append((slug, c['stem'], str(py), args.task_timeout, str(RENDER_CACHE)))
    pending = [t for t in tasks if not (RENDER_CACHE / f'{t[0]}__{t[1]}.png').exists()]
    print(f'Render: {len(tasks)} total, {len(pending)} pending', flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=10) as pool:
            for fut in as_completed({pool.submit(_render_one, t): t for t in pending}):
                try: fut.result(timeout=args.task_timeout * 2)
                except Exception as e: print(f'  ! {e}', flush=True)

    # Build pages
    from PIL import Image
    cell_h = LABEL_H + SIDE + ANNOT_H
    page_w = NUM_W + 3 * (SIDE + PAD)  # GT + CE + Q3VL
    chunk = (len(chosen) + args.n_chunks - 1) // args.n_chunks
    out_paths = []
    for ci in range(args.n_chunks):
        lo = ci * chunk
        hi = min(lo + chunk, len(chosen))
        if lo >= hi: break
        sub = chosen[lo:hi]
        page_h = HEADER_H + len(sub) * (cell_h + PAD)
        page = Image.new('RGB', (page_w, page_h), color=(10,10,10))
        page.paste(_page_header(page_w, HEADER_H, ci, args.n_chunks,
                                lo+1, hi, len(chosen)), (0, 0))
        for ri, c in enumerate(sub):
            y = HEADER_H + ri * (cell_h + PAD)
            page.paste(_info_cell(lo+ri+1, c['family'], c['n'],
                                  c['ce_ess'], c['q_ess'],
                                  c['ce_iou'], c['q_iou'],
                                  NUM_W, cell_h), (0, y))
            x = NUM_W + PAD
            es = ess_set(c['family'])
            gt = gt_by_stem.get(c['stem'])
            if gt:
                gt_img = gt.convert('RGB').resize((SIDE, SIDE), Image.LANCZOS)
                page.paste(_annot_cell(gt_img, 'GT', None, None, None, sorted(es), es), (x, y))
            else:
                page.paste(_annot_cell(_fail_tile(SIDE, 'NO GT'), 'GT', None, None, None, [], set()), (x, y))
            x += SIDE + PAD
            for label in ('CADEvolve v3', 'Q3VL (ours)'):
                slug = SLUG[label]
                rec = metas[label].get(c['stem']) or {}
                ec = ess_pc.get(slug, {}).get(c['stem'], {})
                iou = rec.get('iou'); ep = ec.get('essential_pass'); f1 = ec.get('feature_f1')
                gen_ops = ec.get('gen_ops')
                cp = RENDER_CACHE / f'{slug}__{c["stem"]}.png'
                if cp.exists() and cp.stat().st_size > 0:
                    img = Image.open(cp).convert('RGB').resize((SIDE, SIDE), Image.LANCZOS)
                else:
                    et = rec.get('error_type', 'no pred')
                    img = _fail_tile(SIDE, et.upper())
                page.paste(_annot_cell(img, label, iou, ep, f1, gen_ops, es), (x, y))
                x += SIDE + PAD
        out_p = out_dir / f'low_ess_families_p{ci+1}.png'
        page.save(out_p, optimize=True)
        sz = out_p.stat().st_size / 1024 / 1024
        print(f'  page {ci+1}/{args.n_chunks}: {hi-lo} fams  {page.size[0]}×{page.size[1]}  {sz:.1f}MB → {out_p.name}',
              flush=True)
        out_paths.append((out_p, lo+1, hi))

    if args.discord:
        for i, (p, lo, hi) in enumerate(out_paths):
            msg = (f'📉 **CADEvolve low-ESS families** — page {i+1}/{len(out_paths)} '
                   f'(families {lo}–{hi} of {len(chosen)})\n'
                   f'All families with mean CE_ess < {args.ce_ess_thr*100:.0f}%, '
                   f'sorted by CE_ess asc. Each row uses the canonical '
                   f'per-family case (median Q3VL IoU).\n'
                   f'Numbers next to family name: n / CE_ess / Q_ess / CE_IoU / Q_IoU '
                   f'(family means).')
            ok = _post(msg, p)
            print(f'  page {i+1} → {"sent" if ok else "FAILED"}', flush=True)
            time.sleep(2)


if __name__ == '__main__':
    main()
