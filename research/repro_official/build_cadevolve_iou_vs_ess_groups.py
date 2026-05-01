"""Compare CADEvolve vs Q3VL on two groups defined by CADEvolve's IoU+ESS:

  Group A: CE IoU ≥ 0.85 AND CE ESS=False — geometry right, vocab wrong
  Group B: CE IoU ≤ 0.40 AND CE ESS=False — both wrong

For each group, pick 6 cases, render GT + CADEvolve + Q3VL side-by-side
with full op annotations. Posts both grids to Discord.

Usage:
    set -a; source .env; set +a
    uv run python research/repro_official/build_cadevolve_iou_vs_ess_groups.py --discord
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
sys.path.insert(0, str(REPO / 'research/essential_ops'))

EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'
RENDER_CACHE = Path('/tmp/cad_bench_722_renders')
OUT_DIR = EVAL_ROOT / 'iou_vs_ess_groups'

PRED_DIR = {
    'CADEvolve v3': EVAL_ROOT / 'cadevolve_rl1',
    'Q3VL (ours)':  EVAL_ROOT / 'cadrille_qwen3vl_v3',
}
META_PATH = {
    'CADEvolve v3': EVAL_ROOT / 'cadevolve_rl1' / 'metadata.jsonl',
    'Q3VL (ours)':  EVAL_ROOT / 'cadrille_qwen3vl_v3' / 'metadata.jsonl',
}
SLUG = {'CADEvolve v3': 'cadevolve_rl1', 'Q3VL (ours)': 'cadrille_qwen3vl_v3'}

SIDE     = 256
LABEL_H  = 22
ANNOT_H  = 88
NUM_W    = 110
PAD      = 4
HEADER_H = 70

N_PER_GROUP = 6


# ── render worker ──────────────────────────────────────────────────────────

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
    tile = side // 2
    color = (1.0, 1.0, 0.534)
    fronts = [(1,1,1),(-1,-1,-1),(-1,1,-1),(1,-1,1)]
    tiles = []
    for fx, fy, fz in fronts:
        pl = pv.Plotter(off_screen=True, window_size=(tile, tile))
        pl.background_color = (0.07,0.07,0.07)
        pl.add_mesh(mesh, color=color, lighting=True, smooth_shading=True)
        pl.camera_position = [(fx*1.6, fy*1.6, fz*1.6),(0,0,0),(0,0,1)]
        pl.enable_parallel_projection(); pl.camera.zoom(1.4)
        arr = pl.screenshot(None, return_img=True); pl.close()
        tiles.append(arr)
    canvas = np.vstack([np.hstack([tiles[0],tiles[1]]),
                        np.hstack([tiles[2],tiles[3]])])
    return Image.fromarray(canvas).convert('RGB').resize((side,side), Image.LANCZOS)


def _render_one(args):
    slug, stem, py_path, timeout_sec, cache_dir = args
    cache_path = Path(cache_dir) / f'{slug}__{stem}.png'
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path), 'error': None}
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
            tmp_png = str(cache_path) + '.tmp'
            r2 = subprocess.run([sys.executable, '-c', _RENDER_TMPL.format(repo=str(REPO)), stl, tmp_png],
                                capture_output=True, timeout=20)
            if r2.returncode == 0 and Path(tmp_png).exists() and Path(tmp_png).stat().st_size > 0:
                Path(tmp_png).rename(cache_path)
                return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path), 'error': None}
            try: Path(tmp_png).unlink(missing_ok=True)
            except Exception: pass
            img = _render_4view_pyvista(stl, side=268)
            img.save(cache_path, format='PNG')
            return {'slug': slug, 'stem': stem, 'cache_path': str(cache_path),
                    'error': 'render_img_segfault_pyvista_fallback'}
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
    d = ImageDraw.Draw(img); f = _font(18, bold=True)
    bbox = d.textbbox((0,0), msg, font=f); w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
    d.text(((side-w)/2,(side-h)/2), msg, fill=(220,90,90), font=f)
    return img


def _annot_cell(img, label, iou, ep, f1, gen_ops, essential_set):
    from PIL import Image, ImageDraw
    side = img.width
    canvas = Image.new('RGB', (side, side+LABEL_H+ANNOT_H), color=(18,18,18))
    canvas.paste(img, (0, LABEL_H))
    d = ImageDraw.Draw(canvas)
    f_lab = _font(11, bold=True); f_val = _font(10)
    d.rectangle([0,0,side,LABEL_H], fill=(35,35,50))
    d.text((4,3), label, fill=(225,225,225), font=f_lab)
    y0 = side + LABEL_H

    if ep is True:    ep_str, ep_color = 'ESS✓', (90,220,100)
    elif ep is False: ep_str, ep_color = 'ESS✗', (240,90,90)
    else:             ep_str, ep_color = 'ESS—', (160,160,160)
    iou_str = f'IoU={iou:.2f}' if iou is not None else 'IoU=—'
    f1_str  = f'F1={f1:.2f}' if f1 is not None else ''
    d.text((4, y0+2), iou_str, fill=(225,225,225), font=f_lab)
    d.text((side-78, y0+2), ep_str, fill=ep_color, font=f_lab)
    d.text((side-36, y0+2), f1_str, fill=(220,220,220), font=f_val)

    if gen_ops is None:
        d.text((4, y0+18), '(no pred)', fill=(160,160,160), font=f_val)
        return canvas
    used = set(gen_ops)
    pairs = []
    for op in sorted(used):
        pairs.append((op, (90,220,100) if op in essential_set else (170,170,170)))
    for op in sorted(essential_set - used):
        pairs.append(('!' + op, (240,90,90)))
    line_x, line_y, line_h = 4, y0+18, 13
    for op, col in pairs:
        text = op + ' '
        bbox = d.textbbox((0,0), text, font=f_val)
        w = bbox[2] - bbox[0]
        if line_x + w > side - 4: line_y += line_h; line_x = 4
        d.text((line_x, line_y), text, fill=col, font=f_val)
        line_x += w
    return canvas


def _number_cell(idx, family, ce_iou, q_iou, width, height):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(20,20,20))
    d = ImageDraw.Draw(img)
    fbig = _font(20, bold=True); fmid = _font(11, bold=True); fsm = _font(9)
    s = f'#{idx}'
    bbox = d.textbbox((0,0), s, font=fbig); w = bbox[2]-bbox[0]
    d.text(((width-w)/2, 4), s, fill=(240,240,240), font=fbig)
    fam = (family or '?')[:18]
    bbox = d.textbbox((0,0), fam, font=fmid); w = bbox[2]-bbox[0]
    d.text(((width-w)/2, 32), fam, fill=(255,200,100), font=fmid)
    l1 = f'CE {ce_iou:.2f}'
    l2 = f'Q  {q_iou:.2f}'
    bbox = d.textbbox((0,0), l1, font=fsm); w = bbox[2]-bbox[0]
    d.text(((width-w)/2, 56), l1, fill=(220,140,140), font=fsm)
    bbox = d.textbbox((0,0), l2, font=fsm); w = bbox[2]-bbox[0]
    d.text(((width-w)/2, 72), l2, fill=(140,220,140), font=fsm)
    return img


def _page_header(width, height, title, sub):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(35,35,50))
    d = ImageDraw.Draw(img); f1 = _font(15, bold=True); f2 = _font(11)
    d.text((10, 8), title, fill=(230,230,230), font=f1)
    d.text((10, 32), sub, fill=(180,200,230), font=f2)
    d.text((10, 50), 'cols: # | family | GT | CADEvolve v3 | Q3VL (ours)   '
                    'green=essential matched, !red=missing, gray=extra',
           fill=(180,180,200), font=f2)
    return img


def _post(content, attachment):
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        try: url = open(REPO / '.env').read().split('DISCORD_WEBHOOK_URL=')[1].split()[0]
        except Exception: return False
    if not url: return False
    boundary = uuid.uuid4().hex; body = io.BytesIO()
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
        'User-Agent': 'cad-iou-vs-ess-groups/1.0',
    })
    try: urllib.request.urlopen(req, timeout=30).read(); return True
    except Exception as e: print(f'Discord failed: {e}'); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-group', type=int, default=N_PER_GROUP)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--task-timeout', type=int, default=30)
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)

    # Load data
    metas = {}
    for label, p in META_PATH.items():
        metas[label] = {}
        for line in open(p):
            try: r = json.loads(line); metas[label][r['stem']] = r
            except Exception: pass

    ess = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    ess_per_case = {slug: {c['stem']: c for c in m['per_case']}
                    for slug, m in ess['models'].items()}

    from canonical_ops import ESSENTIAL_BY_FAMILY
    def ess_set(fam):
        spec = ESSENTIAL_BY_FAMILY.get(fam) or []
        out = set()
        for el in spec:
            if isinstance(el, str): out.add(el)
            else: out.update(el)
        return out

    # Find candidates
    ce_per_case = ess_per_case['cadevolve_rl1']
    group_a, group_b = [], []
    for stem, c in ce_per_case.items():
        if c.get('essential_pass') is not False: continue
        ce_iou = (metas['CADEvolve v3'].get(stem) or {}).get('iou')
        q_iou  = (metas['Q3VL (ours)'].get(stem) or {}).get('iou')
        if ce_iou is None: continue
        # require both exec_ok for renderable contrast
        if (metas['CADEvolve v3'].get(stem) or {}).get('error_type') != 'success': continue
        if (metas['Q3VL (ours)'].get(stem) or {}).get('error_type') != 'success': continue
        if ce_iou >= 0.85:
            group_a.append((stem, c['family'], ce_iou, q_iou or 0))
        if ce_iou <= 0.40:
            group_b.append((stem, c['family'], ce_iou, q_iou or 0))

    # Diversify by family — at most 1 case per family per group
    def diversify(lst, n, sort_key):
        lst.sort(key=sort_key)
        seen, out = set(), []
        for stem, fam, ci, qi in lst:
            if fam in seen: continue
            seen.add(fam); out.append((stem, fam, ci, qi))
            if len(out) >= n: break
        return out

    # Group A: prefer cases where Q3VL also low (so contrast highlights vocab gap)
    group_a = diversify(group_a, args.n_per_group, lambda t: -t[2])
    # Group B: prefer cases where Q3VL is also low (joint failure contrast)
    group_b = diversify(group_b, args.n_per_group, lambda t: t[2])

    print(f'\nGroup A (CE IoU≥0.85, ESS=False): {len(group_a)} picked')
    for stem, fam, ci, qi in group_a: print(f'  {fam:<22} {stem:<42}  CE={ci:.2f} Q={qi:.2f}')
    print(f'\nGroup B (CE IoU≤0.40, ESS=False): {len(group_b)} picked')
    for stem, fam, ci, qi in group_b: print(f'  {fam:<22} {stem:<42}  CE={ci:.2f} Q={qi:.2f}')

    # GT images
    print('\nFetching GT …', flush=True)
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    all_stems = {t[0] for t in group_a} | {t[0] for t in group_b}
    gt_by_stem = {row['stem']: row['composite_png'] for row in ds if row['stem'] in all_stems}

    # Render preds
    tasks = []
    for stem, fam, _, _ in group_a + group_b:
        for label in ('CADEvolve v3', 'Q3VL (ours)'):
            slug = SLUG[label]
            py = PRED_DIR[label] / f'{stem}.py'
            if py.exists():
                tasks.append((slug, stem, str(py), args.task_timeout, str(RENDER_CACHE)))
    pending = [t for t in tasks if not (RENDER_CACHE / f'{t[0]}__{t[1]}.png').exists()]
    print(f'Render: {len(tasks)} total, {len(pending)} pending', flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=10) as pool:
            for fut in as_completed({pool.submit(_render_one, t): t for t in pending}):
                try: fut.result(timeout=args.task_timeout * 2)
                except Exception as e: print(f'  ! {e}')

    # Build figures
    from PIL import Image
    def build_grid(group, title, sub, fname):
        cell_h = LABEL_H + SIDE + ANNOT_H
        page_w = NUM_W + 3 * (SIDE + PAD)  # GT + CE + Q3VL
        page_h = HEADER_H + len(group) * (cell_h + PAD)
        page = Image.new('RGB', (page_w, page_h), color=(10,10,10))
        page.paste(_page_header(page_w, HEADER_H, title, sub), (0, 0))
        for ri, (stem, fam, ce_iou, q_iou) in enumerate(group):
            y = HEADER_H + ri * (cell_h + PAD)
            page.paste(_number_cell(ri+1, fam, ce_iou, q_iou, NUM_W, cell_h), (0, y))
            x = NUM_W + PAD
            es = ess_set(fam)
            # GT
            gt = gt_by_stem.get(stem)
            if gt:
                gt_img = gt.convert('RGB').resize((SIDE, SIDE), Image.LANCZOS)
                page.paste(_annot_cell(gt_img, 'GT', None, None, None, sorted(es), es), (x, y))
            else:
                page.paste(_annot_cell(_fail_tile(SIDE, 'NO GT'), 'GT', None, None, None, [], set()), (x, y))
            x += SIDE + PAD
            # 2 model cols
            for label in ('CADEvolve v3', 'Q3VL (ours)'):
                slug = SLUG[label]
                rec = metas[label].get(stem) or {}
                ec  = ess_per_case.get(slug, {}).get(stem, {})
                iou = rec.get('iou'); ep = ec.get('essential_pass'); f1 = ec.get('feature_f1')
                gen_ops = ec.get('gen_ops')
                cp = RENDER_CACHE / f'{slug}__{stem}.png'
                if cp.exists() and cp.stat().st_size > 0:
                    img = Image.open(cp).convert('RGB').resize((SIDE, SIDE), Image.LANCZOS)
                else:
                    et = rec.get('error_type', 'no pred')
                    img = _fail_tile(SIDE, et.upper())
                page.paste(_annot_cell(img, label, iou, ep, f1, gen_ops, es), (x, y))
                x += SIDE + PAD
        out_p = OUT_DIR / fname
        page.save(out_p, optimize=True)
        sz = out_p.stat().st_size / 1024 / 1024
        print(f'  → {fname}  {page.size[0]}×{page.size[1]}  {sz:.1f}MB', flush=True)
        return out_p

    out_a = build_grid(group_a,
                       'Group A: CADEvolve geometry-right, vocab-wrong',
                       'CE IoU ≥ 0.85 AND CE ESS=False — model nailed shape but used non-canonical ops',
                       'group_A_high_iou_low_ess.png')
    out_b = build_grid(group_b,
                       'Group B: CADEvolve both wrong (geometry + vocab)',
                       'CE IoU ≤ 0.40 AND CE ESS=False — model failed shape AND used non-canonical ops',
                       'group_B_low_iou_low_ess.png')

    if args.discord:
        time.sleep(1)
        msg_a = ('🅰️ **Group A — CADEvolve 几何对、词汇错** (CE IoU ≥ 0.85, ESS=✗)\n'
                 '6 个 family 各 1 例：CADEvolve 几何 IoU ≥ 0.85 但 strict ESS 仍判 fail '
                 '(几何用别的方法做的，没用 spec 要求的 canonical op)。同 case 看 Q3VL '
                 '怎么处理 — 多数情况下我们用 canonical 词汇但几何也没那么好。')
        _post(msg_a, out_a)
        time.sleep(2)
        msg_b = ('🅱️ **Group B — CADEvolve 两边都错** (CE IoU ≤ 0.40, ESS=✗)\n'
                 '6 个 family 各 1 例：CADEvolve 几何 IoU ≤ 0.40 且 ESS fail。这是真的硬骨头 '
                 'case — 通常是 helical / impeller / 复杂 array 这类。看 Q3VL 是不是也跪。')
        _post(msg_b, out_b)


if __name__ == '__main__':
    main()
