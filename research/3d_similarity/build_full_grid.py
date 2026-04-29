"""Build per-case visual grids for ALL 720 cad_bench_722 cases, split into
N long PNG images, and post them to Discord.

Each row of the figure is one case:
    [#case]  [GT]  [Cadrille-rl]  [CADEvolve-rl1]  [Qwen2.5-VL-3B-zs]  [scores]
where each render is the same 4-view 268×268 collage style used everywhere
else in the project (GT = upstream composite_png; pred = exec'd code →
trimesh → render_img). Failed cells are dark tiles labelled with the
error class. The score column carries IoU / IoU-24 / CD per model so
the viewer can scan numerical disagreement next to visual disagreement.

Cases are numbered 1..720 in alphabetical-stem order so the same case
keeps the same number across every chunk.

Usage:
    set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
    uv run python research/3d_similarity/build_full_grid.py --discord
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
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutTimeout
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'
OUT_DIR   = EVAL_ROOT / 'full_case_grids'
RENDER_CACHE = Path('/tmp/cad_bench_722_renders')   # PNG bytes by f'{slug}__{stem}.png'

MODELS = [
    ('cadrille_rl',     'Cadrille-rl'),
    ('cadevolve_rl1',   'CADEvolve-rl1'),
    ('qwen25vl_3b_zs',  'Qwen-zs'),
]

SIDE     = 268    # cell pixel size (square)
LABEL_H  = 26     # label band under each cell
NUM_W    = 90     # left numbering column
SCORES_W = 360    # right score column
PAD      = 4
HEADER_H = 36     # column header per page


# ---------------------------------------------------------------------------
# Render worker (pred .py → 4-view PNG bytes)
# ---------------------------------------------------------------------------

class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout('budget')

_EXEC_TMPL = textwrap.dedent('''\
    import sys, io
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


def _render_one(args) -> dict:
    """Worker: (slug, stem, py_path, timeout_sec) → cached PNG path on disk."""
    slug, stem, py_path, timeout_sec, cache_dir = args
    cache_path = Path(cache_dir) / f'{slug}__{stem}.png'
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
            from common.meshio import render_img
            img = render_img(stl)['video'][0]
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


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _font(size, bold=False):
    from PIL import ImageFont
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{name}', size)
    except Exception:
        return ImageFont.load_default()


def fail_tile(side: int, msg: str):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (side, side), color=(28, 28, 28))
    d = ImageDraw.Draw(img)
    f = _font(18, bold=True)
    bbox = d.textbbox((0, 0), msg, font=f)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    d.text(((side - w) / 2, (side - h) / 2), msg, fill=(220, 90, 90), font=f)
    return img


def annotate_cell(img, label: str, color=(220, 220, 220)):
    from PIL import Image, ImageDraw
    side = img.width
    canvas = Image.new('RGB', (side, side + LABEL_H), color=(18, 18, 18))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    f = _font(13)
    bbox = d.textbbox((0, 0), label, font=f)
    w = bbox[2] - bbox[0]
    d.text(((side - w) / 2, side + 5), label, fill=color, font=f)
    return canvas


def number_cell(idx: int, total: int, stem: str, family: str, difficulty: str,
                width: int, height: int):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(20, 20, 20))
    d = ImageDraw.Draw(img)
    fbig = _font(28, bold=True)
    fmid = _font(11)
    fsm  = _font(10)
    # Big case number
    s = f'#{idx}'
    bbox = d.textbbox((0, 0), s, font=fbig)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 4), s, fill=(240, 240, 240), font=fbig)
    # Family / difficulty
    fd = f'{family[:14]}'
    bbox = d.textbbox((0, 0), fd, font=fmid)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 44), fd, fill=(180, 180, 180), font=fmid)
    fd2 = f'[{difficulty}]'
    bbox = d.textbbox((0, 0), fd2, font=fsm)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 60), fd2, fill=(150, 150, 150), font=fsm)
    # Stem (truncated)
    short = stem.replace('synth_', '').replace('dvsub_', 'dv:')[:13]
    bbox = d.textbbox((0, 0), short, font=fsm)
    w = bbox[2] - bbox[0]
    d.text(((width - w) / 2, 76), short, fill=(120, 120, 120), font=fsm)
    return img


def scores_cell(model_records: dict, width: int, height: int):
    """Right-side score column. model_records[slug] = {'iou','cd','iou_24',...}"""
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(15, 15, 15))
    d = ImageDraw.Draw(img)
    f_label = _font(11, bold=True)
    f_val   = _font(11)
    # Per-model row
    row_h = (height - 16) // len(MODELS)
    for mi, (slug, label) in enumerate(MODELS):
        y0 = 8 + mi * row_h
        rec = model_records.get(slug, {})
        et  = rec.get('error_type', 'missing')
        # color band per model
        band_color = (50, 70, 50) if et == 'success' else (60, 40, 40)
        d.rectangle([2, y0, width - 2, y0 + row_h - 4], fill=band_color)
        d.text((6, y0 + 2), label, fill=(225, 225, 225), font=f_label)
        if et != 'success':
            d.text((6, y0 + 16), f'{et}', fill=(220, 150, 150), font=f_val)
            continue
        # IoU / IoU-24 / CD lines
        iou    = rec.get('iou')
        iou24  = rec.get('iou_24')
        rotidx = rec.get('rot_idx', -1)
        cd     = rec.get('cd')
        line1 = f'IoU={iou:.3f}' if iou is not None else 'IoU=—'
        if iou24 is not None:
            line1 += f'  IoU24={iou24:.3f} (r{rotidx})'
        line2 = f'CD={cd:.4f}' if cd is not None else 'CD=—'
        d.text((6, y0 + 16), line1, fill=(225, 225, 225), font=f_val)
        d.text((6, y0 + 30), line2, fill=(180, 180, 200), font=f_val)
    return img


def page_header(width: int, page_idx: int, n_pages: int, case_lo: int, case_hi: int,
                height: int = HEADER_H):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (width, height), color=(35, 35, 50))
    d = ImageDraw.Draw(img)
    f = _font(15, bold=True)
    title = (f'cad_bench_722  —  page {page_idx + 1}/{n_pages}  '
             f'(cases {case_lo}..{case_hi})  '
             f'cols: # | GT | Cadrille-rl | CADEvolve-rl1 | Qwen2.5-VL-3B-zs | scores')
    d.text((10, 10), title, fill=(230, 230, 230), font=f)
    return img


# ---------------------------------------------------------------------------
# Discord upload
# ---------------------------------------------------------------------------

def post_image_to_discord(path: Path, content: str) -> bool:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  (no DISCORD_WEBHOOK_URL — skip)'); return False
    import urllib.request
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def w(s): body.write(s.encode())
    w(f'--{boundary}\r\n'
      f'Content-Disposition: form-data; name="payload_json"\r\n'
      f'Content-Type: application/json\r\n\r\n')
    w(json.dumps({'content': content}) + '\r\n')
    w(f'--{boundary}\r\n'
      f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
      f'Content-Type: image/png\r\n\r\n')
    body.write(path.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    req = urllib.request.Request(url, data=body.getvalue(), headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cad-bench-722-fullgrid/1.0',
    })
    try:
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except Exception as e:
        print(f'  Discord post failed: {e}')
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-chunks',     type=int, default=15,
                    help='Split 720 cases into this many PNGs (default 15 ≈ 48/case each)')
    ap.add_argument('--workers',      type=int, default=6)
    ap.add_argument('--task-timeout', type=int, default=30)
    ap.add_argument('--limit',        type=int, default=0,
                    help='Cap total cases (debug). 0 = all 720.')
    ap.add_argument('--out-dir',      default=str(OUT_DIR))
    ap.add_argument('--discord',      action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata for all 3 models (use metadata_24.jsonl when available
    #    so we get the iou_24 / rot_idx fields)
    print('Loading metadata …', flush=True)
    metas = {}
    for slug, _ in MODELS:
        d24 = {}
        meta_24 = EVAL_ROOT / slug / 'metadata_24.jsonl'
        if meta_24.exists():
            for line in open(meta_24):
                try: r = json.loads(line); d24[r['stem']] = r
                except: pass
        # Fallback: original metadata for any stems missing
        d_orig = {}
        with open(EVAL_ROOT / slug / 'metadata.jsonl') as f:
            for line in f:
                try: r = json.loads(line); d_orig[r['stem']] = r
                except: pass
        # merge: if d24 has it, use that (has iou_24); else fall back
        merged = {}
        for stem in set(d_orig) | set(d24):
            merged[stem] = {**d_orig.get(stem, {}), **d24.get(stem, {})}
        metas[slug] = merged
        print(f'  {slug}: {len(merged)} samples ({len(d24)} with iou_24)', flush=True)

    # 2. Build canonical ordered case list (sorted by stem; numbered 1..720)
    all_stems = sorted(set().union(*[set(m.keys()) for m in metas.values()]))
    if args.limit:
        all_stems = all_stems[:args.limit]
    n_cases = len(all_stems)
    print(f'  total cases: {n_cases}', flush=True)

    # 3. Load GT composite_png for every case
    print('Fetching GT composite_png …', flush=True)
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN')
    ds = load_dataset('BenchCAD/cad_bench_722', split='train', token=token)
    gt_by_stem = {row['stem']: row['composite_png'] for row in ds
                  if row['stem'] in set(all_stems)}
    print(f'  loaded {len(gt_by_stem)} GT images', flush=True)

    # 4. Render every successful pred (cached on disk so reruns are cheap)
    print('Rendering preds (with disk cache) …', flush=True)
    tasks = []
    for slug, _ in MODELS:
        for stem in all_stems:
            if metas[slug].get(stem, {}).get('error_type') != 'success':
                continue
            py_path = EVAL_ROOT / slug / f'{stem}.py'
            if not py_path.exists():
                continue
            tasks.append((slug, stem, str(py_path), args.task_timeout,
                          str(RENDER_CACHE)))

    # Pre-skip cached ones for the progress count
    pending = [t for t in tasks
               if not (RENDER_CACHE / f'{t[0]}__{t[1]}.png').exists()]
    print(f'  {len(tasks)} render tasks total, {len(pending)} not yet cached', flush=True)

    if pending:
        t0 = time.time(); done = 0
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   max_tasks_per_child=100)
        try:
            futs = {pool.submit(_render_one, t): (t[0], t[1]) for t in pending}
            iter_timeout = max(args.task_timeout * 4, 240)
            try:
                for fut in as_completed(futs, timeout=iter_timeout):
                    res = fut.result(timeout=args.task_timeout * 2) \
                          if fut.done() else {'error': 'pending'}
                    done += 1
                    if done % 100 == 0:
                        rate = done / (time.time() - t0 + 1e-6)
                        eta = (len(pending) - done) / max(rate, 1e-6) / 60
                        print(f'    [{done}/{len(pending)}] {rate:.2f}/s ETA {eta:.1f}min',
                              flush=True)
            except FutTimeout:
                pending_left = sum(1 for f in futs if not f.done())
                print(f'  !! {pending_left} render(s) hung > {iter_timeout}s — '
                      f'abandoning, proceeding with cached', flush=True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        print(f'  render phase done in {(time.time() - t0)/60:.1f}min', flush=True)

    # 5. Build chunked figures
    print(f'\nBuilding {args.n_chunks} figures …', flush=True)
    from PIL import Image

    chunk_size = (n_cases + args.n_chunks - 1) // args.n_chunks
    row_height = SIDE + LABEL_H
    page_w = NUM_W + 4 * (SIDE + PAD) + SCORES_W
    out_paths = []

    for ci in range(args.n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n_cases)
        if lo >= hi: break
        chunk_stems = all_stems[lo:hi]

        page_h = HEADER_H + len(chunk_stems) * (row_height + PAD)
        page = Image.new('RGB', (page_w, page_h), color=(10, 10, 10))
        page.paste(page_header(page_w, ci, args.n_chunks, lo + 1, hi), (0, 0))

        for ri, stem in enumerate(chunk_stems):
            y = HEADER_H + ri * (row_height + PAD)
            case_idx = lo + ri + 1   # 1-based
            # case meta from any model
            mref = next((metas[slug][stem] for slug, _ in MODELS
                         if stem in metas[slug]), {})
            family = mref.get('family', '?')
            diff   = mref.get('difficulty', '?')

            # # column
            page.paste(number_cell(case_idx, n_cases, stem, family, diff,
                                   NUM_W, row_height), (0, y))
            # GT column
            x = NUM_W + PAD
            gt = gt_by_stem.get(stem)
            if gt is not None:
                page.paste(annotate_cell(gt.convert('RGB'), 'GT'), (x, y))
            else:
                page.paste(annotate_cell(fail_tile(SIDE, 'NO GT'), 'GT'), (x, y))
            x += SIDE + PAD
            # 3 model columns
            model_records = {}
            for slug, label in MODELS:
                rec = metas[slug].get(stem, {})
                model_records[slug] = rec
                cache_path = RENDER_CACHE / f'{slug}__{stem}.png'
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    img = Image.open(cache_path).convert('RGB')
                    iou = rec.get('iou')
                    iou_label = (f'iou={iou:.3f}' if iou is not None
                                 else label.split('-')[0])
                    page.paste(annotate_cell(img, iou_label), (x, y))
                else:
                    et = rec.get('error_type', 'missing')
                    page.paste(annotate_cell(fail_tile(SIDE, et.upper()), label),
                               (x, y))
                x += SIDE + PAD
            # Scores column
            page.paste(scores_cell(model_records, SCORES_W, row_height), (x, y))

        path = out_dir / f'cases_{lo + 1:04d}-{hi:04d}.png'
        page.save(path, optimize=True)
        sz = path.stat().st_size / 1024 / 1024
        print(f'  page {ci + 1}/{args.n_chunks}: {hi - lo} cases  '
              f'{page.size[0]}×{page.size[1]}  {sz:.1f} MB → {path.name}',
              flush=True)
        out_paths.append((path, lo + 1, hi))

    # 6. Discord upload
    if args.discord:
        print('\nPosting to Discord …', flush=True)
        for i, (p, lo, hi) in enumerate(out_paths):
            desc = (
                f'📦 **cad_bench_722 case-by-case grid** — '
                f'page {i + 1}/{len(out_paths)} (cases {lo}–{hi})\n'
                f'columns: case# | GT | Cadrille-rl | CADEvolve-rl1 | Qwen2.5-VL-3B-zs | scores '
                f'(IoU / IoU-24 / CD per model)\n'
                f'green-band cells = pred exec\'d successfully; red-band cells = error_type'
            )
            ok = post_image_to_discord(p, desc)
            print(f'  page {i + 1}/{len(out_paths)} ({p.name}) → '
                  f'{"sent" if ok else "FAILED"}', flush=True)
            time.sleep(2)  # avoid rate-limit (5 req / 2s on webhooks)

    print(f'\nAll done. {len(out_paths)} pages in {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
