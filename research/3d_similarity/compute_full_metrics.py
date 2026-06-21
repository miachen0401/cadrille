"""Compute extended per-case metrics for ALL cad_bench_722 successful preds.

Beyond the IoU / IoU-24 / CD already in metadata.jsonl, this script adds:
  - fscore_05 : F-score @ τ=0.05 (geometry-with-tolerance)
  - dino_cos  : DINOv2-S [CLS] cosine on rendered 4-view collages
  - lpips     : LPIPS-AlexNet perceptual distance (lower = better)
  - ssim, psnr: pixel-level baselines

Pipeline:
  1. Build STL cache for every successful pred + every GT (one cadquery exec
     per mesh, parallel via 6-process pool with the same SIGALRM + wall-clock
     cap + cancel_futures shutdown hardening as elsewhere).
  2. Reuse the PNG cache built by build_full_grid.py for image inputs;
     GT image is the upstream composite_png from BenchCAD/cad_bench_722.
  3. Sequentially compute per-case metrics in the main process (image
     metrics use lazy-loaded models cached on the function objects).
  4. Emit eval_outputs/cad_bench_722/metrics_per_case_full.json.

Usage (smoke):
    uv run python research/3d_similarity/compute_full_metrics.py --limit 50

Usage (full):
    uv run python research/3d_similarity/compute_full_metrics.py
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutTimeout
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'
PNG_CACHE = Path('/tmp/cad_bench_722_renders')      # from build_full_grid.py
STL_CACHE = Path('/tmp/cad_bench_722_stls')         # new
OUT_JSON  = EVAL_ROOT / 'metrics_per_case_full.json'

MODELS = [
    ('cadrille_rl',         'Cadrille-rl'),
    ('cadevolve_rl1',       'CADEvolve-rl1'),
    ('qwen25vl_3b_zs',      'Qwen-zs'),
    ('cadrille_qwen3vl_v3', 'Cadrille-Q3VL-v3'),
]


# ---------------------------------------------------------------------------
# STL builder worker (exec arbitrary cadquery code → normalised STL on disk)
# ---------------------------------------------------------------------------

class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout('budget')


_EXEC_TMPL = textwrap.dedent('''\
    import sys, io
    import cadquery as cq
    import trimesh, numpy as np
    show_object = lambda *a, **kw: None

    {code}

    _r = locals().get("result") or locals().get("r")
    if _r is None: raise ValueError("no result")
    compound = _r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    if len(verts) < 4 or len(faces) < 4: raise ValueError("degenerate")
    mesh = trimesh.Trimesh([(v.x,v.y,v.z) for v in verts], faces)
    # Normalise to [-1, 1]^3 — same convention as compute_iou
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    ext = float(np.max(mesh.extents))
    if ext > 1e-7:
        mesh.apply_scale(2.0 / ext)
    mesh.export(sys.argv[1])
''')


def _stl_worker(args) -> dict:
    cache_path, code, timeout_sec = args
    cp = Path(cache_path)
    if cp.exists() and cp.stat().st_size > 100:
        return {'cache_path': str(cp), 'error': None}
    if timeout_sec > 0:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout_sec))
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        script = _EXEC_TMPL.format(code=code)
        r = subprocess.run([sys.executable, '-c', script, str(cp)],
                           capture_output=True, timeout=min(timeout_sec, 30))
        if r.returncode != 0 or not cp.exists() or cp.stat().st_size < 100:
            return {'cache_path': None, 'error': 'exec_fail'}
        return {'cache_path': str(cp), 'error': None}
    except Exception as e:
        kind = 'timeout' if isinstance(e, _Timeout) else type(e).__name__
        return {'cache_path': None, 'error': f'{kind}: {str(e)[:60]}'}
    finally:
        if timeout_sec > 0:
            signal.alarm(0)


def normalised_mesh(stl_path: str):
    """Load already-normalised STL into trimesh."""
    import trimesh
    return trimesh.load(stl_path, force='mesh')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit',        type=int, default=0,
                    help='Cap total cases (smoke). 0 = all 720.')
    ap.add_argument('--workers',      type=int, default=6)
    ap.add_argument('--task-timeout', type=int, default=30)
    ap.add_argument('--tau',          type=float, default=0.05)
    ap.add_argument('--out',          default=str(OUT_JSON))
    args = ap.parse_args()

    PNG_CACHE.mkdir(parents=True, exist_ok=True)
    STL_CACHE.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata
    print('Loading metadata …', flush=True)
    metas = {}
    for slug, _ in MODELS:
        d = {}
        meta_path = EVAL_ROOT / slug / 'metadata_24.jsonl'
        if not meta_path.exists():
            meta_path = EVAL_ROOT / slug / 'metadata.jsonl'
        with open(meta_path) as f:
            for line in f:
                try: r = json.loads(line); d[r['stem']] = r
                except: pass
        metas[slug] = d
    all_stems = sorted(set().union(*[set(m.keys()) for m in metas.values()]))
    if args.limit:
        all_stems = all_stems[:args.limit]
    n_cases = len(all_stems)
    print(f'  {n_cases} cases', flush=True)

    # 2. Load GT codes + composite_png from upstream
    print('Loading GT (codes + composite_png) from BenchCAD/cad_bench_722 …', flush=True)
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN')
    ds = load_dataset('BenchCAD/cad_bench_722', split='train', token=token)
    keep = set(all_stems)
    gt_by_stem: dict = {}
    for row in ds:
        if row['stem'] in keep:
            gt_by_stem[row['stem']] = {
                'gt_code': row['gt_code'],
                'composite_png': row['composite_png'],
            }
    print(f'  {len(gt_by_stem)} GT entries', flush=True)

    # 3. Build STL cache (gt + each successful pred)
    print('Building STL cache …', flush=True)
    tasks = []
    # GT STLs
    for stem in all_stems:
        gt_code = gt_by_stem.get(stem, {}).get('gt_code')
        if gt_code is None: continue
        cp = STL_CACHE / f'_gt__{stem}.stl'
        if cp.exists() and cp.stat().st_size > 100: continue
        tasks.append((str(cp), gt_code, args.task_timeout))
    # Pred STLs
    for slug, _ in MODELS:
        for stem in all_stems:
            rec = metas[slug].get(stem, {})
            if rec.get('error_type') != 'success': continue
            py = EVAL_ROOT / slug / f'{stem}.py'
            if not py.exists(): continue
            cp = STL_CACHE / f'{slug}__{stem}.stl'
            if cp.exists() and cp.stat().st_size > 100: continue
            tasks.append((str(cp), py.read_text(), args.task_timeout))
    print(f'  {len(tasks)} STLs to build', flush=True)

    if tasks:
        t0 = time.time(); done = 0
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   max_tasks_per_child=100)
        try:
            futs = [pool.submit(_stl_worker, t) for t in tasks]
            iter_timeout = max(args.task_timeout * 4, 240)
            try:
                for fut in as_completed(futs, timeout=iter_timeout):
                    done += 1
                    if done % 100 == 0:
                        rate = done / (time.time() - t0 + 1e-6)
                        eta = (len(tasks) - done) / max(rate, 1e-6) / 60
                        print(f'    [{done}/{len(tasks)}] {rate:.2f}/s ETA {eta:.1f}min',
                              flush=True)
            except FutTimeout:
                pending = sum(1 for f in futs if not f.done())
                print(f'  !! {pending} STL build(s) hung > {iter_timeout}s — '
                      f'abandoning, proceeding', flush=True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        print(f'  STL phase done in {(time.time()-t0)/60:.1f}min', flush=True)

    # 4. Compute per-case metrics
    print('Computing per-case metrics …', flush=True)
    from PIL import Image
    from geom_metrics  import fscore_at_tau
    from image_metrics import lpips_distance, ssim_score, dino_cos, psnr

    out: dict = {'tau': args.tau, 'cases': {}}
    t0 = time.time()
    for ci, stem in enumerate(all_stems):
        gt_stl = STL_CACHE / f'_gt__{stem}.stl'
        gt_img = gt_by_stem.get(stem, {}).get('composite_png')
        if not gt_stl.exists() or gt_img is None:
            out['cases'][stem] = {'family': metas[MODELS[0][0]].get(stem, {}).get('family'),
                                  'difficulty': metas[MODELS[0][0]].get(stem, {}).get('difficulty'),
                                  'models': {slug: {'error': 'no_gt_stl_or_img'}
                                             for slug, _ in MODELS}}
            continue
        gt_mesh = normalised_mesh(str(gt_stl))
        per_model = {}
        for slug, _ in MODELS:
            rec = metas[slug].get(stem, {})
            base = {
                'iou':    rec.get('iou'),
                'iou_24': rec.get('iou_24'),
                'rot_idx': rec.get('rot_idx', -1),
                'cd':     rec.get('cd'),
                'error_type': rec.get('error_type'),
            }
            if rec.get('error_type') != 'success':
                per_model[slug] = base
                continue
            pred_stl = STL_CACHE / f'{slug}__{stem}.stl'
            pred_png = PNG_CACHE / f'{slug}__{stem}.png'
            if not pred_stl.exists() or not pred_png.exists():
                base['error'] = 'cache_miss'
                per_model[slug] = base
                continue
            try:
                pred_mesh = normalised_mesh(str(pred_stl))
                pred_img  = Image.open(pred_png).convert('RGB')
                f_, _, _ = fscore_at_tau(gt_mesh, pred_mesh, tau=args.tau)
                base['fscore_05'] = round(f_, 4) if f_ is not None else None
                d = dino_cos(gt_img, pred_img)
                base['dino_cos']  = round(d, 4) if d is not None else None
                lp = lpips_distance(gt_img, pred_img)
                base['lpips']     = round(lp, 4) if lp is not None else None
                s = ssim_score(gt_img, pred_img)
                base['ssim']      = round(s, 4) if s is not None else None
                p = psnr(gt_img, pred_img)
                base['psnr']      = round(p, 4) if p is not None else None
            except Exception as e:
                base['error'] = f'metric_err:{type(e).__name__}'
            per_model[slug] = base
        out['cases'][stem] = {
            'family':     metas[MODELS[0][0]].get(stem, {}).get('family'),
            'difficulty': metas[MODELS[0][0]].get(stem, {}).get('difficulty'),
            'models':     per_model,
        }
        if (ci + 1) % 50 == 0:
            rate = (ci + 1) / (time.time() - t0 + 1e-6)
            eta  = (n_cases - ci - 1) / max(rate, 1e-6) / 60
            print(f'  [{ci + 1}/{n_cases}] {rate:.2f}/s ETA {eta:.1f}min', flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f'\nWrote {args.out}', flush=True)
    print(f'  total cases: {len(out["cases"])}', flush=True)


if __name__ == '__main__':
    main()
