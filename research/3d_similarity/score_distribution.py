"""Score the 3 cad_bench_722 baselines with distribution-level metrics:
FID, KID, CLIP R-Precision @ {1, 5, 10}.

Pipeline:
  1. Load all 720 GT composite_png from BenchCAD/cad_bench_722.
  2. For each model: render all successful pred meshes to 268×268 4-view
     PIL images (in a 6-process pool with SIGALRM + wall-clock cap).
  3. Inception-V3 pool features for FID/KID.
  4. CLIP image features for R-Precision (paired: pred_i ↔ gt_i).
  5. Save eval_outputs/cad_bench_722/distribution_metrics.json
     and post a markdown table to Discord.

Usage:
    set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
    uv run python research/3d_similarity/score_distribution.py --discord
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'
OUT_JSON  = EVAL_ROOT / 'distribution_metrics.json'

MODELS = [
    ('cadrille_rl',         'Cadrille-rl'),
    ('cadevolve_rl1',       'CADEvolve-rl1'),
    ('qwen25vl_3b_zs',      'Qwen-zs'),
    ('cadrille_qwen3vl_v3', 'Cadrille-Q3VL-v3'),
]


# ---------------------------------------------------------------------------
# Render worker
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
    """Worker: pred .py path → 4-view 268×268 PNG bytes."""
    idx, py_path, timeout_sec = args
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
                return {'idx': idx, 'png_bytes': None, 'error': 'exec_fail'}
            from common.meshio import render_img
            img = render_img(stl)['video'][0]
            buf = io.BytesIO(); img.save(buf, format='PNG')
            return {'idx': idx, 'png_bytes': buf.getvalue(), 'error': None}
        finally:
            try: Path(stl).unlink()
            except Exception: pass
    except Exception as e:
        kind = 'timeout' if isinstance(e, _Timeout) else type(e).__name__
        return {'idx': idx, 'png_bytes': None, 'error': f'{kind}: {str(e)[:80]}'}
    finally:
        if timeout_sec > 0:
            signal.alarm(0)


# ---------------------------------------------------------------------------
# Discord upload
# ---------------------------------------------------------------------------

def post_to_discord(text: str, file_path: Optional[Path] = None) -> None:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  no DISCORD_WEBHOOK_URL — skipping ping'); return
    import urllib.request
    if file_path is None:
        data = json.dumps({'content': text}).encode()
        req = urllib.request.Request(url, data=data,
            headers={'Content-Type': 'application/json',
                     'User-Agent': 'cad-bench-722-dist/1.0'})
    else:
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        def w(s: str): body.write(s.encode())
        w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n'
          f'Content-Type: application/json\r\n\r\n{json.dumps({"content": text})}\r\n')
        w(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
          f'filename="{file_path.name}"\r\nContent-Type: application/json\r\n\r\n')
        body.write(file_path.read_bytes()); w('\r\n')
        w(f'--{boundary}--\r\n')
        req = urllib.request.Request(url, data=body.getvalue(),
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}',
                     'User-Agent': 'cad-bench-722-dist/1.0'})
    try:
        urllib.request.urlopen(req, timeout=20).read()
        print('  posted to Discord ✓')
    except Exception as e:
        print(f'  Discord post failed: {e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers',  type=int, default=6)
    ap.add_argument('--task-timeout', type=int, default=30)
    ap.add_argument('--limit',    type=int, default=0,
                    help='Per-model cap on # successful preds rendered (0=all). For smoke.')
    ap.add_argument('--discord',  action='store_true')
    ap.add_argument('--out',      default=str(OUT_JSON))
    args = ap.parse_args()

    print('Loading metadata for all 3 models …', flush=True)
    metas = {}
    for slug, _ in MODELS:
        with open(EVAL_ROOT / slug / 'metadata.jsonl') as f:
            metas[slug] = {(r := json.loads(line))['stem']: r for line in f
                           if line.strip()}
    n720 = max(len(d) for d in metas.values())
    print(f'  loaded ({n720} samples each)', flush=True)

    print('Loading 720 GT composite_png from BenchCAD/cad_bench_722 …', flush=True)
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN')
    ds = load_dataset('BenchCAD/cad_bench_722', split='train', token=token)
    gt_by_stem = {row['stem']: row['composite_png'] for row in ds
                  if row['composite_png'] is not None}
    print(f'  loaded {len(gt_by_stem)} non-null GT images', flush=True)

    # Pre-compute Inception + CLIP features for the FULL 720 GT pool ONCE.
    print('\n=== GT features (Inception + CLIP) ===', flush=True)
    gt_stems  = sorted(gt_by_stem.keys())
    gt_images = [gt_by_stem[s] for s in gt_stems]
    from distribution_metrics import (
        inception_pool_features, clip_image_features,
        fid_from_features, kid_from_features, clip_r_precision)
    t0 = time.time()
    gt_inc = inception_pool_features(gt_images)
    print(f'  Inception   GT  shape={gt_inc.shape} ({time.time()-t0:.1f}s)', flush=True)
    t0 = time.time()
    gt_clip = clip_image_features(gt_images)
    print(f'  CLIP        GT  shape={gt_clip.shape} ({time.time()-t0:.1f}s)', flush=True)
    gt_idx_of_stem = {s: i for i, s in enumerate(gt_stems)}

    summary = {'models': {}, 'tau_kid_subset': 100, 'n_gt': len(gt_stems)}

    for slug, label in MODELS:
        print(f'\n=== {label} ({slug}) ===', flush=True)
        # Successful preds with both pred .py and a matched GT
        good_stems = sorted(s for s, r in metas[slug].items()
                            if r.get('error_type') == 'success'
                            and (EVAL_ROOT / slug / f'{s}.py').exists()
                            and s in gt_idx_of_stem)
        if args.limit:
            good_stems = good_stems[:args.limit]
        n = len(good_stems)
        print(f'  rendering {n} successful preds (workers={args.workers}) …', flush=True)
        t0 = time.time()
        # Dispatch to pool
        png_bytes_by_stem: dict = {}
        from concurrent.futures import as_completed, TimeoutError as FutTimeout
        # Don't use `with ProcessPoolExecutor(...)` — its __exit__ calls
        # shutdown(wait=True) which blocks forever on a dead worker. Manage
        # the pool explicitly so we can call shutdown(cancel_futures=True).
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   max_tasks_per_child=100)
        try:
            futs = {pool.submit(_render_one,
                                (i, str(EVAL_ROOT / slug / f'{s}.py'),
                                 args.task_timeout)): s
                    for i, s in enumerate(good_stems)}
            done = 0
            iter_timeout = max(args.task_timeout * 4, 240)
            try:
                for fut in as_completed(futs, timeout=iter_timeout):
                    s = futs[fut]
                    try:
                        res = fut.result(timeout=args.task_timeout * 2)
                    except Exception as e:
                        res = {'png_bytes': None, 'error': f'fut_err:{e}'}
                    if res.get('png_bytes'):
                        png_bytes_by_stem[s] = res['png_bytes']
                    done += 1
                    if done % 50 == 0:
                        rate = done / (time.time() - t0 + 1e-6)
                        eta_min = (n - done) / max(rate, 1e-6) / 60
                        print(f'    [{done}/{n}] {rate:.2f}/s ETA {eta_min:.1f}min '
                              f'rendered={len(png_bytes_by_stem)}', flush=True)
            except FutTimeout:
                pending = sum(1 for f in futs if not f.done())
                print(f'  !! {pending} future(s) hung past {iter_timeout}s '
                      f'(dead workers) → abandoning, proceeding with '
                      f'{len(png_bytes_by_stem)}/{n} rendered', flush=True)
        finally:
            # cancel_futures=True (Python 3.9+) tells shutdown to drop pending
            # work; wait=False does not block on running tasks. Together they
            # let us escape a deadlocked pool.
            pool.shutdown(wait=False, cancel_futures=True)

        n_rendered = len(png_bytes_by_stem)
        print(f'  rendered {n_rendered}/{n} ({time.time()-t0:.1f}s total)', flush=True)
        if n_rendered == 0:
            summary['models'][slug] = {'n_pred_rendered': 0, 'error': 'nothing rendered'}
            continue

        # Decode pred images, build matched GT subset (for paired R-Precision)
        from PIL import Image
        pred_stems  = sorted(png_bytes_by_stem.keys())
        pred_images = [Image.open(io.BytesIO(png_bytes_by_stem[s])).convert('RGB')
                       for s in pred_stems]
        gt_paired   = [gt_by_stem[s] for s in pred_stems]

        print(f'  computing Inception features for {len(pred_images)} preds …', flush=True)
        t0 = time.time()
        pred_inc = inception_pool_features(pred_images)
        print(f'    done ({time.time()-t0:.1f}s) shape={pred_inc.shape}', flush=True)

        # Match-subset GT inception features (use the cached gt_inc by index)
        gt_paired_inc = np.stack([gt_inc[gt_idx_of_stem[s]] for s in pred_stems])

        # FID / KID against the full 720 GT distribution (more standard)
        print('  FID + KID against full 720 GT distribution …', flush=True)
        t0 = time.time()
        fid_full = fid_from_features(gt_inc, pred_inc)
        kid_mean, kid_std = kid_from_features(
            gt_inc, pred_inc,
            n_subsets=100, subset_size=min(100, len(pred_inc)))
        print(f'    FID={fid_full:.2f}  KID_mean={kid_mean:.4f} ({time.time()-t0:.1f}s)',
              flush=True)

        # CLIP features + R-Precision (paired, against full 720)
        print('  CLIP features + R-Precision …', flush=True)
        t0 = time.time()
        pred_clip = clip_image_features(pred_images)
        # Build paired GT clip features (re-use full gt_clip)
        # For paired R-precision we need to know the index of each pred's GT in
        # the FULL GT pool. Then "correct retrieval" = top-K contains that idx.
        n_gt = len(gt_stems)
        # rankings against full GT pool
        sim = pred_clip @ gt_clip.T
        rankings = np.argsort(-sim, axis=1)
        true_idx = np.array([gt_idx_of_stem[s] for s in pred_stems])
        rp = {}
        for k in (1, 5, 10):
            top_k = rankings[:, :k]
            hits = sum(1 for i in range(len(pred_stems)) if true_idx[i] in top_k[i])
            rp[f'r_at_{k}'] = hits / len(pred_stems)
        print(f'    R@1={rp["r_at_1"]:.3f}  R@5={rp["r_at_5"]:.3f}  '
              f'R@10={rp["r_at_10"]:.3f} ({time.time()-t0:.1f}s)', flush=True)

        summary['models'][slug] = {
            'n_pred_rendered': n_rendered,
            'n_pred_attempted': n,
            'fid_vs_full_720_gt':  round(fid_full, 4),
            'kid_mean':            round(kid_mean, 6),
            'kid_std':             round(kid_std, 6),
            'clip_r_precision':    {k: round(v, 4) for k, v in rp.items()},
        }
        # Persist per-model checkpoint so a later stall doesn't lose this work
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f'  → checkpointed {args.out}', flush=True)

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f'\nWrote {args.out}', flush=True)
    print(json.dumps(summary, indent=2))

    if args.discord:
        # Build a clean Discord-friendly markdown summary
        lines = ['📊 **cad_bench_722 distribution-level metrics** '
                 '(720 GT samples per model; FID/KID lower=better; CLIP R-Precision higher=better)\n']
        lines.append('```')
        lines.append(f'{"model":<22} {"n_pred":>6} {"FID":>8} {"KID":>11} '
                     f'{"R@1":>6} {"R@5":>6} {"R@10":>6}')
        lines.append('-' * 72)
        for slug, label in MODELS:
            d = summary['models'].get(slug, {})
            n = d.get('n_pred_rendered', 0)
            if n == 0 or d.get('fid_vs_full_720_gt') is None:
                lines.append(f'{label:<22} {n:>6}     —            —      —      —      —')
                continue
            fid = d['fid_vs_full_720_gt']
            kid = d['kid_mean']
            rp  = d['clip_r_precision']
            lines.append(f'{label:<22} {n:>6} {fid:>8.2f} {kid:>11.5f} '
                         f'{rp["r_at_1"]:>6.3f} {rp["r_at_5"]:>6.3f} {rp["r_at_10"]:>6.3f}')
        lines.append('```')
        post_to_discord('\n'.join(lines))


if __name__ == '__main__':
    main()
