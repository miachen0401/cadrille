"""Score Cadrille-rl paper-repro preds on DeepCAD-300 + Fusion360-300 using
common.metrics.compute_metrics — same metric pipeline as the v3 pipeline,
so IoU/CD numbers are directly comparable across all 4 models.

Why this script exists:
  /tmp/score_cadrille_rl_ood.py used ProcessPoolExecutor wrapping
  compute_metrics (which itself spawns subprocesses). That double-process
  setup deadlocked at 200/300 — when an inner subprocess timed out and
  was killed, the outer worker process was left holding stale state, and
  ProcessPool's broken-pool recovery never kicked in for our case.

Fix: ThreadPoolExecutor. compute_metrics already spawns its own subprocess
with a hard `subprocess.run(timeout=30)`, so threads in the parent just
wait on subprocess completion — no extra fork barrier, no deadlock.

Usage:
    .venv-eval/bin/python research/repro_official/score_ood_unified.py
    # writes metadata.jsonl into eval_outputs/<dataset>_n300/cadrille_rl_repro/

Run with .venv-eval (paper-repro env): the pred .py files were generated
against transformers 4.50.3, but compute_metrics doesn't touch the model —
it just execs the cadquery code, so .venv (main) works equally well. Pick
either; both share common.metrics.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _score_one(pred_py, gt_stl, timeout):
    """Score a single (pred, gt) pair via common.metrics in subprocess."""
    from common.metrics import compute_metrics
    try:
        code = Path(pred_py).read_text()
        iou, cd = compute_metrics(code, gt_stl, timeout=timeout, use_pool=False)
        return {
            'iou': float(iou) if iou is not None and iou >= 0 else None,
            'cd':  float(cd)  if cd  is not None else None,
            'error_type': 'success' if (iou is not None and iou >= 0) else 'runtime_error',
        }
    except Exception as e:
        return {'iou': None, 'cd': None, 'error_type': f'exc:{type(e).__name__}'}


def main():
    SEED = 42
    N = 300
    TIMEOUT = 30  # subprocess timeout per case
    MAX_WORKERS = 4  # threads — each spawns its own subprocess

    for ds_label, stl_dir, pred_dir, out_dir in [
        ('DeepCAD',
         REPO / 'data/deepcad_test_mesh',
         REPO / 'eval_outputs/repro_official/deepcad_test_mesh_n300/py',
         REPO / 'eval_outputs/deepcad_n300/cadrille_rl_repro'),
        ('Fusion360',
         REPO / 'data/fusion360_test_mesh',
         REPO / 'eval_outputs/repro_official/fusion360_test_mesh_n300/py',
         REPO / 'eval_outputs/fusion360_n300/cadrille_rl_repro'),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_out = out_dir / 'metadata.jsonl'
        if meta_out.exists() and len([1 for _ in open(meta_out)]) >= N - 5:
            print(f'{ds_label}: already scored ({sum(1 for _ in open(meta_out))} rows), skipping')
            continue

        # Match the same 300 STLs the paper-repro used (seed=42 shuffle of all)
        all_stls = sorted(p.name for p in stl_dir.iterdir() if p.suffix == '.stl')
        rng = random.Random(SEED); rng.shuffle(all_stls)
        keep = all_stls[:N]
        # paper-repro saved preds as `<stem>+0.py`
        tasks = []
        for stl_fname in keep:
            stem = stl_fname[:-4]
            pred_py = pred_dir / f'{stem}+0.py'
            gt_stl  = stl_dir / stl_fname
            if not pred_py.exists():
                continue
            tasks.append((str(pred_py), str(gt_stl), TIMEOUT, stem))
        print(f'{ds_label}: {len(tasks)}/{N} preds to score', flush=True)

        results = []
        t0 = time.time(); done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(_score_one, t[0], t[1], t[2]): t[3] for t in tasks}
            for fut in as_completed(futs):
                stem = futs[fut]
                try:
                    r = fut.result(timeout=TIMEOUT + 30)  # subprocess timeout + slack
                except Exception as e:
                    r = {'iou': None, 'cd': None,
                         'error_type': f'fut_err:{type(e).__name__}'}
                results.append({'stem': stem, **r})
                done += 1
                if done % 25 == 0:
                    rate = done / (time.time() - t0 + 1e-6)
                    eta = (len(tasks) - done) / max(rate, 1e-6) / 60
                    print(f'  [{done}/{len(tasks)}] {rate:.1f}/s ETA {eta:.1f}min', flush=True)

        with open(meta_out, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

        ok = [r for r in results if r['error_type'] == 'success']
        ious = [r['iou'] for r in ok if r['iou'] is not None]
        cds  = [r['cd']  for r in ok if r['cd']  is not None]
        if ious:
            print(f'  {ds_label}: n={len(results)}  exec={len(ok)}/{len(results)}  '
                  f'mean_iou={sum(ious)/len(ious):.4f}  mean_cd={sum(cds)/len(cds):.4f}  '
                  f'→ {meta_out}', flush=True)
        else:
            print(f'  {ds_label}: n={len(results)}  exec={len(ok)}/{len(results)}  '
                  f'NO SUCCESSFUL SCORES', flush=True)


if __name__ == '__main__':
    main()
