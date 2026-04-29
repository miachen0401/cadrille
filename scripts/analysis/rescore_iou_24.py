"""Re-score already-evaluated cad_bench_722 metadata with rotation-invariant IoU.

For each model dir under --root, reads metadata.jsonl + per-sample <stem>.py,
re-executes pred + GT through the iou_24 subprocess worker, and writes
metadata_24.jsonl with two extra fields per record:
  - iou_24:  max IoU over 24 cube rotations of pred_mesh (None on failure)
  - rot_idx: winning rotation idx (0 = identity, -1 = none worked)

Original `iou` and `cd` are preserved; iou_24 ≥ iou by construction (rot 0
is the identity, so the search includes the original orientation).

Only retries records with error_type ∈ {success, zero_iou} — exec failures
cannot benefit from rotating a pred mesh that was never produced.

Usage:
    python scripts/analysis/rescore_iou_24.py \\
        --root eval_outputs/cad_bench_722 \\
        --hf-repo BenchCAD/cad_bench_722 --split train \\
        --workers 6 --early-stop 0.95
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from common.metrics import compute_metrics_24


# Same GT-execution snippet eval/bench.py uses — produces normalized STL.
_GT_TMPL = textwrap.dedent('''\
    import sys, io
    import cadquery as cq
    import trimesh, numpy as np
    show_object = lambda *a, **kw: None

    {code}

    _r = locals().get('result') or locals().get('r')
    if _r is None:
        raise ValueError('no result/r variable')
    compound = _r.val()
    verts, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x,v.y,v.z) for v in verts], faces)
    buf = trimesh.exchange.stl.export_stl(mesh)
    mesh2 = trimesh.load(io.BytesIO(buf), file_type='stl', force='mesh')
    mesh2.apply_translation(-(mesh2.bounds[0]+mesh2.bounds[1])/2.0)
    ext = float(np.max(mesh2.extents))
    if ext > 1e-7:
        mesh2.apply_scale(2.0/ext)
    mesh2.export(sys.argv[1])
''')

_LD = os.environ.get('LD_LIBRARY_PATH', '/workspace/.local/lib')


def _exec_gt(gt_code: str, timeout: float = 60.0):
    script = _GT_TMPL.format(code=gt_code)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl = f.name
    env = {**os.environ, 'LD_LIBRARY_PATH': _LD}
    try:
        r = subprocess.run([sys.executable, '-c', script, stl],
                           capture_output=True, timeout=timeout, env=env)
        if r.returncode == 0 and Path(stl).stat().st_size > 100:
            return stl
        return None
    except Exception:
        return None


def _rescore_one(rec: dict, pred_dir: Path, gt_code: str,
                 early_stop: float, timeout: float) -> dict:
    out = dict(rec)
    if rec.get('error_type') not in ('success', 'zero_iou'):
        out['iou_24'] = None
        out['rot_idx'] = -1
        return out
    stem = rec['stem']
    pred_py = pred_dir / f'{stem}.py'
    if not pred_py.exists():
        out['iou_24'] = None
        out['rot_idx'] = -1
        out['rescore_error'] = 'pred_py_missing'
        return out
    pred_code = pred_py.read_text()
    gt_stl = _exec_gt(gt_code)
    if gt_stl is None:
        out['iou_24'] = None
        out['rot_idx'] = -1
        out['rescore_error'] = 'gt_exec_fail'
        return out
    try:
        iou_naive, cd_new, iou_24, rot_idx = compute_metrics_24(
            pred_code, gt_stl, timeout=timeout, iou_24_early_stop=early_stop)
    finally:
        Path(gt_stl).unlink(missing_ok=True)
    out['iou_24']  = None if iou_24 is None else round(iou_24, 4)
    out['rot_idx'] = rot_idx
    # Also re-record iou_naive for sanity (it may differ slightly from the
    # original because exec is non-deterministic for some cadquery shapes).
    out['iou_recheck'] = None if iou_naive < 0 else round(iou_naive, 4)
    return out


def _summarize_24(records: list[dict]) -> dict:
    n = len(records)
    ok    = [r for r in records if r.get('iou_24') is not None]
    iou1  = [r['iou']    for r in records if r.get('iou')    is not None]
    iou24 = [r['iou_24'] for r in ok]
    # delta on samples scored under both
    paired = [(r['iou'], r['iou_24']) for r in records
              if r.get('iou') is not None and r.get('iou_24') is not None]
    delta = ([b - a for a, b in paired])
    return {
        'n_total':        n,
        'n_iou_24_ok':    len(ok),
        'mean_iou':       round(sum(iou1)/len(iou1), 4)   if iou1  else None,
        'mean_iou_24':    round(sum(iou24)/len(iou24), 4) if iou24 else None,
        'mean_delta':     round(sum(delta)/len(delta), 4) if delta else None,
        'n_improved':     sum(1 for d in delta if d > 1e-4),
        'n_unchanged':    sum(1 for d in delta if abs(d) <= 1e-4),
        'n_rotated_win':  sum(1 for r in ok if (r.get('rot_idx') or 0) > 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='cad_bench_722 eval root')
    ap.add_argument('--hf-repo', default='BenchCAD/cad_bench_722')
    ap.add_argument('--split',   default='train')
    ap.add_argument('--workers', type=int, default=6,
                    help='Parallel rescore subprocesses (each spawns 1 cadquery worker)')
    ap.add_argument('--early-stop', type=float, default=0.95,
                    help='Stop searching rotations once IoU ≥ this (default 0.95)')
    ap.add_argument('--per-sample-timeout', type=float, default=300.0)
    ap.add_argument('--limit', type=int, default=0,
                    help='Per-model cap on records to rescore (0=all). For smoke testing.')
    args = ap.parse_args()

    # Load gt_code per stem once
    print(f'Loading {args.hf_repo} ...', flush=True)
    from datasets import load_dataset
    token = os.environ.get('HF_TOKEN')
    ds = load_dataset(args.hf_repo, token=token)
    gt_by_stem = {row['stem']: row['gt_code'] for row in ds[args.split]}
    print(f'  {len(gt_by_stem)} GT codes loaded.', flush=True)

    root = Path(args.root)
    model_dirs = sorted(p for p in root.iterdir()
                        if p.is_dir() and not p.name.startswith('_')
                        and (p / 'metadata.jsonl').exists())
    print(f'Models to rescore: {[d.name for d in model_dirs]}', flush=True)

    summary_24 = {}
    for mdir in model_dirs:
        print(f'\n=== {mdir.name} ===', flush=True)
        meta_in  = mdir / 'metadata.jsonl'
        meta_out = mdir / 'metadata_24.jsonl'

        # Resume support: skip stems already in metadata_24.jsonl
        done = set()
        if meta_out.exists():
            for line in open(meta_out):
                try: done.add(json.loads(line)['stem'])
                except Exception: pass
            print(f'  resume: {len(done)} already rescored', flush=True)

        records = [json.loads(l) for l in open(meta_in)]
        todo = [r for r in records if r['stem'] not in done]
        if args.limit:
            todo = todo[:args.limit]
        print(f'  rescoring {len(todo)} / {len(records)} records (workers={args.workers})', flush=True)

        out_f = open(meta_out, 'a')
        t0 = time.time()
        n_done_iter = 0

        def _task(rec):
            gt_code = gt_by_stem.get(rec['stem'])
            if gt_code is None:
                return {**rec, 'iou_24': None, 'rot_idx': -1,
                        'rescore_error': 'gt_code_missing'}
            return _rescore_one(rec, mdir, gt_code,
                                early_stop=args.early_stop,
                                timeout=args.per_sample_timeout)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_task, r) for r in todo]
            for fut in as_completed(futs):
                rec = fut.result()
                out_f.write(json.dumps(rec) + '\n'); out_f.flush()
                n_done_iter += 1
                if n_done_iter % 20 == 0:
                    rate = n_done_iter / (time.time() - t0 + 1e-6)
                    eta_min = (len(todo) - n_done_iter) / max(rate, 1e-6) / 60
                    print(f'  [{n_done_iter}/{len(todo)}] {rate:.2f}/s  ETA {eta_min:.1f}min', flush=True)
        out_f.close()

        # Re-read full metadata_24.jsonl for summary (includes resumed lines)
        all_24 = [json.loads(l) for l in open(meta_out)]
        s = _summarize_24(all_24)
        summary_24[mdir.name] = s
        print(f'  → {s}', flush=True)

    out_path = root / 'summary_iou_24.json'
    out_path.write_text(json.dumps({'dataset': args.hf_repo,
                                    'early_stop': args.early_stop,
                                    'models': summary_24}, indent=2))
    print(f'\nWrote {out_path}', flush=True)
    print(json.dumps(summary_24, indent=2))


if __name__ == '__main__':
    main()
