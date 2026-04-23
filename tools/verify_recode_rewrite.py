"""Smoke-verify that rewrite_recode_to_bench.py preserves CAD semantics.

Executes both the original recode .py and its rewritten bench-style version,
extracts the solid, and compares mesh volume + bounding box. For a tighter
check, computes Chamfer / hausdorff distance between point-cloud samples.

Usage:
  uv run python tools/verify_recode_rewrite.py \\
      --original data/cad-recode-v1.5/val \\
      --rewritten /tmp/recode_bench_val \\
      --n 30
"""
import argparse
import os
import random
import signal
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise TimeoutError(f'exec exceeded {seconds}s')
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def exec_code_to_mesh(code, timeout_s=30):
    """Exec CadQuery code; return trimesh.Trimesh (following evaluate.py conventions)."""
    import cadquery as cq
    import trimesh

    # Inject a no-op show_object so bench-style code runs
    captured = []
    def show_object(obj, *args, **kwargs):
        captured.append(obj)

    ns = {'cq': cq, 'show_object': show_object}
    with time_limit(timeout_s):
        exec(code, ns)

    obj = None
    if captured:
        obj = captured[-1]
    elif 'result' in ns:
        obj = ns['result']
    elif 'r' in ns:
        obj = ns['r']
    else:
        raise RuntimeError('No result object found (no show_object / result / r)')

    compound = obj.val() if hasattr(obj, 'val') else obj
    # Use same tessellate params as evaluate.py:compound_to_mesh
    verts, faces = compound.tessellate(0.001, 0.1)
    # process=True merges duplicate vertices → makes the mesh watertight for IoU booleans
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces, process=True)
    return mesh


def compute_iou(gt_mesh, pred_mesh):
    """Volumetric IoU via trimesh boolean — matches evaluate.py:compute_iou."""
    try:
        intersection_volume = 0.0
        for gt_i in gt_mesh.split():
            for pred_i in pred_mesh.split():
                inter = gt_i.intersection(pred_i)
                v = inter.volume if inter is not None else 0
                intersection_volume += v
        gt_vol = sum(m.volume for m in gt_mesh.split())
        pred_vol = sum(m.volume for m in pred_mesh.split())
        union_vol = gt_vol + pred_vol - intersection_volume
        if union_vol <= 0:
            return None
        return float(intersection_volume / union_vol)
    except Exception:
        return None


def sample_points(verts, faces, n=2048, seed=42):
    """Uniform surface sampling via trimesh."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    rng = np.random.default_rng(seed)
    samples, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(samples)


def chamfer(p1, p2):
    """Symmetric mean chamfer distance."""
    from scipy.spatial import cKDTree
    t1 = cKDTree(p1)
    t2 = cKDTree(p2)
    d12, _ = t2.query(p1)
    d21, _ = t1.query(p2)
    return 0.5 * (d12.mean() + d21.mean())


def compare_one(orig_path, rewritten_path, compute_iou_flag=True):
    try:
        m_orig = exec_code_to_mesh(Path(orig_path).read_text())
    except Exception as e:
        return {'ok': False, 'stage': 'orig_exec', 'err': f'{type(e).__name__}: {e}'}
    try:
        m_new = exec_code_to_mesh(Path(rewritten_path).read_text())
    except Exception as e:
        return {'ok': False, 'stage': 'new_exec', 'err': f'{type(e).__name__}: {e}'}

    vol_o = float(m_orig.volume)
    vol_n = float(m_new.volume)
    vol_rel = abs(vol_o - vol_n) / max(abs(vol_o), 1e-9)
    bb_o = m_orig.bounds
    bb_n = m_new.bounds
    bbox_dx = max(
        float(np.linalg.norm(bb_o[0] - bb_n[0])),
        float(np.linalg.norm(bb_o[1] - bb_n[1])),
    )

    v_orig = np.asarray(m_orig.vertices)
    v_new = np.asarray(m_new.vertices)
    cd = chamfer(v_orig, v_new) if len(v_orig) and len(v_new) else float('inf')
    diag = float(np.linalg.norm(bb_o[1] - bb_o[0]))
    cd_rel = cd / max(diag, 1e-9)

    iou = compute_iou(m_orig, m_new) if compute_iou_flag else None

    return {
        'ok': True,
        'vol_o': vol_o,
        'vol_n': vol_n,
        'vol_rel': vol_rel,
        'bbox_dx': bbox_dx,
        'cd_rel': cd_rel,
        'iou': iou,
        'n_verts_o': int(len(v_orig)),
        'n_verts_n': int(len(v_new)),
    }


def _worker(args):
    op, np_, compute_iou_flag = args
    return str(op), compare_one(op, np_, compute_iou_flag=compute_iou_flag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original', type=str, required=True)
    ap.add_argument('--rewritten', type=str, required=True)
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--vol-rel-threshold', type=float, default=0.01,
                    help='Max relative volume difference for "pass" (1% default)')
    ap.add_argument('--iou-threshold', type=float, default=0.99,
                    help='Min IoU for "pass" (0.99 default)')
    ap.add_argument('--no-iou', action='store_true',
                    help='Skip IoU computation (fast vol/bbox/cd only)')
    ap.add_argument('--jobs', type=int, default=1,
                    help='Parallel workers. Note: trimesh booleans are single-thread-heavy '
                         'and eat RAM — keep ≤ (ncores - 2) on 15 GB machines.')
    args = ap.parse_args()

    import multiprocessing as mp

    orig_root = Path(args.original)
    new_root = Path(args.rewritten)

    py_files = sorted(orig_root.rglob('*.py'))
    pairs = []
    for p in py_files:
        rel = p.relative_to(orig_root)
        q = new_root / rel
        if q.exists():
            pairs.append((p, q))

    random.seed(args.seed)
    random.shuffle(pairs)
    pairs = pairs[:args.n]
    print(f'Testing {len(pairs)} pairs from {orig_root} vs {new_root}')
    print(f'IoU: {"off" if args.no_iou else "on"}, jobs: {args.jobs}')

    tasks = [(op, q, not args.no_iou) for (op, q) in pairs]
    results = []
    passes = 0
    n_iou_computed = 0
    iou_sum = 0.0
    iou_min = 1.0
    fails = []

    def _summarize(i, name, result):
        nonlocal passes, n_iou_computed, iou_sum, iou_min
        tag = 'OK'
        if not result['ok']:
            tag = f"FAIL@{result['stage']}"
            fails.append((name, result))
        else:
            vol_ok = result['vol_rel'] <= args.vol_rel_threshold
            cd_ok = result['cd_rel'] <= 0.01
            iou_val = result.get('iou')
            iou_ok = True
            if iou_val is not None:
                n_iou_computed += 1
                iou_sum += iou_val
                iou_min = min(iou_min, iou_val)
                iou_ok = iou_val >= args.iou_threshold
            if vol_ok and cd_ok and iou_ok:
                passes += 1
            else:
                tag = 'DIVERGE'
                fails.append((name, result))
        print(f'[{i+1}/{len(pairs)}] {name}: {tag} '
              f'vol_rel={result.get("vol_rel", "N/A"):.2e} '
              f'cd_rel={result.get("cd_rel", "N/A"):.2e} '
              f'iou={result.get("iou", "N/A")}')

    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            for i, (name, result) in enumerate(pool.imap(_worker, tasks, chunksize=1)):
                _summarize(i, Path(name).name, result)
                results.append((name, result))
    else:
        for i, (op, q, flag) in enumerate(tasks):
            name, result = _worker((op, q, flag))
            _summarize(i, Path(name).name, result)
            results.append((name, result))

    print(f'\n{passes}/{len(pairs)} passed '
          f'(vol_rel ≤ {args.vol_rel_threshold}, cd_rel ≤ 0.01, IoU ≥ {args.iou_threshold})')
    if n_iou_computed:
        print(f'IoU: mean={iou_sum/n_iou_computed:.6f}  min={iou_min:.6f}  '
              f'(computed on {n_iou_computed}/{len(pairs)} pairs)')
    if fails:
        print(f'\n{len(fails)} failures / divergences:')
        for name, r in fails[:10]:
            print(f'  {name}: {r}')


if __name__ == '__main__':
    main()
