"""Audit geometric/code complexity per training source.

For each source, compute on a sample (1000 items by default):
  - n_lines per code (median, p90)
  - n_unique_ops per code (median)
  - mesh face count (median, p90)  — requires render OR existing STL
  - bbox volume / surface ratio

Output: experiments_log/dataset_complexity_audit.csv  + a summary plot.

Usage:
    uv run python -m scripts.analysis.dataset_complexity_audit
        # defaults: 1000 samples per source, seed=42
"""
from __future__ import annotations

import argparse
import csv
import pickle
import random
import re
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = REPO_ROOT / 'data'

CADQUERY_OP_RE = re.compile(
    r'\.(box|circle|rect|polygon|polyline|ellipse|arc|line|sketch|workplane|'
    r'extrude|cut|union|intersect|loft|sweep|revolve|fillet|chamfer|shell|'
    r'mirror|rotate|translate|transformed|edges|faces|vertices|tag|sphere|'
    r'cylinder|wedge|center|moveTo|lineTo|threePointArc|spline|close)\b'
)


def load_code(root: Path, item: dict) -> str | None:
    if item.get('code'):
        return item['code']
    if 'py_path' in item:
        p = root / item['py_path']
        if p.exists(): return p.read_text()
    # legacy text2cad: cadquery/{uid}.py
    p = root / 'cadquery' / f"{item['uid']}.py"
    if p.exists(): return p.read_text()
    return None


def code_stats(code: str) -> tuple[int, int]:
    n_lines = sum(1 for l in code.split('\n') if l.strip())
    ops = set(CADQUERY_OP_RE.findall(code))
    return n_lines, len(ops)


def stl_stats(root: Path, item: dict) -> tuple[int | None, float | None]:
    """Return (face_count, log_volume) from the GT STL on disk if present."""
    stl_path = None
    if 'mesh_path' in item:
        stl_path = root / item['mesh_path']
    elif 'png_path' in item:
        # bench-shell layout: alongside the render
        stem = Path(item['png_path']).with_suffix('').name.replace('_render', '')
        candidate = root / Path(item['png_path']).parent.name / f'{stem}.stl'
        if candidate.exists():
            stl_path = candidate
    if not stl_path or not stl_path.exists():
        return None, None
    try:
        import trimesh
        m = trimesh.load(str(stl_path), force='mesh')
        if m.is_empty: return 0, None
        return len(m.faces), float(np.log10(max(m.volume, 1e-9)))
    except Exception:
        return None, None


def percentile(xs: list[float], p: float) -> float | str:
    if not xs: return 'NA'
    return float(np.percentile(xs, p))


def audit_source(name: str, root: Path, n_sample: int, seed: int) -> dict:
    pkl = root / 'train.pkl'
    if not pkl.exists():
        return {'source': name, 'error': f'{pkl} missing'}
    rows = pickle.load(open(pkl, 'rb'))
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_sample, len(rows)))

    n_lines_list, n_ops_list, faces_list, vol_list = [], [], [], []
    n_no_code = 0; n_no_stl = 0
    for item in sample:
        code = load_code(root, item)
        if code:
            l, o = code_stats(code)
            n_lines_list.append(l); n_ops_list.append(o)
        else:
            n_no_code += 1
        faces, vol = stl_stats(root, item)
        if faces is not None:
            faces_list.append(faces)
            if vol is not None: vol_list.append(vol)
        else:
            n_no_stl += 1

    return {
        'source': name,
        'n_total': len(rows),
        'n_sampled': len(sample),
        'n_no_code': n_no_code,
        'n_no_stl': n_no_stl,
        'lines_p50': statistics.median(n_lines_list) if n_lines_list else 'NA',
        'lines_p90': percentile(n_lines_list, 90),
        'ops_p50':   statistics.median(n_ops_list) if n_ops_list else 'NA',
        'ops_p90':   percentile(n_ops_list, 90),
        'faces_p50': statistics.median(faces_list) if faces_list else 'NA',
        'faces_p90': percentile(faces_list, 90),
        'log_vol_p50': statistics.median(vol_list) if vol_list else 'NA',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-sample', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=str(REPO_ROOT / 'experiments_log' /
                                          'dataset_complexity_audit.csv'))
    args = ap.parse_args()

    sources = [
        ('benchcad',         DATA_ROOT / 'benchcad'),
        ('cad_iso_106',      DATA_ROOT / 'cad-iso-106'),
        ('benchcad_simple',  DATA_ROOT / 'benchcad-simple'),
        ('text2cad_bench',   DATA_ROOT / 'text2cad-bench'),
        ('text2cad_legacy',  DATA_ROOT / 'text2cad'),
        ('recode_bench',     DATA_ROOT / 'cad-recode-bench'),
        ('cad_recode_20k',   DATA_ROOT / 'cad-recode-20k'),
    ]

    rows = []
    for name, root in sources:
        print(f'[{name}] auditing ...', flush=True)
        r = audit_source(name, root, args.n_sample, args.seed)
        print(f'  {r}', flush=True)
        rows.append(r)

    fields = ['source', 'n_total', 'n_sampled', 'n_no_code', 'n_no_stl',
              'lines_p50', 'lines_p90', 'ops_p50', 'ops_p90',
              'faces_p50', 'faces_p90', 'log_vol_p50']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\nwrote → {args.out}', flush=True)


if __name__ == '__main__':
    main()
