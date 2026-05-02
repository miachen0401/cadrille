"""Assemble the RL training candidate pool.

Walks the BenchCAD train_v4_holdout pkl + (optional) iso pkl + (optional) DC/Fu
mesh dirs, and writes a unified pkl that mining (`train.rl.mine`) consumes.

Each row: {gt_mesh_path, file_name, family} — `family` may be None for rows
whose stem doesn't match the BenchCAD/iso family pattern (those auto-fall to
pure-IoU reward inside the worker).

Output: data/_rl/candidates_<tag>.pkl  (tag derived from --include flags)

Usage:
    uv run python scripts/build_rl_candidate_pool.py \
        --include bench,deepcad,fusion360 \
        --out data/_rl/candidates_bench_dc_fu.pkl

Notes:
  * cad-iso-106 has no local STLs — `--include iso` is ignored unless
    --iso-stl-root is provided (point at a dir containing pre-generated STLs).
  * cad-recode-bench / text2cad-bench have neither family nor STLs and are
    intentionally skipped.
"""
from __future__ import annotations
import argparse
import os
import pickle
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.rl.dataset import _extract_family  # reuse the family-from-stem regex


def _bench_rows() -> list[dict]:
    pkl = REPO / 'data/benchcad/train_v4_holdout.pkl'
    if not pkl.exists():
        print(f'[skip] {pkl} not found')
        return []
    rows = pickle.load(open(pkl, 'rb'))
    out = []
    bench_root = REPO / 'data/benchcad'
    for r in rows:
        stl = bench_root / r['mesh_path']
        if not stl.exists():
            continue
        out.append({
            'gt_mesh_path': str(stl),
            'file_name':    r['uid'],
            'family':       r.get('family'),
        })
    print(f'[bench] {len(out)} rows from {pkl.name} (STLs verified)')
    return out


def _iso_rows(stl_root: Path) -> list[dict]:
    pkl = REPO / 'data/cad-iso-106/train_v4_holdout.pkl'
    if not pkl.exists() or not stl_root.is_dir():
        print(f'[skip] iso (pkl or stl_root missing: {pkl} / {stl_root})')
        return []
    rows = pickle.load(open(pkl, 'rb'))
    out = []
    skipped = 0
    for r in rows:
        stl = stl_root / f'{r["uid"]}.stl'
        if not stl.exists():
            skipped += 1
            continue
        out.append({
            'gt_mesh_path': str(stl),
            'file_name':    r['uid'],
            'family':       _extract_family(r['uid']),
        })
    print(f'[iso] {len(out)} rows ({skipped} skipped, no STL)')
    return out


def _mesh_dir_rows(label: str, mesh_dir: Path, limit: int | None = None) -> list[dict]:
    if not mesh_dir.is_dir():
        print(f'[skip] {label}: {mesh_dir} not found')
        return []
    out = []
    for stl in sorted(mesh_dir.glob('*.stl')):
        out.append({
            'gt_mesh_path': str(stl),
            'file_name':    stl.stem,
            'family':       None,   # external benchmark — no family / no ess spec
        })
        if limit and len(out) >= limit:
            break
    print(f'[{label}] {len(out)} rows from {mesh_dir}')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--include', default='bench',
                    help='Comma-separated subset of {bench, iso, deepcad, fusion360}.')
    ap.add_argument('--iso-stl-root', type=Path, default=REPO / 'data/cad-iso-106/train',
                    help='Directory containing pre-generated iso STLs (if --include iso).')
    ap.add_argument('--dc-limit', type=int, default=2000,
                    help='Max DeepCAD rows to include (full set is 8k).')
    ap.add_argument('--fu-limit', type=int, default=1500,
                    help='Max Fusion360 rows to include (full set is 1.7k).')
    ap.add_argument('--out',    type=Path, required=True)
    args = ap.parse_args()

    sources = [s.strip() for s in args.include.split(',') if s.strip()]
    rows: list[dict] = []
    if 'bench' in sources:
        rows.extend(_bench_rows())
    if 'iso' in sources:
        rows.extend(_iso_rows(args.iso_stl_root))
    if 'deepcad' in sources:
        rows.extend(_mesh_dir_rows('deepcad', REPO / 'data/deepcad_test_mesh',
                                   limit=args.dc_limit))
    if 'fusion360' in sources:
        rows.extend(_mesh_dir_rows('fusion360', REPO / 'data/fusion360_test_mesh',
                                   limit=args.fu_limit))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(rows, f)

    n_with_fam = sum(1 for r in rows if r['family'])
    fam_count: dict[str, int] = {}
    for r in rows:
        if r['family']:
            fam_count[r['family']] = fam_count.get(r['family'], 0) + 1
    print(f'\n=== Wrote {args.out} ===')
    print(f'  total: {len(rows)}')
    print(f'  with family (eligible for ess reward): {n_with_fam} '
          f'({100*n_with_fam/max(len(rows),1):.0f}%)')
    print(f'  unique families: {len(fam_count)}')
    print(f'  top 5 by count: {sorted(fam_count.items(), key=lambda x: -x[1])[:5]}')


if __name__ == '__main__':
    main()
