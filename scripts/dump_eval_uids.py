"""Dump every eval bucket's exact uid list to a frozen JSON.

For paper provenance + cross-config sanity: every v2 training run scores on
EXACT same uids (eval seed hardcoded to 42 in online_eval). This script
re-derives the uid lists offline so you can cite them, diff them, or
verify two HPCs landed on the same rows.

Output:
  data/_eval_uids/v2_eval_uids.json
    {
      "BenchCAD val IID":    ["uid1", "uid2", ...],   # 50 stratified
      "BenchCAD val OOD":    [...],                    # 50 stratified
      "bench-simple OOD":    [...],                    # 50 stratified
      "DeepCAD test":        [...],                    # 50 random (seed=42)
      "Fusion360 test":      [...],                    # 50 random (seed=42)
    }

Usage:
    uv run python scripts/dump_eval_uids.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.sft.online_eval import (  # noqa: E402
    set_holdout_families, set_holdout_families_v2,
    _load_benchcad_val, _load_iso_val, _load_bench_simple_val,
    _load_stl_dir_test,
)


def main() -> None:
    import yaml

    # Activate the v2 holdout lists so the eval loaders emit the right buckets
    holdout_v1 = yaml.safe_load(
        (REPO / 'configs/sft/holdout_families.yaml').read_text())['holdout_families']
    holdout_v2 = yaml.safe_load(
        (REPO / 'configs/sft/holdout_families_v2.yaml').read_text())['holdout_families_v2']

    set_holdout_families(holdout_v1)
    set_holdout_families_v2(holdout_v2)

    # Each eval bucket — uid list (hardcoded seed=42 in loaders)
    out: dict[str, list[str]] = {}

    print('Loading BC val (IID + OOD) ...')
    for it in _load_benchcad_val(n=50, seed=42):
        out.setdefault(it['_dataset_label'], []).append(it['file_name'])

    print('Loading iso val (IID + OOD) ...')
    for it in _load_iso_val(n=50, seed=42):
        out.setdefault(it['_dataset_label'], []).append(it['file_name'])

    # bench-simple kept available but NOT in default eval subsets — included
    # here for completeness only (paper docs may cite it as deprecated).
    print('Loading bench-simple val (IID + OOD, deprecated) ...')
    for it in _load_bench_simple_val(n=20, seed=42):
        out.setdefault(it['_dataset_label'], []).append(it['file_name'])

    print('Loading DeepCAD + Fusion360 test ...')
    for label, root in [('DeepCAD test',   'data/deepcad_test_mesh'),
                        ('Fusion360 test', 'data/fusion360_test_mesh')]:
        items = _load_stl_dir_test(root, label, 50, 42)
        out[label] = [it['file_name'] for it in items]

    out_path = REPO / 'data/_eval_uids/v2_eval_uids.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f'\nDumped {sum(len(v) for v in out.values())} uids across '
          f'{len(out)} buckets → {out_path}')
    for k, v in sorted(out.items()):
        print(f'  {k:<24}  n={len(v)}  first 3: {v[:3]}')

    # Sanity: verify NO eval uid is in any training pkl
    print('\nSanity check — eval uids vs training pkls:')
    train_pkls = [
        ('benchcad/train.pkl',                     'data/benchcad/train.pkl'),
        ('benchcad/train_v4_holdout.pkl',          'data/benchcad/train_v4_holdout.pkl'),
        ('cad-iso-106/train.pkl',                  'data/cad-iso-106/train.pkl'),
        ('cad-iso-106/train_v4_holdout.pkl',       'data/cad-iso-106/train_v4_holdout.pkl'),
        ('benchcad-simple/train.pkl',              'data/benchcad-simple/train.pkl'),
        ('benchcad-simple/train_v2_holdout.pkl',   'data/benchcad-simple/train_v2_holdout.pkl'),
        ('benchcad-easy/train.pkl',                'data/benchcad-easy/train.pkl'),
    ]
    import pickle
    eval_uids = {u for v in out.values() for u in v}
    for label, p in train_pkls:
        try:
            rows = pickle.load(open(REPO / p, 'rb'))
        except FileNotFoundError:
            print(f'  {label:<42}  pkl missing — skip')
            continue
        train_uids = {r.get('uid') for r in rows if r.get('uid')}
        overlap = eval_uids & train_uids
        if overlap:
            print(f'  {label:<42}  ⚠ OVERLAP {len(overlap)} uids: {sorted(overlap)[:3]}')
        else:
            print(f'  {label:<42}  ✓ disjoint ({len(train_uids):,} train uids)')


if __name__ == '__main__':
    main()
