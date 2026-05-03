"""§7 v2 holdout split — bench-simple op patterns held out from training.

For each source that contains the v2 OOD pattern families, write a
`train_v2_holdout.pkl` with those rows removed. Reuses the same shape as
`build_holdout_split.py` but operates on a different family list and writes
files alongside the existing v4 holdout pkls.

Sources affected:
  data/benchcad-simple/train.pkl  → train_v2_holdout.pkl

Other sources (benchcad, cad-iso-106, recode_bench, text2cad-bench,
benchcad-easy) do NOT contain bench-simple op-pattern families, so no
filtering is needed for v2 — they continue to use whatever pkl their config
points at (e.g. train_v4_holdout.pkl for the 10-mech holdout).

Usage:
    uv run python -m data_prep.build_holdout_v2
"""
from __future__ import annotations
import pickle
import sys
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from train.rl.dataset import _extract_family  # noqa: E402


def main() -> None:
    holdout = set(yaml.safe_load(
        (REPO / 'configs/sft/holdout_families_v2.yaml').read_text()
    )['holdout_families_v2'])
    print(f'[holdout_v2] {len(holdout)} families: {sorted(holdout)}')

    # bench-simple split
    src = REPO / 'data/benchcad-simple/train.pkl'
    rows = pickle.load(open(src, 'rb'))
    keep = [r for r in rows
            if (r.get('family') or _extract_family(r.get('uid', ''))) not in holdout]
    drop_n = len(rows) - len(keep)
    out = REPO / 'data/benchcad-simple/train_v2_holdout.pkl'
    pickle.dump(keep, out.open('wb'))
    print(f'[bench-simple] {len(rows)} → {len(keep)} kept, {drop_n} held out')
    print(f'  wrote {out}')

    # OOD test pool (val.pkl rows in holdout families) — sanity check
    val = pickle.load(open(REPO / 'data/benchcad-simple/val.pkl', 'rb'))
    ood_val = [r for r in val
               if _extract_family(r.get('uid', '')) in holdout]
    iid_val = [r for r in val
               if _extract_family(r.get('uid', '')) and
               _extract_family(r.get('uid', '')) not in holdout]
    print(f'[bench-simple val] OOD pool: {len(ood_val)} rows  '
          f'(IID pool: {len(iid_val)} rows for sanity)')

    # Per-family count summary
    from collections import Counter
    fam_counts = Counter(_extract_family(r.get('uid', '')) for r in ood_val)
    print(f'[bench-simple OOD val] per-family counts:')
    for fam in sorted(holdout):
        print(f'    {fam:35s}  {fam_counts.get(fam, 0):>4}')


if __name__ == '__main__':
    main()
