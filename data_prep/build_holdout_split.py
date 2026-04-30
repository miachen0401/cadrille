"""Partition benchcad + cad_iso_106 into train (no holdout) + holdout-test pkl.

Reads:
  data/benchcad/{train,val}.pkl
  data/cad-iso-106/train.pkl   (no family field → derived from stem)

Writes:
  data/_holdout/holdout_train_drop.pkl    # items REMOVED from training (4 sources combined)
  data/_holdout/holdout_test.pkl          # eval set: per-family stratified sample
  data/benchcad/train_v4_holdout.pkl      # benchcad train minus holdout families
  data/cad-iso-106/train_v4_holdout.pkl   # iso train minus holdout families

Constraints:
  - Holdout families must have all ops covered by REST of corpus (verified externally)
  - val.pkl is NOT touched (it's the eval set; we just label some items as IID/OOD)

Usage:
    uv run python -m data_prep.build_holdout_split \\
        --holdout-families tapered_boss taper_pin venturi_tube bucket \\
                           dome_cap nozzle enclosure waffle_plate bolt duct_elbow
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _extract_family(uid: str) -> str | None:
    """Recover family name from stem patterns used by benchcad / cad_iso_106."""
    if not uid:
        return None
    s = uid
    if s.startswith('dvsub_'):
        s = s[len('dvsub_'):]
    if s.startswith('synth_'):
        s = s[len('synth_'):]
    m = re.match(r'(?P<fam>[a-z][a-z0-9_]+?)_\d{3,}(_s\w+)?$', s)
    return m.group('fam') if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout-families', nargs='+', required=True,
                    help='List of family names to hold out from training')
    ap.add_argument('--out-dir', type=Path,
                    default=REPO_ROOT / 'data' / '_holdout')
    args = ap.parse_args()

    holdout = set(args.holdout_families)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[holdout] families ({len(holdout)}): {sorted(holdout)}', flush=True)

    # ── 1. Partition benchcad train ────────────────────────────────────────
    bc_train = pickle.load(open('data/benchcad/train.pkl', 'rb'))
    bc_keep, bc_holdout = [], []
    for r in bc_train:
        if r.get('family') in holdout:
            bc_holdout.append(r)
        else:
            bc_keep.append(r)
    print(f'[benchcad] train: {len(bc_train)} → keep {len(bc_keep)} + holdout {len(bc_holdout)}',
          flush=True)

    out_p = REPO_ROOT / 'data/benchcad/train_v4_holdout.pkl'
    pickle.dump(bc_keep, out_p.open('wb'))
    print(f'[benchcad] wrote {out_p}', flush=True)

    # ── 2. Partition cad_iso_106 train (family from stem) ──────────────────
    iso_train = pickle.load(open('data/cad-iso-106/train.pkl', 'rb'))
    iso_keep, iso_holdout = [], []
    for r in iso_train:
        fam = _extract_family(r.get('uid', ''))
        if fam in holdout:
            iso_holdout.append(r)
        else:
            iso_keep.append(r)
    print(f'[cad_iso_106] train: {len(iso_train)} → keep {len(iso_keep)} + holdout {len(iso_holdout)}',
          flush=True)

    out_p = REPO_ROOT / 'data/cad-iso-106/train_v4_holdout.pkl'
    pickle.dump(iso_keep, out_p.open('wb'))
    print(f'[cad_iso_106] wrote {out_p}', flush=True)

    # ── 3. benchcad VAL — split into IID + OOD  ────────────────────────────
    # val.pkl untouched on disk; we just count what would be IID vs OOD.
    bc_val = pickle.load(open('data/benchcad/val.pkl', 'rb'))
    val_iid = [r for r in bc_val if r.get('family') not in holdout]
    val_ood = [r for r in bc_val if r.get('family') in holdout]
    print(f'[benchcad val] IID={len(val_iid)}  OOD={len(val_ood)}  (total {len(bc_val)})',
          flush=True)
    by_fam_ood = Counter(r['family'] for r in val_ood)
    print(f'[benchcad val OOD] per-family counts: {dict(by_fam_ood)}', flush=True)

    # ── 4. holdout_test.pkl — stratified eval set (all OOD train items) ────
    # Combines OOD-train (removed from SFT) + OOD-val items so we can do
    # full-distribution evaluation on each holdout family.
    all_holdout = bc_holdout + iso_holdout + val_ood
    out_p = args.out_dir / 'holdout_test.pkl'
    pickle.dump(all_holdout, out_p.open('wb'))
    print(f'[holdout_test] wrote {out_p}: {len(all_holdout)} items', flush=True)

    # Summary table per family
    fam_counts: dict[str, dict[str, int]] = defaultdict(lambda: dict(bc_train=0, iso_train=0, bc_val=0))
    for r in bc_holdout: fam_counts[r['family']]['bc_train'] += 1
    for r in iso_holdout:
        fam = _extract_family(r.get('uid',''))
        if fam: fam_counts[fam]['iso_train'] += 1
    for r in val_ood: fam_counts[r['family']]['bc_val'] += 1

    print(f'\n=== Per-family holdout summary ===')
    print(f'{"family":<25} {"bc_train":>10} {"iso_train":>11} {"bc_val":>8} {"total":>8}')
    print('-' * 70)
    grand_total = 0
    for fam in sorted(holdout):
        c = fam_counts[fam]
        tot = c['bc_train'] + c['iso_train'] + c['bc_val']
        grand_total += tot
        print(f'{fam:<25} {c["bc_train"]:>10} {c["iso_train"]:>11} {c["bc_val"]:>8} {tot:>8}')
    print(f'{"TOTAL":<25} {sum(c["bc_train"] for c in fam_counts.values()):>10} '
          f'{sum(c["iso_train"] for c in fam_counts.values()):>11} '
          f'{sum(c["bc_val"] for c in fam_counts.values()):>8} '
          f'{grand_total:>8}')


if __name__ == '__main__':
    main()
