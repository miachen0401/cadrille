"""Apply per-source filter recommendations from comprehensive_audit.

Reads experiments_log/comprehensive_audit/source_{name}.csv (per-item
DROP/CUT/KEEP labels), produces filtered pkls for v3 training.

Strategy (per user 2026-04-28):
  - DROP rows: 80 % randomly dropped, 20 % kept for diversity coverage
  - CUT rows: kept entirely (mix-weight will down-weight later if needed)
  - KEEP rows: kept entirely
  - Then code-hash DEDUP: keep 1 representative per unique code

Backup the original pkl as `{split}.pkl.unfiltered` before overwriting.

Usage:
    uv run python -m scripts.analysis.apply_data_filter --apply

To restore original:
    cp data/<source>/train.pkl.unfiltered data/<source>/train.pkl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = REPO_ROOT / 'data'
AUDIT_DIR = REPO_ROOT / 'experiments_log' / 'comprehensive_audit'

# text2cad_legacy intentionally excluded — user 2026-04-28: 「直接删了 别留着了 碍眼」
SOURCES = {
    'benchcad':         DATA_ROOT / 'benchcad',
    'cad_iso_106':      DATA_ROOT / 'cad-iso-106',
    'benchcad_simple':  DATA_ROOT / 'benchcad-simple',
    'text2cad_bench':   DATA_ROOT / 'text2cad-bench',
    'recode_bench':     DATA_ROOT / 'cad-recode-bench',
}


def load_uid_recommendations(source: str) -> dict[str, str]:
    """Return {uid: 'DROP'/'CUT'/'KEEP'}."""
    csv_path = AUDIT_DIR / f'source_{source}.csv'
    if not csv_path.exists():
        print(f'  [warn] no audit CSV at {csv_path}')
        return {}
    out = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            out[r['uid']] = r['recommendation']
    return out


def code_hash_for_item(root: Path, item: dict) -> str | None:
    """Compute a short hash of the item's code (for dedup)."""
    code = item.get('code')
    if not code and 'py_path' in item:
        p = root / item['py_path']
        if p.exists():
            try: code = p.read_text()
            except Exception: pass
    if not code:
        # legacy text2cad: cadquery/{uid}.py
        p = root / 'cadquery' / f"{item['uid']}.py"
        if p.exists():
            try: code = p.read_text()
            except Exception: pass
    if not code: return None
    return hashlib.md5(code.encode()).hexdigest()[:16]


def filter_pkl(root: Path, pkl_path: Path, recs: dict[str, str],
               drop_prob: float, dedup: bool, seed: int,
               apply: bool) -> tuple[int, int, dict]:
    """Filter pkl: 1) random `drop_prob` of DROP-labeled items, 2) dedup by code hash.
    Returns (n_in, n_out, breakdown_dict)."""
    if not pkl_path.exists():
        return 0, 0, {}
    rows = pickle.load(open(pkl_path, 'rb'))
    n_in = len(rows)
    rng = random.Random(seed)

    # Step 1: DROP filter (probabilistic)
    after_drop = []
    n_dropped, n_drop_kept = 0, 0
    for r in rows:
        rec = recs.get(r['uid'], 'KEEP')
        if rec == 'DROP':
            if rng.random() < drop_prob:
                n_dropped += 1
                continue
            else:
                n_drop_kept += 1
        after_drop.append(r)

    # Step 2: dedup by code hash
    n_dedup_removed = 0
    if dedup:
        seen_hashes = set()
        deduped = []
        for r in after_drop:
            h = code_hash_for_item(root, r)
            if h is None:
                deduped.append(r); continue
            if h in seen_hashes:
                n_dedup_removed += 1; continue
            seen_hashes.add(h)
            deduped.append(r)
        kept = deduped
    else:
        kept = after_drop

    n_out = len(kept)
    breakdown = {
        'n_in': n_in, 'n_dropped_random': n_dropped,
        'n_drop_kept_for_diversity': n_drop_kept,
        'n_dedup_removed': n_dedup_removed, 'n_out': n_out,
    }

    if apply:
        backup = pkl_path.with_suffix('.pkl.unfiltered')
        if not backup.exists():
            shutil.copy2(pkl_path, backup)
            print(f'    backup → {backup}')
        with pkl_path.open('wb') as f:
            pickle.dump(kept, f)
    return n_in, n_out, breakdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                    help='Actually write filtered pkls (default: dry-run)')
    ap.add_argument('--drop-prob', type=float, default=0.8,
                    help='Probability of dropping a DROP-labeled item (0.8 = drop 80%, keep 20% for diversity)')
    ap.add_argument('--dedup', action='store_true', default=True,
                    help='Also dedup by code hash (default: True)')
    ap.add_argument('--no-dedup', dest='dedup', action='store_false')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sources', nargs='+', default=list(SOURCES.keys()))
    args = ap.parse_args()

    mode = 'APPLYING' if args.apply else 'DRY RUN'
    print(f'{mode} filter: drop_prob={args.drop_prob}, dedup={args.dedup}, seed={args.seed}\n',
          flush=True)

    total_in, total_out = 0, 0
    for src in args.sources:
        if src not in SOURCES:
            print(f'unknown source: {src}'); continue
        root = SOURCES[src]
        recs = load_uid_recommendations(src)
        n_drop = sum(1 for r in recs.values() if r == 'DROP')
        n_cut = sum(1 for r in recs.values() if r == 'CUT')
        print(f'== {src} (DROP={n_drop}, CUT={n_cut}, recs={len(recs)}) ==')
        for split in ('train', 'val'):
            pkl = root / f'{split}.pkl'
            n_in, n_out, bd = filter_pkl(root, pkl, recs,
                                          args.drop_prob, args.dedup,
                                          args.seed, args.apply)
            if n_in:
                pct = 100 * (n_in - n_out) / n_in
                print(f'  {split}: {n_in} → {n_out}  ({pct:.1f}% removed)')
                print(f'    breakdown: dropped_random={bd["n_dropped_random"]}, '
                      f'drop_kept_for_diversity={bd["n_drop_kept_for_diversity"]}, '
                      f'dedup_removed={bd["n_dedup_removed"]}')
                if split == 'train':
                    total_in += n_in; total_out += n_out

    print(f'\nTOTAL train items: {total_in} → {total_out}  '
          f'({100*(total_in-total_out)/total_in:.1f}% removed)')
    if not args.apply:
        print('\nThis was a dry run. Pass --apply to actually overwrite the pkls.')


if __name__ == '__main__':
    main()
