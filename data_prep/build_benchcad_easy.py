"""Assemble data/benchcad-easy/train.pkl + per-item .py/.png files from
local Hula0401 shards.

Reads:
  /ephemeral/data/_hula0401_shards/train-NNNNN-of-00055.parquet  (shards 15–54)
    Columns: stem, code, render_img.bytes, family, difficulty, n_ops, ops_json, ...

Writes:
  data/benchcad-easy/train/<stem>.py
  data/benchcad-easy/train/<stem>_render.png
  data/benchcad-easy/train.pkl    # list of {uid, py_path, png_path, family}

Skips shards 0–14 — never rendered locally, would need a multi-hour render
job. The 80k items from shards 15–54 cover all 11 simple_* families.

Usage:
    uv run python -m data_prep.build_benchcad_easy
    [--shards-dir /ephemeral/data/_hula0401_shards]
    [--out-dir data/benchcad-easy]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-dir', type=Path,
                    default=Path('/ephemeral/data/_hula0401_shards'))
    ap.add_argument('--out-dir', type=Path,
                    default=REPO_ROOT / 'data' / 'benchcad-easy')
    ap.add_argument('--max-shards', type=int, default=None,
                    help='cap for smoke testing')
    ap.add_argument('--shard-range', type=str, default='15-54',
                    help='Inclusive shard index range to ingest, e.g. "15-54". '
                         'Pass "all" to take every shard (not recommended — '
                         'shards 0–14 were not rendered locally and are missing PNG bytes).')
    args = ap.parse_args()

    train_dir = args.out_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    all_shards = sorted(args.shards_dir.glob('train-*.parquet'))
    if args.shard_range == 'all':
        shards = list(all_shards)
    else:
        try:
            lo, hi = (int(x) for x in args.shard_range.split('-', 1))
        except ValueError:
            ap.error(f'--shard-range must be "lo-hi" or "all", got {args.shard_range!r}')
        # Match shard index from filename: train-NNNNN-of-00055.parquet
        import re
        rx = re.compile(r'train-(\d+)-of-\d+\.parquet$')
        shards = []
        for p in all_shards:
            m = rx.search(p.name)
            if m and lo <= int(m.group(1)) <= hi:
                shards.append(p)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print(f'[benchcad_easy] {len(shards)} shards (range={args.shard_range}) '
          f'from {args.shards_dir}', flush=True)
    if not shards:
        ap.error(f'no shards matched in {args.shards_dir} for range {args.shard_range!r}')

    rows: list[dict] = []
    n_skipped = 0
    for i, shard in enumerate(shards):
        t = pq.read_table(shard, columns=['stem', 'code', 'render_img', 'family'])
        df = t.to_pandas()
        for _, r in df.iterrows():
            stem = r['stem']
            code = r['code'] or ''
            png_blob = r['render_img']
            png_bytes = png_blob.get('bytes') if isinstance(png_blob, dict) else None
            if not code or not png_bytes:
                n_skipped += 1
                continue

            py_rel = f'train/{stem}.py'
            png_rel = f'train/{stem}_render.png'
            (args.out_dir / py_rel).write_text(code)
            (args.out_dir / png_rel).write_bytes(png_bytes)
            rows.append({
                'uid': stem,
                'py_path': py_rel,
                'png_path': png_rel,
                'family': r['family'],
            })

        if (i + 1) % 5 == 0 or i == len(shards) - 1:
            print(f'  shard {i + 1}/{len(shards)} → {len(rows)} ok, '
                  f'{n_skipped} skipped', flush=True)

    pkl_path = args.out_dir / 'train.pkl'
    pickle.dump(rows, pkl_path.open('wb'))
    print(f'[benchcad_easy] wrote {pkl_path}: {len(rows)} items '
          f'(skipped {n_skipped})', flush=True)

    # Per-family count
    from collections import Counter
    fams = Counter(r['family'] for r in rows)
    print('per-family:')
    for k, v in fams.most_common():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
