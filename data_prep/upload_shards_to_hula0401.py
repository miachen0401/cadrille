"""Slice the merged BenchCAD/benchcad-easy parquet into shards (2000 rows each)
and upload to Hula0401/cad-sft/benchcad-easy/train-NNNNN-of-00055.parquet, with
schema converted to match the existing Hula0401 layout.

Schema mapping:
    stem            -> stem
    gt_code         -> code
    composite_png   -> render_img (struct{bytes, path:null})
    family          -> family
    difficulty      -> difficulty
    ops_used (list) -> n_ops (int64) + ops_json (string)
    base_plane      -> base_plane

Usage:
    BenchCAD_HF_TOKEN=... HF_TOKEN=... \\
    uv run python -m data_prep.upload_shards_to_hula0401 \\
        --start-shard 15 --end-shard 55 --shard-size 2000 \\
        --upload
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPO_SRC = 'BenchCAD/benchcad-easy'
PARQUET_IN_REPO = 'data/test-00000-of-00001.parquet'
REPO_DST = 'Hula0401/cad-sft'
SHARD_TOTAL = 55  # naming convention: train-NNNNN-of-00055


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-parquet',
                    default='/ephemeral/data/_hf_upload_benchcad_easy/test-00000-of-00001.parquet',
                    help='Path to merged BenchCAD parquet (already has filled composite_png)')
    ap.add_argument('--start-shard', type=int, required=True)
    ap.add_argument('--end-shard',   type=int, required=True,
                    help='Exclusive end (e.g. 55 means shards [start:55))')
    ap.add_argument('--shard-size', type=int, default=2000)
    ap.add_argument('--out-dir', default='/ephemeral/data/_hula0401_shards')
    ap.add_argument('--upload', action='store_true',
                    help='Upload each shard after writing')
    ap.add_argument('--workers-upload', type=int, default=4,
                    help='Parallel upload workers')
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    # Read source parquet (large, ~1.5 GB)
    print(f'Reading {args.src_parquet} ...', flush=True)
    src = pq.read_table(args.src_parquet)
    n_total = len(src)
    print(f'  rows: {n_total}', flush=True)

    # Pre-extract columns we need
    print('Extracting columns ...', flush=True)
    stems    = src['stem'].to_pylist()
    codes    = src['gt_code'].to_pylist()
    imgs     = src['composite_png'].to_pylist()
    fams     = src['family'].to_pylist()
    diffs    = src['difficulty'].to_pylist()
    bplanes  = src['base_plane'].to_pylist()
    ops_used = src['ops_used'].to_pylist()  # list of strings per row

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Hula0401 schema
    schema = pa.schema([
        ('stem',       pa.string()),
        ('code',       pa.string()),
        ('render_img', pa.struct([('bytes', pa.binary()), ('path', pa.null())])),
        ('family',     pa.string()),
        ('difficulty', pa.string()),
        ('n_ops',      pa.int64()),
        ('ops_json',   pa.string()),
        ('base_plane', pa.string()),
    ])

    # Build all shards first (so any error fails fast before upload)
    shards_to_write: list[tuple[int, Path, int, int]] = []  # (idx, path, n_rows, n_with_png)
    for idx in range(args.start_shard, args.end_shard):
        lo = idx * args.shard_size
        hi = min(lo + args.shard_size, n_total)
        if lo >= n_total:
            break
        rng = range(lo, hi)
        rows = []
        n_with_png = 0
        for i in rng:
            r_img = imgs[i]
            if r_img and r_img.get('bytes'):
                ri = {'bytes': r_img['bytes'], 'path': None}
                n_with_png += 1
            else:
                ri = None
            ops = ops_used[i] or []
            rows.append({
                'stem':       stems[i],
                'code':       codes[i],
                'render_img': ri,
                'family':     fams[i],
                'difficulty': diffs[i],
                'n_ops':      int(len(ops)),
                'ops_json':   json.dumps(list(ops), ensure_ascii=False),
                'base_plane': bplanes[i],
            })
        # Write shard parquet
        shard_name = f'train-{idx:05d}-of-{SHARD_TOTAL:05d}.parquet'
        out_p = out_root / shard_name
        tab = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(tab, out_p, compression='snappy')
        size_mb = out_p.stat().st_size // (1024*1024)
        print(f'  shard {idx:02d}: {out_p.name}  rows={len(rows)}  with_png={n_with_png}'
              f'  ({size_mb} MB)', flush=True)
        shards_to_write.append((idx, out_p, len(rows), n_with_png))

    print(f'\nWrote {len(shards_to_write)} shard files locally', flush=True)
    if not args.upload:
        return

    # Upload each shard. Use BenchCAD_HF_TOKEN if writing to BenchCAD; else HF_TOKEN
    # for Hula0401 (which is the target here).
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise SystemExit('HF_TOKEN not set (needed for Hula0401 push)')
    api = HfApi(token=token)

    print(f'\nUploading {len(shards_to_write)} shards to {REPO_DST}/benchcad-easy/ ...',
          flush=True)
    t0 = time.time()
    total_bytes = 0
    for idx, out_p, n_rows, n_with_png in shards_to_write:
        path_in_repo = f'benchcad-easy/{out_p.name}'
        ts = time.time()
        api.upload_file(
            path_or_fileobj=str(out_p),
            path_in_repo=path_in_repo,
            repo_id=REPO_DST, repo_type='dataset',
            commit_message=(f'shard {idx:05d}: {n_rows} rows, '
                            f'{n_with_png} with render_img'),
        )
        elapsed = time.time() - ts
        size = out_p.stat().st_size
        total_bytes += size
        print(f'  ✓ shard {idx:02d} pushed in {elapsed:.1f}s  '
              f'({size//(1024*1024)} MB)', flush=True)

    total_min = (time.time() - t0) / 60
    print(f'\nAll {len(shards_to_write)} shards uploaded in {total_min:.1f} min  '
          f'(total {total_bytes//(1024*1024)} MB)')


if __name__ == '__main__':
    main()
