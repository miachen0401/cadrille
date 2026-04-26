"""Phase A: re-pack the existing 20k cad-recode parquet shards on HF with
v2-rewritten BenchCAD-style code. PNG bytes are reused (mesh geometry is
unchanged by the AST rewrite, so re-rendering is unnecessary).

Pipeline:
  1. Download 9 existing parquet shards from Hula0401/cad-sft/cad-recode-20k/
  2. For each row (stem, code, render_img):
     a. Look up data/cad-recode-v1.5/train/{batch}/{stem}.py (raw recode style)
     b. Run rewrite_source() to get bench-style code
     c. On rewrite_fail, fall back to the raw (compact) code from .py file
        (NOT the parquet's `code` field — those are the same, but read from
        disk to ensure consistency with what new 80k pipeline will use)
     d. Replace `code` field; keep `stem` and `render_img` bytes verbatim
  3. Upload to Hula0401/cad-sft/cad-recode-bench/train-XXXXX-of-00009.parquet
  4. Verify by sampling.

RAM-safe: processes one shard at a time (~400 MB peak per shard), single-process.
HF cache caps at ~150 MB / shard so total disk transient is bounded.

Usage:
  set -a; source .env; set +a
  uv run python -m data_prep.repack_recode_to_bench
  uv run python -m data_prep.repack_recode_to_bench --dry-run     # skip upload
  uv run python -m data_prep.repack_recode_to_bench --limit-shards 1  # smoke test
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_prep.rewrite_recode_to_benchcad_v2 import rewrite_source


def _process_shard(shard_idx, total, src_repo, src_prefix, dst_prefix,
                   src_dir, cache_dir, out_dir, dry_run):
    """Download one shard, rewrite codes, write new shard, upload, cleanup."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, HfApi

    fname_in = f'{src_prefix}/train-{shard_idx:05d}-of-{total:05d}.parquet'
    fname_out_local = out_dir / f'train-{shard_idx:05d}-of-{total:05d}.parquet'
    fname_out_remote = f'{dst_prefix}/train-{shard_idx:05d}-of-{total:05d}.parquet'

    print(f'[shard {shard_idx + 1}/{total}] downloading {fname_in} ...', flush=True)
    t0 = time.time()
    p = hf_hub_download(src_repo, fname_in, repo_type='dataset',
                        token=os.environ.get('HF_TOKEN'), local_dir=str(cache_dir))
    print(f'  downloaded in {time.time()-t0:.1f}s', flush=True)

    table = pq.read_table(p)
    n_rows = table.num_rows
    print(f'  shard rows: {n_rows}', flush=True)

    stems = table.column('stem').to_pylist()
    images = table.column('render_img').to_pylist()  # list of {bytes, path}

    # Rewrite codes
    new_codes = []
    n_rewritten = 0
    n_fallback_disk = 0
    n_fallback_parquet = 0
    raw_codes_in_parquet = table.column('code').to_pylist()

    for i, stem in enumerate(stems):
        # stem is e.g. "batch_94/949381" — locate raw .py
        py_path = src_dir / f'{stem}.py'
        raw_disk = None
        if py_path.exists():
            try:
                raw_disk = py_path.read_text()
            except Exception:
                pass
        # Source for rewrite: prefer disk (canonical), else parquet's `code`
        raw_src = raw_disk if raw_disk is not None else raw_codes_in_parquet[i]

        try:
            new_code = rewrite_source(raw_src)
            n_rewritten += 1
        except Exception:
            new_code = raw_src  # fallback: raw recode style
            if raw_disk is not None:
                n_fallback_disk += 1
            else:
                n_fallback_parquet += 1

        new_codes.append(new_code)

    print(f'  rewritten: {n_rewritten}/{n_rows}  '
          f'fallback_disk: {n_fallback_disk}  fallback_parquet: {n_fallback_parquet}',
          flush=True)

    # Build new table — same schema, replace `code` field
    new_table = pa.table({
        'stem': stems,
        'code': new_codes,
        'render_img': images,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, str(fname_out_local), compression='snappy')
    size_mb = fname_out_local.stat().st_size / 1024 / 1024
    print(f'  wrote {fname_out_local} ({size_mb:.1f} MB)', flush=True)

    if not dry_run:
        api = HfApi()
        print(f'  uploading to {src_repo}:{fname_out_remote} ...', flush=True)
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(fname_out_local),
            path_in_repo=fname_out_remote,
            repo_id=src_repo,
            repo_type='dataset',
            token=os.environ.get('HF_TOKEN'),
            commit_message=f'phase-A repack: bench-style code for shard {shard_idx + 1}/{total}',
        )
        print(f'  uploaded in {time.time()-t0:.1f}s', flush=True)

    # Cleanup local files to save disk
    fname_out_local.unlink()
    Path(p).unlink()
    return n_rows, n_rewritten


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src-repo', default='Hula0401/cad-sft')
    ap.add_argument('--src-prefix', default='cad-recode-20k')
    ap.add_argument('--dst-prefix', default='cad-recode-bench')
    ap.add_argument('--total-shards', type=int, default=9)
    ap.add_argument('--src-dir', default='data/cad-recode-v1.5/train')
    ap.add_argument('--cache-dir', default='data/_cache_phase_a')
    ap.add_argument('--out-dir', default='data/_phase_a_out')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit-shards', type=int, default=None,
                    help='Only process first N shards (for smoke test)')
    ap.add_argument('--start-shard', type=int, default=0,
                    help='Resume from shard N (0-indexed)')
    args = ap.parse_args()

    if not os.environ.get('HF_TOKEN'):
        print('ERROR: HF_TOKEN not set; run `set -a; source .env; set +a` first.')
        sys.exit(1)

    src_dir = Path(args.src_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.total_shards
    end = min(args.start_shard + (args.limit_shards or total), total)

    sum_rows = 0
    sum_rewritten = 0
    for i in range(args.start_shard, end):
        rows, rewritten = _process_shard(
            shard_idx=i, total=total,
            src_repo=args.src_repo,
            src_prefix=args.src_prefix,
            dst_prefix=args.dst_prefix,
            src_dir=src_dir, cache_dir=cache_dir, out_dir=out_dir,
            dry_run=args.dry_run,
        )
        sum_rows += rows
        sum_rewritten += rewritten

    print(f'\n=== Phase A done ===')
    print(f'  shards processed: {end - args.start_shard}')
    print(f'  total rows: {sum_rows}  rewritten: {sum_rewritten} ({100*sum_rewritten/max(sum_rows,1):.1f}%)')
    print(f'  dry_run: {args.dry_run}')


if __name__ == '__main__':
    main()
