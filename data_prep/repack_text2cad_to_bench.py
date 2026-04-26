"""Phase C: rewrite all text2cad cadquery code to BenchCAD-style and push to HF.

Pipeline (per split: train/val/test):
  1. Read data/text2cad/{split}.pkl  →  rows of {uid, description}
  2. For each row: load data/text2cad/cadquery/{uid}.py, run v2 rewriter
     (fallback to raw on rewrite_fail), assemble {uid, description, code}
  3. Write a new pkl + a parquet shard (parquet for HF compatibility)
  4. Upload to Hula0401/cad-sft/text2cad-bench/{split}.{pkl,parquet}

Text2CAD has no images and no STL, so this is pure text+code rewriting:
no rendering, no exec, no RAM pressure. Single-process is fine; ~90k files
take ~3-5 min total.

Usage:
  set -a; source .env; set +a
  uv run python -m data_prep.repack_text2cad_to_bench
  uv run python -m data_prep.repack_text2cad_to_bench --dry-run
  uv run python -m data_prep.repack_text2cad_to_bench --splits train  # one split
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_prep.rewrite_recode_to_benchcad_v2 import rewrite_source


def _process_split(split, src_dir, code_dir, out_dir, repo_id, dst_prefix, dry_run):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    pkl_path = src_dir / f'{split}.pkl'
    if not pkl_path.exists():
        print(f'[{split}] {pkl_path} missing, skip')
        return 0, 0

    with pkl_path.open('rb') as f:
        rows = pickle.load(f)
    print(f'[{split}] {len(rows)} rows from {pkl_path}', flush=True)

    n_rewritten = 0
    n_fallback = 0
    n_missing = 0
    new_rows = []
    t0 = time.time()
    for i, row in enumerate(rows):
        uid = row['uid']
        desc = row.get('description', '')
        py_path = code_dir / f'{uid}.py'
        if not py_path.exists():
            n_missing += 1
            continue
        try:
            raw = py_path.read_text()
        except Exception:
            n_missing += 1
            continue
        try:
            new_code = rewrite_source(raw)
            n_rewritten += 1
        except Exception:
            new_code = raw  # fallback
            n_fallback += 1
        new_rows.append({'uid': uid, 'description': desc, 'code': new_code})
        if (i + 1) % 10000 == 0:
            print(f'  {i + 1}/{len(rows)}  rewritten={n_rewritten} fallback={n_fallback} missing={n_missing}', flush=True)

    print(f'[{split}] done in {time.time()-t0:.1f}s. '
          f'final={len(new_rows)} rewritten={n_rewritten} fallback={n_fallback} missing={n_missing}', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_out = out_dir / f'{split}.pkl'
    parquet_out = out_dir / f'{split}.parquet'
    with pkl_out.open('wb') as f:
        pickle.dump(new_rows, f)
    print(f'  wrote {pkl_out} ({pkl_out.stat().st_size / 1024 / 1024:.1f} MB)', flush=True)

    table = pa.table({
        'uid': [r['uid'] for r in new_rows],
        'description': [r['description'] for r in new_rows],
        'code': [r['code'] for r in new_rows],
    })
    pq.write_table(table, str(parquet_out), compression='snappy')
    print(f'  wrote {parquet_out} ({parquet_out.stat().st_size / 1024 / 1024:.1f} MB)', flush=True)

    if not dry_run:
        api = HfApi()
        for f, label in [(pkl_out, 'pkl'), (parquet_out, 'parquet')]:
            remote = f'{dst_prefix}/{f.name}'
            print(f'  uploading {label} → {repo_id}:{remote} ...', flush=True)
            tu = time.time()
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type='dataset',
                token=os.environ.get('HF_TOKEN'),
                commit_message=f'phase-C: bench-style text2cad {split}.{label}',
            )
            print(f'    uploaded in {time.time()-tu:.1f}s', flush=True)

    # Cleanup local
    pkl_out.unlink()
    parquet_out.unlink()

    return len(new_rows), n_rewritten


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src-dir', default='data/text2cad')
    ap.add_argument('--code-dir', default='data/text2cad/cadquery')
    ap.add_argument('--out-dir', default='data/_phase_c_out')
    ap.add_argument('--repo-id', default='Hula0401/cad-sft')
    ap.add_argument('--dst-prefix', default='text2cad-bench')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not os.environ.get('HF_TOKEN'):
        print('ERROR: HF_TOKEN not set; run `set -a; source .env; set +a` first.')
        sys.exit(1)

    src_dir = Path(args.src_dir)
    code_dir = Path(args.code_dir)
    out_dir = Path(args.out_dir)

    sum_rows = 0
    sum_rewritten = 0
    for split in args.splits:
        rows, rewritten = _process_split(
            split=split, src_dir=src_dir, code_dir=code_dir,
            out_dir=out_dir, repo_id=args.repo_id,
            dst_prefix=args.dst_prefix, dry_run=args.dry_run,
        )
        sum_rows += rows
        sum_rewritten += rewritten

    print(f'\n=== Phase C done ===')
    print(f'  total rows: {sum_rows}  rewritten: {sum_rewritten} '
          f'({100*sum_rewritten/max(sum_rows,1):.1f}%)')


if __name__ == '__main__':
    main()
