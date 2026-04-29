"""Merge per-VM render pickles into the final parquet for BenchCAD/benchcad-easy
and (optionally) upload back to HF.

Each render pickle is the output of `data_prep/render_benchcad_easy.py
--renders-out`, structured as:
    {'renders': {row_idx: png_bytes, ...},
     'slice':   [start_row, end_row),
     'n_ok': int, 'n_failed': int, 'fail_reasons': {...}}

Usage (after VM1 + VM2 pickles are present on this machine):

    uv run python -m data_prep.merge_benchcad_easy_renders \\
        --pickles \\
            /ephemeral/data/benchcad_easy_renders/vm1_shards_0-33.pkl \\
            /ephemeral/data/benchcad_easy_renders/vm2_shards_33-55.pkl \\
        --upload

Asserts no overlapping row_idx between pickles.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPO = 'BenchCAD/benchcad-easy'
PARQUET_PATH_IN_REPO = 'data/test-00000-of-00001.parquet'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pickles', nargs='+', required=True,
                    help='Render pickle paths from each VM')
    ap.add_argument('--out-parquet',
                    default=str(REPO_ROOT / 'data/_hf_upload/benchcad-easy/data/test-00000-of-00001.parquet'))
    ap.add_argument('--upload', action='store_true',
                    help='Push merged parquet to HF after writing')
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, HfApi

    # Load all pickles, combine
    all_renders: dict[int, bytes] = {}
    total_ok = 0
    total_failed = 0
    for p in args.pickles:
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        r = obj['renders']
        sl = obj.get('slice', [None, None])
        n_ok = obj.get('n_ok', len(r))
        n_failed = obj.get('n_failed', 0)
        print(f'{p}: slice={sl}, ok={n_ok}, failed={n_failed}, '
              f'fail_reasons={obj.get("fail_reasons", {})}', flush=True)
        # Assert no overlap with already-loaded keys
        overlap = set(all_renders) & set(r)
        if overlap:
            raise SystemExit(f'OVERLAP between {p} and earlier pickles: '
                             f'{len(overlap)} row_idx in both '
                             f'(first 5: {list(overlap)[:5]})')
        all_renders.update(r)
        total_ok += n_ok
        total_failed += n_failed

    print(f'\nMerged {len(all_renders)} renders from {len(args.pickles)} pickles '
          f'(total_ok={total_ok}, total_failed={total_failed})\n', flush=True)

    # Download original parquet
    print(f'Downloading {REPO}/{PARQUET_PATH_IN_REPO} ...', flush=True)
    parquet_in = hf_hub_download(REPO, PARQUET_PATH_IN_REPO, repo_type='dataset')
    table = pq.read_table(parquet_in)
    n_total = len(table)
    imgs = table['composite_png'].to_pylist()
    n_with_old = sum(1 for r in imgs if r and r.get('bytes'))
    print(f'  rows: {n_total}, with_png_old={n_with_old}', flush=True)

    # Build new composite_png list
    n_added = 0
    for i, r in enumerate(imgs):
        if i in all_renders:
            if r and r.get('bytes'):
                # already had one — skip overwriting
                continue
            imgs[i] = {'bytes': all_renders[i], 'path': None}
            n_added += 1
    n_with_new = sum(1 for r in imgs if r and r.get('bytes'))
    print(f'  added={n_added}, with_png_new={n_with_new} ({n_with_new}/{n_total}'
          f' = {100*n_with_new/n_total:.1f}%)', flush=True)

    # Build new table preserving all other columns
    new_cols = {}
    for col in table.column_names:
        if col == 'composite_png':
            new_cols[col] = imgs
        else:
            new_cols[col] = table[col].to_pylist()
    new_table = pa.table(new_cols, schema=table.schema)

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, out, compression='snappy')
    print(f'\nWrote merged parquet → {out}  ({out.stat().st_size // (1024*1024)} MB)')

    if args.upload:
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise SystemExit('HF_TOKEN not set in env')
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=PARQUET_PATH_IN_REPO,
            repo_id=REPO, repo_type='dataset',
            commit_message=(f'Fill composite_png: +{n_added} renders '
                            f'({n_with_old}→{n_with_new}/{n_total})'),
        )
        print(f'Uploaded to HF {REPO}/{PARQUET_PATH_IN_REPO}')


if __name__ == '__main__':
    main()
