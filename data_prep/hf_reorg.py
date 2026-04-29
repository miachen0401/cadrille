"""HF dataset reorg: delete legacy folders + upload text2cad-bench-img.

Per user 2026-04-28:
  1. Delete `Hula0401/cad-sft/text2cad/` entirely (low-quality legacy)
  2. Rename `Hula0401/cad-sft/cad-recode-20k/` → `cad-recode-20k_archive/`
     (no longer used, but keep as historical record)
  3. Upload new `Hula0401/cad-sft/text2cad-bench-img/` parquet shards with
     {uid, description, code, render_img} schema for the 53 339 cleaned
     text2cad-bench items (image-conditioned training source for v3+).

Usage:
    uv run python -m data_prep.hf_reorg --do delete-text2cad
    uv run python -m data_prep.hf_reorg --do rename-recode20k
    uv run python -m data_prep.hf_reorg --do upload-text2cad-img
    uv run python -m data_prep.hf_reorg --do all   # all three sequentially
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REPO = 'Hula0401/cad-sft'


def get_api():
    from huggingface_hub import HfApi
    token = os.environ.get('HF_TOKEN')
    if not token:
        # try .env
        env = REPO_ROOT / '.env'
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith('HF_TOKEN='):
                    token = line.split('=', 1)[1].strip().strip("'\"")
                    break
    if not token:
        raise RuntimeError('HF_TOKEN not set')
    return HfApi(token=token), token


def delete_text2cad():
    api, token = get_api()
    print('--- Deleting Hula0401/cad-sft/text2cad/ ---')
    files = api.list_repo_files(REPO, repo_type='dataset')
    targets = [f for f in files if f.startswith('text2cad/')]
    print(f'Targets: {len(targets)} files')
    for f in targets:
        try:
            api.delete_file(f, repo_id=REPO, repo_type='dataset',
                            commit_message=f'delete legacy text2cad: {f}')
            print(f'  deleted: {f}')
        except Exception as e:
            print(f'  ERROR deleting {f}: {e}')


def rename_cad_recode_20k():
    """HF doesn't rename folders; do upload-then-delete."""
    api, token = get_api()
    print('--- Renaming cad-recode-20k → cad-recode-20k_archive ---')
    from huggingface_hub import hf_hub_download
    files = api.list_repo_files(REPO, repo_type='dataset')
    targets = [f for f in files if f.startswith('cad-recode-20k/')]
    print(f'Targets: {len(targets)} files')

    cache = REPO_ROOT / 'data' / '_hf_reorg_cache' / 'cad-recode-20k'
    cache.mkdir(parents=True, exist_ok=True)

    for f in targets:
        try:
            local = hf_hub_download(REPO, f, repo_type='dataset',
                                     local_dir=str(cache.parent),
                                     token=token)
            new_path_in_repo = f.replace('cad-recode-20k/', 'cad-recode-20k_archive/', 1)
            api.upload_file(path_or_fileobj=local,
                            path_in_repo=new_path_in_repo,
                            repo_id=REPO, repo_type='dataset',
                            commit_message=f'archive: copy {f} → {new_path_in_repo}')
            print(f'  copied: {f} → {new_path_in_repo}')
            api.delete_file(f, repo_id=REPO, repo_type='dataset',
                            commit_message=f'archive: remove old {f}')
            print(f'  deleted old: {f}')
        except Exception as e:
            print(f'  ERROR on {f}: {e}')


def upload_text2cad_bench_img(shards: int = 6, items_per_shard: int = 9000):
    """Pack text2cad-bench train into parquet shards with render_img bytes,
    upload to text2cad-bench-img/."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    api, token = get_api()
    root = REPO_ROOT / 'data' / 'text2cad-bench'
    rows = pickle.load(open(root / 'train.pkl', 'rb'))
    print(f'--- Packing {len(rows)} text2cad-bench items into parquet shards ---')

    shard_dir = REPO_ROOT / 'data' / '_hf_upload' / 'text2cad-bench-img'
    shard_dir.mkdir(parents=True, exist_ok=True)

    actual_shards = max(1, (len(rows) + items_per_shard - 1) // items_per_shard)
    print(f'Will write {actual_shards} shards ~{items_per_shard} items each')

    for s in range(actual_shards):
        chunk = rows[s * items_per_shard : (s + 1) * items_per_shard]
        records = {'uid': [], 'description': [], 'code': [], 'render_img': []}
        for r in chunk:
            png_path = root / r['png_path']
            if not png_path.exists():
                continue
            try:
                png_bytes = png_path.read_bytes()
            except Exception:
                continue
            records['uid'].append(r['uid'])
            records['description'].append(r.get('description') or '')
            records['code'].append(r.get('code') or '')
            records['render_img'].append({'bytes': png_bytes, 'path': None})

        if not records['uid']:
            continue

        # Note: render_img as struct{bytes, path} mimics HF datasets Image() column
        schema = pa.schema([
            pa.field('uid', pa.string()),
            pa.field('description', pa.string()),
            pa.field('code', pa.string()),
            pa.field('render_img', pa.struct([
                pa.field('bytes', pa.binary()),
                pa.field('path', pa.string()),
            ])),
        ])
        table = pa.table(records, schema=schema)
        out = shard_dir / f'train-{s:05d}-of-{actual_shards:05d}.parquet'
        pq.write_table(table, out, compression='snappy')
        print(f'  wrote {out.name}: {len(records["uid"])} rows, '
              f'{out.stat().st_size // (1024*1024)} MB')

    # Upload
    print(f'\nUploading {actual_shards} shards to {REPO}/text2cad-bench-img/ ...')
    api.upload_folder(
        folder_path=str(shard_dir),
        path_in_repo='text2cad-bench-img',
        repo_id=REPO, repo_type='dataset',
        commit_message='Add text2cad-bench-img: 53k items with render_img bytes for image-conditioned training',
    )
    print('Upload complete.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--do', required=True,
                    choices=['delete-text2cad', 'rename-recode20k',
                             'upload-text2cad-img', 'all'])
    args = ap.parse_args()

    if args.do in ('delete-text2cad', 'all'):
        delete_text2cad()
    if args.do in ('rename-recode20k', 'all'):
        rename_cad_recode_20k()
    if args.do in ('upload-text2cad-img', 'all'):
        upload_text2cad_bench_img()


if __name__ == '__main__':
    main()
