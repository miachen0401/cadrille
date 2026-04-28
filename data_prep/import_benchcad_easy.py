"""Import BenchCAD/benchcad-easy → repackage into Hula0401/cad-sft/benchcad-easy/.

Unlike `import_benchcad_simple.py`, this dataset already ships a 268×268
`composite_png` per row, so there is **no rendering step** — we just stream
the upstream rows and re-pack them into the canonical Hula0401/cad-sft
image-conditioned schema (stem / code / render_img / family / difficulty /
n_ops / ops_json / base_plane). 109,804 rows → ~55 shards of 2k.

Pipeline (per row):
  1. Read row from BenchCAD/benchcad-easy `data/test-00000-of-00001.parquet`
  2. Map columns:
       gt_code        → code   (no shim to strip on this dataset)
       composite_png  → render_img.bytes  (PNG-encoded)
       feature_count  → n_ops              (kept as-is per existing prefixes)
       ops_used       → ops_json           (already JSON-stringified list)
       stem/family/difficulty/base_plane    direct passthrough
  3. Append to in-memory shard buffer; flush every --shard-size rows.
  4. Upload each shard with the same wall-clock-timeout retry wrapper used by
     prepare_hf_cadrecode_v2.py (HfApi.upload_file has been observed to hang
     silently on large repos).

Usage (smoke):
    set -a; source .env; set +a
    uv run python -m data_prep.import_benchcad_easy --n 50 --no-upload

Usage (full):
    nohup uv run python -m data_prep.import_benchcad_easy \\
        --shard-size 2000 \\
        > logs/benchcad_easy_upload.log 2>&1 &

Resume after partial run:
    --start-shard N        skip the first N output shards (already uploaded)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


SRC_REPO       = 'BenchCAD/benchcad-easy'
SRC_PARQUET    = 'data/test-00000-of-00001.parquet'
DST_REPO       = 'Hula0401/cad-sft'
DST_PREFIX     = 'benchcad-easy'
UPLOAD_TIMEOUT = 300  # seconds — abandon hung threads


def _push_shard(rows: list[dict], shard_idx: int, total_shards: int,
                out_dir: Path, dry_run: bool) -> None:
    """Write rows → parquet, upload to HF, delete local. Same retry pattern as
    prepare_hf_cadrecode_v2.py (HfApi.upload_file has no timeout)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fname_remote = f'{DST_PREFIX}/train-{shard_idx:05d}-of-{total_shards:05d}.parquet'
    fname_local  = out_dir / fname_remote.replace('/', '_')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build pyarrow Table with the canonical schema
    schema = pa.schema([
        ('stem',       pa.string()),
        ('code',       pa.string()),
        ('render_img', pa.struct([
            ('bytes', pa.binary()),
            ('path',  pa.null()),
        ])),
        ('family',     pa.string()),
        ('difficulty', pa.string()),
        ('n_ops',      pa.int64()),
        ('ops_json',   pa.string()),
        ('base_plane', pa.string()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, fname_local, compression='snappy')
    n = len(rows)
    sz = fname_local.stat().st_size / 1024 / 1024
    print(f'  shard {shard_idx + 1}/{total_shards}: {n} rows, {sz:.1f} MB → {fname_remote}', flush=True)

    if dry_run:
        fname_local.unlink()
        return

    from huggingface_hub import HfApi
    api = HfApi()
    for attempt in range(3):
        result = {'ok': False, 'err': None}
        def task():
            try:
                api.upload_file(
                    path_or_fileobj=str(fname_local),
                    path_in_repo=fname_remote,
                    repo_id=DST_REPO,
                    repo_type='dataset',
                    token=os.environ.get('HF_TOKEN'),
                    commit_message=f'benchcad-easy shard {shard_idx + 1}/{total_shards}',
                )
                result['ok'] = True
            except Exception as e:
                result['err'] = f'{type(e).__name__}: {e}'
        t0 = time.time()
        th = threading.Thread(target=task, daemon=True); th.start()
        th.join(timeout=UPLOAD_TIMEOUT)
        if result['ok']:
            print(f'    uploaded in {time.time()-t0:.1f}s', flush=True)
            break
        if th.is_alive():
            print(f'    !! hang > {UPLOAD_TIMEOUT}s, retry {attempt+1}/3', flush=True)
        elif result['err']:
            print(f'    upload err: {result["err"]}; retry {attempt+1}/3', flush=True)
    else:
        raise RuntimeError(f'upload failed after 3 retries: {fname_remote}')
    fname_local.unlink()


def _to_png_bytes(img) -> bytes:
    """Re-encode a PIL Image as PNG bytes (the upstream parquet stores PNG
    already, but datasets.Image() decodes it to PIL on .__getitem__()).
    """
    if isinstance(img, dict) and img.get('bytes'):
        return img['bytes']
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',           type=int, default=None,
                    help='Cap total rows processed (default: all)')
    ap.add_argument('--shard-size',  type=int, default=2000)
    ap.add_argument('--start-shard', type=int, default=0,
                    help='Resume from output shard N (skip first N*shard_size rows)')
    ap.add_argument('--out-dir',     type=str, default='/tmp/benchcad_easy_shards',
                    help='Local staging dir for parquet shards before upload')
    ap.add_argument('--no-upload',   action='store_true',
                    help='Build shards locally but skip the HF upload step')
    args = ap.parse_args()

    if not args.no_upload and not os.environ.get('HF_TOKEN'):
        print('HF_TOKEN not set', file=sys.stderr); sys.exit(1)

    # ── 1. Download upstream parquet (single file)  ───────────────────────
    from huggingface_hub import hf_hub_download
    print(f'Downloading {SRC_REPO}/{SRC_PARQUET} ...', flush=True)
    t0 = time.time()
    cache_dir = Path('/tmp/benchcad_easy_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_path = hf_hub_download(
        repo_id=SRC_REPO, filename=SRC_PARQUET,
        repo_type='dataset', token=os.environ.get('HF_TOKEN'),
        local_dir=str(cache_dir),
    )
    print(f'  done in {time.time()-t0:.1f}s ({Path(src_path).stat().st_size/1024/1024:.1f} MB)', flush=True)

    # ── 2. Stream rows via the datasets library (handles Image decode)  ──
    print('Loading parquet into a dataset stream ...', flush=True)
    from datasets import load_dataset
    ds = load_dataset(SRC_REPO, split='test', token=os.environ.get('HF_TOKEN'))
    n_total = len(ds) if args.n is None else min(len(ds), args.n)
    total_shards = (n_total + args.shard_size - 1) // args.shard_size
    print(f'  {len(ds)} upstream rows, processing {n_total} → {total_shards} output shards', flush=True)

    n_skip = args.start_shard * args.shard_size
    if n_skip > 0:
        print(f'(resume: skipping first {n_skip} rows for start-shard={args.start_shard})', flush=True)

    # ── 3. Pack + upload in 2k-row shards  ───────────────────────────────
    out_dir = Path(args.out_dir)
    buf: list[dict] = []
    shard_idx = args.start_shard
    n_processed = 0
    n_skipped = 0

    for i in range(n_total):
        if i < n_skip:
            n_skipped += 1
            continue
        row = ds[i]
        png_bytes = _to_png_bytes(row['composite_png'])
        rec = {
            'stem':       row['stem'],
            'code':       row['gt_code'],
            'render_img': {'bytes': png_bytes, 'path': None},
            'family':     row.get('family') or '',
            'difficulty': row.get('difficulty') or '',
            'n_ops':      int(row.get('feature_count') or 0),
            'ops_json':   row.get('ops_used') or '[]',
            'base_plane': row.get('base_plane') or '',
        }
        buf.append(rec)
        n_processed += 1
        if len(buf) >= args.shard_size:
            _push_shard(buf, shard_idx, total_shards, out_dir, dry_run=args.no_upload)
            buf.clear()
            shard_idx += 1

    if buf:
        _push_shard(buf, shard_idx, total_shards, out_dir, dry_run=args.no_upload)
        shard_idx += 1

    print(f'\nDone. processed={n_processed} skipped={n_skipped} '
          f'output_shards={shard_idx - args.start_shard}', flush=True)
    print(f'  → https://huggingface.co/datasets/{DST_REPO}/tree/main/{DST_PREFIX}',
          flush=True)


if __name__ == '__main__':
    main()
