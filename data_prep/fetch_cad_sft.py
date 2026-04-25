"""Materialise Hula0401/cad-sft (recode-20k + text2cad) into local files.

Layout produced:
  data/cad-recode-20k/
    train/{stem}.py              # code
    train/{stem}_render.png      # pre-rendered 4-view composite
    val/{stem}.py                # 5% hash-seeded split
    val/{stem}_render.png
    train.pkl                    # [{uid, py_path, png_path}]
    val.pkl
  data/text2cad/
    cadquery/{uid}.py            # untarred from cadquery.tar.gz
    train.pkl val.pkl test.pkl   # raw pkls (rows: {uid, description})

NOTE: this corpus has **no STL**, so pc-modality training is unsupported.
CadRecode20kDataset uses img-modality only (pre-rendered PNG from parquet).
Text2CADDataset is text-only (no visual input).

Usage:
  set -a; source .env; set +a
  uv run python -m data_prep.fetch_cad_sft --what all
Flags:
  --what recode20k|text2cad|all   (default: all)
  --out DIR                        default: data/
  --val-frac                       hash-split fraction for recode-20k val (default 0.05)
  --workers                        parallel workers for extraction (default 8)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import pickle
import sys
import tarfile
from pathlib import Path


def _split(stem: str, val_frac: float) -> str:
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
    return 'val' if (h % 1000) < int(1000 * val_frac) else 'train'


def fetch_recode_20k(out_root: Path, val_frac: float = 0.05) -> None:
    """Download 9 parquet shards and materialise to py + render_img.png."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    token = os.environ.get('HF_TOKEN')
    cache = out_root.parent / '_cache_cad_sft'
    cache.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    for i in range(9):
        fname = f'cad-recode-20k/train-{i:05d}-of-00009.parquet'
        print(f'[recode-20k] downloading {fname} ...', flush=True)
        p = hf_hub_download('Hula0401/cad-sft', fname, repo_type='dataset',
                            token=token, local_dir=str(cache))
        shard_paths.append(Path(p))

    (out_root / 'train').mkdir(parents=True, exist_ok=True)
    (out_root / 'val').mkdir(parents=True, exist_ok=True)
    ann: dict[str, list[dict]] = {'train': [], 'val': []}

    n_total = 0
    for shard in shard_paths:
        t = pq.read_table(shard)
        rows = t.to_pylist()
        for row in rows:
            stem = str(row['stem']).replace('/', '_')   # batch_94/949381 → batch_94_949381
            code = row['code']
            render = row.get('render_img')
            if render is None:
                continue
            png_bytes = render.get('bytes') if isinstance(render, dict) else render

            split = _split(stem, val_frac)
            split_dir = out_root / split
            py_path = split_dir / f'{stem}.py'
            png_path = split_dir / f'{stem}_render.png'
            if not py_path.exists():
                py_path.write_text(code)
            if not png_path.exists():
                png_path.write_bytes(png_bytes)

            ann[split].append({
                'uid': stem,
                'py_path': str(py_path.relative_to(out_root)),
                'png_path': str(png_path.relative_to(out_root)),
            })
            n_total += 1
            if n_total % 2000 == 0:
                print(f'  materialised {n_total}', flush=True)

    for split in ('train', 'val'):
        pkl = out_root / f'{split}.pkl'
        with pkl.open('wb') as fp:
            pickle.dump(ann[split], fp)
        print(f'  {split}.pkl: {len(ann[split])} rows → {pkl}', flush=True)

    print(f'[recode-20k] done. total {n_total} rows.\n', flush=True)


def fetch_text2cad(out_root: Path) -> None:
    """Download text2cad pkls + untar the cadquery code archive."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get('HF_TOKEN')
    out_root.mkdir(parents=True, exist_ok=True)

    # pkls
    for split in ('train', 'val', 'test'):
        fname = f'text2cad/{split}.pkl'
        print(f'[text2cad] downloading {fname} ...', flush=True)
        p = hf_hub_download('Hula0401/cad-sft', fname, repo_type='dataset',
                            token=token, local_dir=str(out_root.parent / '_cache_cad_sft'))
        # Keep a copy directly in out_root for CadRecodeDataset-style access
        target = out_root / f'{split}.pkl'
        if target.exists():
            target.unlink()
        os.link(p, target)
        with open(p, 'rb') as f:
            import pickle as _p
            rows = _p.load(f)
        print(f'  {split}: {len(rows)} rows', flush=True)

    # tar
    print(f'[text2cad] downloading cadquery.tar.gz ...', flush=True)
    tar_p = hf_hub_download('Hula0401/cad-sft', 'text2cad/cadquery.tar.gz',
                            repo_type='dataset', token=token,
                            local_dir=str(out_root.parent / '_cache_cad_sft'))
    cq_dir = out_root / 'cadquery'
    # Guard on "contains .py files" (not just dir-exists) so a partial /
    # interrupted extraction is retried cleanly on the next run.
    n_existing = sum(1 for _ in cq_dir.glob('*.py')) if cq_dir.exists() else 0
    if n_existing == 0:
        cq_dir.mkdir(parents=True, exist_ok=True)
        print('  extracting ...', flush=True)
        with tarfile.open(tar_p) as tf:
            # filter='data' (Python 3.12+) blocks path-traversal entries
            # (CVE-2007-4559). Required for safe untar of foreign archives.
            tf.extractall(out_root, filter='data')
    n_py = sum(1 for _ in cq_dir.glob('*.py'))
    print(f'  cadquery/: {n_py} .py files\n', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', default='all', choices=['recode20k', 'text2cad', 'all'])
    ap.add_argument('--out', default='data')
    ap.add_argument('--val-frac', type=float, default=0.05)
    args = ap.parse_args()
    out_root = Path(args.out)

    if args.what in ('recode20k', 'all'):
        fetch_recode_20k(out_root / 'cad-recode-20k', args.val_frac)
    if args.what in ('text2cad', 'all'):
        fetch_text2cad(out_root / 'text2cad')

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
