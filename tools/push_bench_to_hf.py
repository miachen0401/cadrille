"""Package recode-bench + text2cad-bench + cad_bench into parquet shards and push
to `BenchCAD/cad_sft_training` as a multi-config HF dataset.

Configs:
    recode     — (uid, code)                  from data/cad-recode-v1.5-bench/
    text2cad   — (uid, code, description)     from data/text2cad-bench/
    benchcad   — (uid, code, family, difficulty, base_plane, feature_tags,
                  feature_count, ops_used, qa_pairs, iso_tags)
                 from HF hub BenchCAD/cad_bench (composite_png column dropped;
                 load it separately by `stem` when needed).

All rows are filtered by `--max-code-len` (default 1000) — longer samples are
dropped to keep training throughput predictable.

Usage:
  uv run python tools/push_bench_to_hf.py \\
      --repo BenchCAD/cad_sft_training --private
"""
import argparse
import os
import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo


SHARD_ROW_TARGET = 200_000  # rows per shard


def iter_recode(split_dir: Path, max_len: int):
    for p in sorted(split_dir.rglob('*.py')):
        rel = p.relative_to(split_dir).with_suffix('')
        uid = str(rel).replace(os.sep, '/')
        code = p.read_text()
        if len(code) > max_len:
            continue
        yield uid, code


def iter_text2cad(split_pkl: Path, code_dir: Path, max_len: int):
    with open(split_pkl, 'rb') as f:
        rows = pickle.load(f)
    for r in rows:
        uid = r['uid']
        py = code_dir / f'{uid}.py'
        if not py.exists():
            continue
        code = py.read_text()
        if len(code) > max_len:
            continue
        yield uid, code, r.get('description', '')


def iter_benchcad(hf_token: str, max_len: int):
    """Stream cad_bench, drop composite_png, apply length filter."""
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench', split='test', token=hf_token)
    for r in ds:
        code = r['gt_code']
        if len(code) > max_len:
            continue
        yield (
            r['stem'],
            code,
            r.get('family', ''),
            r.get('difficulty', ''),
            r.get('base_plane', ''),
            r.get('feature_tags', ''),
            int(r.get('feature_count', 0) or 0),
            r.get('ops_used', ''),
            r.get('qa_pairs', ''),
            r.get('iso_tags', ''),
        )


def write_shards(rows_iter, schema, out_dir: Path, split_name: str, row_target=SHARD_ROW_TARGET):
    out_dir.mkdir(parents=True, exist_ok=True)
    col_names = schema.names
    buf = {c: [] for c in col_names}
    shards_written = []
    shard_idx = 0
    n_total = 0

    def flush():
        nonlocal shard_idx
        if not buf[col_names[0]]:
            return
        tmp_path = out_dir / f'{split_name}-{shard_idx:05d}.parquet'
        table = pa.Table.from_pydict(buf, schema=schema)
        pq.write_table(table, tmp_path, compression='zstd')
        shards_written.append(tmp_path)
        print(f'  wrote {tmp_path.name}: {len(buf[col_names[0]])} rows')
        for c in col_names:
            buf[c] = []
        shard_idx += 1

    for row in rows_iter:
        for c, v in zip(col_names, row):
            buf[c].append(v)
        n_total += 1
        if len(buf[col_names[0]]) >= row_target:
            flush()
    flush()

    total = len(shards_written)
    final_paths = []
    for i, p in enumerate(shards_written):
        new = out_dir / f'{split_name}-{i:05d}-of-{total:05d}.parquet'
        p.rename(new)
        final_paths.append(new)
    print(f'  {split_name}: {n_total} rows → {total} shards')
    return final_paths, n_total


def build_readme(counts, max_len):
    recode_c = counts.get('recode', {})
    t2c_c = counts.get('text2cad', {})
    bench_c = counts.get('benchcad', {})

    def tot(d):
        return sum(d.values())

    return f"""---
license: mit
language:
- en
tags:
- cad
- cadquery
- code-generation
- sft
configs:
- config_name: recode
  data_files:
  - split: train
    path: recode/train-*.parquet
  - split: val
    path: recode/val-*.parquet
- config_name: text2cad
  data_files:
  - split: train
    path: text2cad/train-*.parquet
  - split: val
    path: text2cad/val-*.parquet
  - split: test
    path: text2cad/test-*.parquet
- config_name: benchcad
  data_files:
  - split: train
    path: benchcad/train-*.parquet
---

# cad_sft_training

Unified CadQuery SFT corpus for the **Cadrille → BenchCAD** pipeline.
All code is normalised to the BenchCAD shell style:

```python
import cadquery as cq

result = (
    cq.Workplane('XY')
    .box(...)
    ...
)

# Export
show_object(result)
```

## Pipeline

This dataset is the input to **stage 1 (SFT)** of our two-stage training recipe:

1. **SFT** — this dataset. Mixed 3-way from (text2cad : recode : benchcad) with
   weights **1 : 2 : 2** (see *Mixing* below). Train a Qwen2-VL backbone
   (cadrille.py) to emit CadQuery code from an image.
2. **RL post-train** — CPPO with mesh-IoU reward on top of the SFT checkpoint
   (see `rl/` in the cadrille repo). Uses a subset of benchcad prompts.

## Configs

| config | split | rows | columns |
|---|---|---|---|
| recode | train | {recode_c.get('train', 0):,} | `uid`, `code` |
| recode | val | {recode_c.get('val', 0):,} | `uid`, `code` |
| text2cad | train | {t2c_c.get('train', 0):,} | `uid`, `code`, `description` |
| text2cad | val | {t2c_c.get('val', 0):,} | `uid`, `code`, `description` |
| text2cad | test | {t2c_c.get('test', 0):,} | `uid`, `code`, `description` |
| benchcad | train | {bench_c.get('train', 0):,} | `uid`, `code`, `family`, `difficulty`, `base_plane`, `feature_tags`, `feature_count`, `ops_used`, `qa_pairs`, `iso_tags` |

**Totals**: recode {tot(recode_c):,} · text2cad {tot(t2c_c):,} · benchcad {tot(bench_c):,}

### Filters

- `len(code) <= {max_len}` — long samples dropped for training throughput.
- recode: source is filapro/cad-recode-v1.5, AST-rewritten into bench shell.
- text2cad: Text2CAD cadquery subset, same rewrite; `description` preserved.
- benchcad: BenchCAD/cad_bench `gt_code` column (already in bench shell).
  `composite_png` stripped here — load it from `BenchCAD/cad_bench` by `uid`
  (matches `stem` there) when you need the image.

## Mixing

Recommended SFT batch mixing (use a `WeightedRandomSampler` or
`datasets.interleave_datasets`):

```python
from datasets import load_dataset, interleave_datasets

recode = load_dataset('BenchCAD/cad_sft_training', 'recode', split='train')
text2cad = load_dataset('BenchCAD/cad_sft_training', 'text2cad', split='train')
benchcad = load_dataset('BenchCAD/cad_sft_training', 'benchcad', split='train')

mixed = interleave_datasets(
    [text2cad, recode, benchcad],
    probabilities=[1/5, 2/5, 2/5],   # text2cad : recode : benchcad = 1 : 2 : 2
    stopping_strategy='all_exhausted',
    seed=42,
)
```

## Semantic preservation (recode + text2cad rewrite)

Pure AST reformatting — verified via mesh equivalence:

| config | pairs tested | pass | IoU mean / min |
|---|---|---|---|
| recode | 10,000 | 10,000 | 1.000000 / 0.999999 |
| text2cad | 30 | 30 | 1.000000 / 1.000000 |

Verifier + rewriter live in `tools/` of [Hula0401/cadrille](https://github.com/Hula0401/cadrille).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recode-root', type=str, default='data/cad-recode-v1.5-bench')
    ap.add_argument('--text2cad-root', type=str, default='data/text2cad-bench')
    ap.add_argument('--repo', type=str, default='BenchCAD/cad_sft_training')
    ap.add_argument('--staging', type=str, default='/tmp/cad_sft_training_staging')
    ap.add_argument('--max-code-len', type=int, default=1000)
    ap.add_argument('--private', action='store_true')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--skip-push', action='store_true')
    ap.add_argument('--configs', nargs='+', default=['recode', 'text2cad', 'benchcad'])
    args = ap.parse_args()

    token = os.environ.get('BenchCAD_HF_TOKEN') or os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('No HF token in env. Source .env first.')

    staging = Path(args.staging)
    recode_root = Path(args.recode_root)
    t2c_root = Path(args.text2cad_root)

    recode_schema = pa.schema([('uid', pa.string()), ('code', pa.string())])
    t2c_schema = pa.schema([
        ('uid', pa.string()),
        ('code', pa.string()),
        ('description', pa.string()),
    ])
    bench_schema = pa.schema([
        ('uid', pa.string()),
        ('code', pa.string()),
        ('family', pa.string()),
        ('difficulty', pa.string()),
        ('base_plane', pa.string()),
        ('feature_tags', pa.string()),
        ('feature_count', pa.int32()),
        ('ops_used', pa.string()),
        ('qa_pairs', pa.string()),
        ('iso_tags', pa.string()),
    ])

    counts = {'recode': {}, 'text2cad': {}, 'benchcad': {}}

    if not args.skip_build:
        print(f'=== Building parquet shards in {staging} (max_code_len={args.max_code_len}) ===')
        # Clear old shards for configs we're rebuilding
        for cfg in args.configs:
            cfg_dir = staging / cfg
            if cfg_dir.exists():
                for p in cfg_dir.glob('*.parquet'):
                    p.unlink()

        if 'recode' in args.configs:
            for split in ('train', 'val'):
                split_dir = recode_root / split
                if not split_dir.exists():
                    continue
                print(f'recode/{split}: scanning {split_dir} ...')
                _, n = write_shards(
                    iter_recode(split_dir, args.max_code_len),
                    recode_schema, staging / 'recode', split,
                )
                counts['recode'][split] = n

        if 'text2cad' in args.configs:
            t2c_code_dir = t2c_root / 'cadquery'
            for split in ('train', 'val', 'test'):
                split_pkl = t2c_root / f'{split}.pkl'
                if not split_pkl.exists():
                    continue
                print(f'text2cad/{split}: loading {split_pkl} ...')
                _, n = write_shards(
                    iter_text2cad(split_pkl, t2c_code_dir, args.max_code_len),
                    t2c_schema, staging / 'text2cad', split,
                )
                counts['text2cad'][split] = n

        if 'benchcad' in args.configs:
            print(f'benchcad/train: loading BenchCAD/cad_bench from HF ...')
            _, n = write_shards(
                iter_benchcad(token, args.max_code_len),
                bench_schema, staging / 'benchcad', 'train',
            )
            counts['benchcad']['train'] = n
    else:
        for cfg in ('recode', 'text2cad', 'benchcad'):
            for split in ('train', 'val', 'test'):
                shards = sorted((staging / cfg).glob(f'{split}-*.parquet'))
                if not shards:
                    continue
                n = sum(pq.read_metadata(p).num_rows for p in shards)
                counts[cfg][split] = n
                print(f'  existing {cfg}/{split}: {n} rows in {len(shards)} shards')

    readme = build_readme(counts, args.max_code_len)
    (staging / 'README.md').write_text(readme)
    print('\nREADME written.')

    if args.skip_push:
        return

    print(f'\n=== Pushing to {args.repo} (private={args.private}) ===')
    api = HfApi(token=token)
    create_repo(args.repo, repo_type='dataset', private=args.private,
                exist_ok=True, token=token)
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type='dataset',
        folder_path=str(staging),
        private=args.private,
    )
    print(f'\nDone: https://huggingface.co/datasets/{args.repo}')


if __name__ == '__main__':
    main()
