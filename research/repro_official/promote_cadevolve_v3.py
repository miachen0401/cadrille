"""Promote CADEvolve v3 outputs to canonical eval_outputs/.../cadevolve_rl1/.

After `run_cadevolve_v2_all.sh` finishes, this script:
  1. Copies/moves v3 .py preds and metadata.jsonl from
     eval_outputs/repro_official/{deepcad_n300,fusion360_n300,cad_bench_722_full}/cadevolve_v3/
     → eval_outputs/{deepcad_n300,fusion360_n300,cad_bench_722}/cadevolve_rl1/
  2. For cad_bench_722: enriches metadata.jsonl with family / difficulty /
     base_plane / split / feature_tags / feature_count fields by joining
     against the BenchCAD/cad_bench_722 HF dataset on `stem`. Downstream
     scoring (essential_ops, build_full_grid, etc.) expects these fields.

Usage:
    set -a; source .env; set +a
    .venv/bin/python research/repro_official/promote_cadevolve_v3.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# (v3_src_dir, canonical_dst_dir, needs_benchcad_metadata_enrichment?)
PAIRS = [
    (REPO / 'eval_outputs/repro_official/deepcad_n300/cadevolve_v3',
     REPO / 'eval_outputs/deepcad_n300/cadevolve_rl1',
     False),
    (REPO / 'eval_outputs/repro_official/fusion360_n300/cadevolve_v3',
     REPO / 'eval_outputs/fusion360_n300/cadevolve_rl1',
     False),
    (REPO / 'eval_outputs/repro_official/cad_bench_722_full/cadevolve_v3',
     REPO / 'eval_outputs/cad_bench_722/cadevolve_rl1',
     True),
]


def enrich_with_benchcad(rs):
    """Add family, difficulty, base_plane, split, feature_tags, feature_count
    to each record by joining on `stem` against BenchCAD/cad_bench_722."""
    print('  loading BenchCAD/cad_bench_722 …', flush=True)
    from datasets import load_dataset
    ds = load_dataset('BenchCAD/cad_bench_722', split='train',
                      token=os.environ.get('HF_TOKEN'))
    by_stem = {row['stem']: row for row in ds}
    KEYS = ('family', 'difficulty', 'base_plane', 'split',
            'feature_tags', 'feature_count')
    n_enriched = 0
    for r in rs:
        row = by_stem.get(r.get('stem'))
        if not row:
            continue
        for k in KEYS:
            if k in row and r.get(k) is None:
                r[k] = row[k]
        n_enriched += 1
    print(f'  enriched {n_enriched}/{len(rs)} records', flush=True)
    return rs


def main():
    for src, dst, enrich in PAIRS:
        if not src.exists():
            print(f'! missing source: {src}'); continue
        meta_src = src / 'metadata.jsonl'
        if not meta_src.exists():
            print(f'! no metadata.jsonl in {src}'); continue

        print(f'\n=== {src.name} → {dst} ===')
        dst.mkdir(parents=True, exist_ok=True)

        # Copy .py preds (overwrite). Skip metadata.jsonl — handled separately.
        n_copied = 0
        for p in src.iterdir():
            if p.name == 'metadata.jsonl' or not p.suffix == '.py':
                continue
            shutil.copy2(p, dst / p.name)
            n_copied += 1
        print(f'  copied {n_copied} .py files')

        # Enrich + write metadata
        rs = [json.loads(l) for l in open(meta_src)]
        if enrich:
            rs = enrich_with_benchcad(rs)
        with open(dst / 'metadata.jsonl', 'w') as f:
            for r in rs:
                f.write(json.dumps(r) + '\n')
        print(f'  wrote metadata.jsonl with {len(rs)} records')


if __name__ == '__main__':
    main()
