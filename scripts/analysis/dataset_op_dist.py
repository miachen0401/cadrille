"""Compare op-presence frequency across BenchCAD + recode-20k + text2cad GT
+ a model's pred (from a diversity_analysis raw.jsonl).

Output: <out>/dataset_op_dist.md  one big table.

Usage:
    python -m scripts.analysis.dataset_op_dist \\
        --pred-raw eval_outputs/diversity_t3_ckpt4k_<ts>/raw.jsonl \\
        --pred-temp 1.00 \\
        --out eval_outputs/diversity_t3_ckpt4k_<ts>
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scripts.analysis.diversity_analysis import _OPS, detect_ops


def scan_dir(globpat: Path, label: str, sample: int | None = None) -> tuple[Counter, int]:
    files = sorted(globpat.parent.glob(globpat.name)) if globpat.name != '*' \
            else sorted(globpat.parent.glob('*.py'))
    if sample and len(files) > sample:
        rng = random.Random(42)
        rng.shuffle(files)
        files = files[:sample]
    counts: Counter = Counter()
    n = 0
    for p in files:
        try:
            code = p.read_text()
        except Exception:
            continue
        for op in detect_ops(code):
            counts[op] += 1
        n += 1
    print(f'  {label}: {n} files', flush=True)
    return counts, n


def benchcad_gt() -> tuple[Counter, int]:
    """benchcad train + val (all 20143)."""
    counts: Counter = Counter()
    total = 0
    for split in ('train', 'val'):
        d = Path('data/benchcad') / split
        for p in d.glob('*.py'):
            try:
                code = p.read_text()
            except Exception:
                continue
            for op in detect_ops(code):
                counts[op] += 1
            total += 1
    return counts, total


def recode20k_gt() -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for split in ('train', 'val'):
        d = Path('data/cad-recode-20k') / split
        for p in d.glob('*.py'):
            try:
                code = p.read_text()
            except Exception:
                continue
            for op in detect_ops(code):
                counts[op] += 1
            total += 1
    return counts, total


def text2cad_gt(sample: int = 3000) -> tuple[Counter, int]:
    """171k files — sample a fixed seed slice for speed."""
    d = Path('data/text2cad/cadquery')
    files = sorted(d.iterdir())
    files = [f for f in files if f.suffix == '.py']
    rng = random.Random(42)
    rng.shuffle(files)
    files = files[:sample]
    counts: Counter = Counter()
    n = 0
    for p in files:
        try:
            code = p.read_text()
        except Exception:
            continue
        for op in detect_ops(code):
            counts[op] += 1
        n += 1
    return counts, n


def pred_freq(raw_jsonl: Path, temp: str) -> tuple[Counter, int]:
    rows = [json.loads(l) for l in raw_jsonl.read_text().splitlines() if l.strip()]
    counts: Counter = Counter()
    for r in rows:
        seen: set[str] = set()
        for c in r['by_temp'].get(temp, []):
            seen |= detect_ops(c)
        for op in seen:
            counts[op] += 1
    return counts, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-raw',  type=Path, required=False, default=None)
    ap.add_argument('--pred-temp', type=str, default='1.00')
    ap.add_argument('--text2cad-sample', type=int, default=3000)
    ap.add_argument('--out',       type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print('scanning datasets ...', flush=True)
    bc_c,  bc_n  = benchcad_gt()
    print(f'  benchcad: {bc_n} files', flush=True)
    rc_c,  rc_n  = recode20k_gt()
    print(f'  recode20k: {rc_n} files', flush=True)
    t2_c,  t2_n  = text2cad_gt(sample=args.text2cad_sample)
    print(f'  text2cad: {t2_n} files (sampled from 171k)', flush=True)

    pred_c, pred_n = (Counter(), 0)
    pred_label = ''
    if args.pred_raw and args.pred_raw.exists():
        pred_c, pred_n = pred_freq(args.pred_raw, args.pred_temp)
        pred_label = f't={args.pred_temp}'
        print(f'  pred ({pred_label}): {pred_n} items', flush=True)

    md = ['# Op-presence frequency across SFT corpora + pred\n']
    md.append(f'Numbers are **% of items containing the op at least once**. '
              f'text2cad sampled {t2_n} of 171,177 (seed 42).\n')
    md.append('| op | benchcad ({:,}) | recode20k ({:,}) | text2cad sample ({:,}) | pred {} ({:,}) |'
              .format(bc_n, rc_n, t2_n, pred_label, pred_n))
    md.append('|---|---:|---:|---:|---:|')

    rows = []
    for op in _OPS:
        bc = bc_c[op] / bc_n * 100 if bc_n else 0
        rc = rc_c[op] / rc_n * 100 if rc_n else 0
        t2 = t2_c[op] / t2_n * 100 if t2_n else 0
        pd = pred_c[op] / pred_n * 100 if pred_n else 0
        line = f'| `{op}` | {bc:.1f} | {rc:.1f} | {t2:.1f} | {pd:.1f} |'
        md.append(line)
        rows.append((op, bc, rc, t2, pd))

    md.append('\n## Sorted by max(GT %) — biggest opportunity ops first\n')
    md.append('| op | benchcad | recode20k | text2cad | pred | max GT |')
    md.append('|---|---:|---:|---:|---:|---:|')
    for op, bc, rc, t2, pd in sorted(rows, key=lambda r: -max(r[1], r[2], r[3])):
        max_gt = max(bc, rc, t2)
        if max_gt < 1.0:
            continue
        md.append(f'| `{op}` | {bc:.1f} | {rc:.1f} | {t2:.1f} | **{pd:.1f}** | {max_gt:.1f} |')

    out_md = args.out / 'dataset_op_dist.md'
    out_md.write_text('\n'.join(md))
    (args.out / 'dataset_op_dist.json').write_text(json.dumps({
        'benchcad':  {op: bc_c[op] / bc_n  for op in _OPS} if bc_n  else {},
        'recode20k': {op: rc_c[op] / rc_n  for op in _OPS} if rc_n  else {},
        'text2cad':  {op: t2_c[op] / t2_n  for op in _OPS} if t2_n  else {},
        'pred':      {op: pred_c[op] / pred_n for op in _OPS} if pred_n else {},
        'totals': {'benchcad': bc_n, 'recode20k': rc_n, 'text2cad': t2_n, 'pred': pred_n},
    }, indent=2))
    print(f'\nSaved {out_md}', flush=True)


if __name__ == '__main__':
    main()
