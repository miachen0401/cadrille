"""Compare op frequency: BenchCAD GT (full train+val) vs current pred raw.jsonl.

Step 1 of diversity told us "30 GT items had `hole` 14, pred 0". That's a tiny
slice. This widens it to all 20k benchcad items so we can see the true GT
distribution and report pred recall against it.

Output:
  <out>/benchcad_op_freq.md  — markdown table of per-op frequencies
  <out>/benchcad_op_freq.json

Usage:
    python -m scripts.analysis.diversity_benchcad_compare \\
        --pred-raw eval_outputs/diversity_t3_ckpt4k_<ts>/raw.jsonl \\
        --out eval_outputs/diversity_t3_ckpt4k_<ts>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Reuse the same op set + detector for an apples-to-apples comparison.
from scripts.analysis.diversity_analysis import _OPS, detect_ops


def gt_corpus_freq() -> tuple[Counter, int]:
    """Return (op → items containing op, total items) over benchcad train+val."""
    counts: Counter = Counter()
    total = 0
    for split in ('train', 'val'):
        split_dir = Path('data/benchcad') / split
        for py in sorted(split_dir.glob('*.py')):
            code = py.read_text()
            for op in detect_ops(code):
                counts[op] += 1
            total += 1
    return counts, total


def pred_corpus_freq(raw_jsonl: Path) -> dict[str, tuple[Counter, int]]:
    """Return {temp: (op → items, total items)} from a diversity raw.jsonl."""
    out: dict[str, tuple[Counter, int]] = {}
    rows = [json.loads(l) for l in raw_jsonl.read_text().splitlines() if l.strip()]
    if not rows:
        return out
    temps = list(rows[0]['by_temp'].keys())
    for t in temps:
        counts: Counter = Counter()
        for r in rows:
            seen: set[str] = set()
            for c in r['by_temp'].get(t, []):
                seen |= detect_ops(c)
            for op in seen:
                counts[op] += 1
        out[t] = (counts, len(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-raw', required=True, type=Path)
    ap.add_argument('--out',      required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print('scanning benchcad GT corpus ...', flush=True)
    gt_counts, gt_total = gt_corpus_freq()
    print(f'  GT total items: {gt_total}', flush=True)
    print(f'  loading pred raw {args.pred_raw} ...', flush=True)
    pred = pred_corpus_freq(args.pred_raw)
    pred_temps = list(pred.keys())
    pred_total = pred[pred_temps[0]][1] if pred_temps else 0
    print(f'  pred total items: {pred_total} (per temp); temps: {pred_temps}', flush=True)

    md = [f'# BenchCAD op-frequency: GT corpus ({gt_total} items) vs pred ({pred_total} items)\n']
    md.append('Frequencies are **% of items containing the op at least once**.\n')

    md.append('| op | GT % | ' + ' | '.join(f't={t} %' for t in pred_temps)
              + ' | recall@max%(t≤1) |')
    md.append('|---|---:|' + '---:|' * len(pred_temps) + '---:|')

    rows_data = []
    for op in _OPS:
        gt_pct = gt_counts[op] / gt_total * 100 if gt_total else 0
        line = f'| `{op}` | {gt_pct:.1f} |'
        per_temp = []
        for t in pred_temps:
            c, n = pred[t]
            pct = c[op] / n * 100 if n else 0
            line += f' {pct:.1f} |'
            per_temp.append(pct)
        lenient = [t for t in pred_temps if float(t) <= 1.0]
        best_pct = max((pred[t][0][op] / pred[t][1] * 100 if pred[t][1] else 0)
                       for t in lenient) if lenient else 0
        recall = (best_pct / gt_pct) if gt_pct > 0 else 0
        line += f' {recall*100:.0f}% |'
        md.append(line)
        rows_data.append((op, gt_pct, per_temp, recall, best_pct))

    md.append('\n## Sorted by GT frequency (descending) — gap analysis\n')
    md.append('| op | GT % | best pred % (t≤1) | recall | gap |')
    md.append('|---|---:|---:|---:|---|')
    for op, gt_pct, per_temp, recall, best in sorted(rows_data, key=lambda x: -x[1]):
        if gt_pct < 0.1:
            continue
        if best == 0 and gt_pct > 1:
            tag = '— (zero learning)'
        elif recall < 0.3:
            tag = '⚠️ severe under'
        elif recall > 0.7:
            tag = '✓'
        else:
            tag = 'partial'
        md.append(f'| `{op}` | {gt_pct:.1f} | {best:.1f} | {recall*100:.0f}% | {tag} |')

    (args.out / 'benchcad_op_freq.md').write_text('\n'.join(md))
    (args.out / 'benchcad_op_freq.json').write_text(json.dumps({
        'gt':       {op: c / gt_total for op, c in gt_counts.items()},
        'gt_total': gt_total,
        'pred':     {t: {op: c / n for op, c in pred[t][0].items()}
                     for t in pred_temps for c, n in [pred[t]]},
    }, indent=2))
    print(f'\nSaved to {args.out}/benchcad_op_freq.md', flush=True)


if __name__ == '__main__':
    main()
