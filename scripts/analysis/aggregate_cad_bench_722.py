"""Aggregate per-model metadata.jsonl from a cad_bench_722 eval run into one
summary, and optionally ping it to Discord.

Each model dir under --root must contain `metadata.jsonl` (one record per sample).
We compute:
  - n (samples scored)
  - exec_rate (fraction with error_type == 'success')
  - mean_iou (over successful samples)
  - mean_cd  (over successful samples)
  - per-difficulty exec_rate + mean_iou (easy / medium / hard)

Usage:
    python scripts/analysis/aggregate_cad_bench_722.py \
        --root eval_outputs/cad_bench_722 \
        --out  eval_outputs/cad_bench_722/summary.json \
        --discord
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path


def _summarize(meta_path: Path) -> dict:
    recs = []
    with open(meta_path) as f:
        for line in f:
            try: recs.append(json.loads(line))
            except Exception: pass
    if not recs:
        return {'n': 0}
    ok   = [r for r in recs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    cds  = [r['cd']  for r in ok if r.get('cd')  is not None]
    by_diff = defaultdict(list)
    for r in recs: by_diff[r.get('difficulty', '?')].append(r)
    diff_stats = {}
    for diff, rs in by_diff.items():
        ok_d  = [x for x in rs if x.get('error_type') == 'success']
        iou_d = [x['iou'] for x in ok_d if x.get('iou') is not None]
        diff_stats[diff] = {
            'n':         len(rs),
            'exec_rate': round(len(ok_d) / len(rs), 4) if rs else 0.0,
            'mean_iou':  round(sum(iou_d) / len(iou_d), 4) if iou_d else None,
        }
    return {
        'n':         len(recs),
        'exec_rate': round(len(ok) / len(recs), 4) if recs else 0.0,
        'mean_iou':  round(sum(ious) / len(ious), 4) if ious else None,
        'mean_cd':   round(sum(cds)  / len(cds),  6) if cds  else None,
        'by_difficulty': diff_stats,
    }


def _format_discord(summary: dict, dataset: str) -> str:
    lines = [f'📊 **cad_bench_722 eval results** ({dataset}, n=720)\n']
    lines.append('```')
    lines.append(f'{"model":<22} {"n":>4} {"exec":>6} {"IoU":>7} {"CD":>10}')
    lines.append('-' * 55)
    for model, s in summary['models'].items():
        n   = s.get('n', 0)
        ex  = f'{s["exec_rate"]*100:>5.1f}%' if s.get('exec_rate') is not None else '   —'
        iou = f'{s["mean_iou"]:>7.4f}'      if s.get('mean_iou')  is not None else '      —'
        cd  = f'{s["mean_cd"]:>10.6f}'      if s.get('mean_cd')   is not None else '         —'
        lines.append(f'{model:<22} {n:>4} {ex:>6} {iou} {cd}')
    lines.append('```\n')
    # Per-difficulty
    diffs = ['easy', 'medium', 'hard']
    lines.append('**Per-difficulty exec rate / mean IoU:**')
    lines.append('```')
    header = f'{"model":<22}' + ''.join(f'{d:>14}' for d in diffs)
    lines.append(header)
    lines.append('-' * len(header))
    for model, s in summary['models'].items():
        row = [f'{model:<22}']
        for d in diffs:
            ds = s.get('by_difficulty', {}).get(d, {})
            if not ds:
                row.append(f'{"—":>14}')
                continue
            ex  = f'{ds["exec_rate"]*100:.0f}%'
            iou = f'{ds["mean_iou"]:.3f}' if ds.get('mean_iou') is not None else '—'
            row.append(f'{ex+"/"+iou:>14}')
        lines.append(''.join(row))
    lines.append('```')
    return '\n'.join(lines)


def _ping_discord(text: str) -> None:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  (no DISCORD_WEBHOOK_URL — skipping ping)', flush=True)
        return
    # Discord max message length is 2000 chars; chunk if needed.
    chunks = [text[i:i+1900] for i in range(0, len(text), 1900)]
    for c in chunks:
        data = json.dumps({'content': c}).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={'Content-Type': 'application/json',
                     'User-Agent': 'cad-bench-722-aggregator/1.0'})
        try: urllib.request.urlopen(req, timeout=10).read()
        except Exception as e: print(f'  discord ping failed: {e}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='Root dir containing per-model subdirs with metadata.jsonl')
    ap.add_argument('--out', required=True,
                    help='Output summary.json path')
    ap.add_argument('--dataset', default='BenchCAD/cad_bench_722')
    ap.add_argument('--discord', action='store_true',
                    help='Ping DISCORD_WEBHOOK_URL with the formatted summary')
    args = ap.parse_args()

    root = Path(args.root)
    summary = {'dataset': args.dataset, 'models': {}}
    for d in sorted(p for p in root.iterdir()
                    if p.is_dir() and not p.name.startswith('_')):
        meta = d / 'metadata.jsonl'
        if not meta.exists():
            continue
        summary['models'][d.name] = _summarize(meta)

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f'Wrote {args.out}', flush=True)

    formatted = _format_discord(summary, args.dataset)
    print('\n' + formatted, flush=True)

    if args.discord:
        _ping_discord(formatted)


if __name__ == '__main__':
    main()
