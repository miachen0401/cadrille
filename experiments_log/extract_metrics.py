"""Parse a cadrille SFT training log → eval_metrics.csv.

Each `[online-eval] step=N running IoU eval` block contains 5 lines, one per
bucket. We extract op_loss_w / recall / rare_recall / IoU per bucket per step.

Output CSV columns:
    step, bucket, modality, op_loss_w, recall, rare_recall, iou, exec_pct,
    distinct_ops, distinct_codes, n

Usage:
    uv run python -m experiments_log.extract_metrics LOG_PATH > out.csv
or
    uv run python experiments_log/extract_metrics.py LOG_PATH OUT_CSV
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

BUCKETS = ('BenchCAD val', 'recode20k train', 'text2cad train',
           'DeepCAD test', 'Fusion360 test')

LINE_RE = re.compile(
    r'\[(?P<mod>img|text)/(?P<bucket>'
    + '|'.join(re.escape(b) for b in BUCKETS) + r')\]\s+'
    r'(?:op_loss_w=(?P<op>-?[\d.]+)\s+)?'
    r'(?:recall=(?P<rec>[\d.]+)\s+)?'
    r'(?:rare_recall=(?P<rare>[\d.]+)\s+)?'
    r'(?:IoU=(?P<iou>[\d.]+)\s+)?'
    r'(?:exec=(?P<exec>[\d.]+)%\s+)?'
    r'(?:distinct_ops=(?P<dops>\d+)\s+)?'
    r'(?:distinct_codes=(?P<dcod>[\d.]+)\s+)?'
    r'(?:\(n=(?P<n>\d+)\))?'
)
STEP_RE = re.compile(r'\[online-eval\] step=(\d+) running IoU eval')


def extract(log_path: Path) -> list[dict]:
    text = log_path.read_text()
    out = []
    cur_step = None
    for line in text.splitlines():
        m_step = STEP_RE.search(line)
        if m_step:
            cur_step = int(m_step.group(1))
            continue
        if cur_step is None:
            continue
        m = LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        out.append({
            'step': cur_step,
            'bucket': d['bucket'],
            'modality': d['mod'],
            'op_loss_w': d['op'],
            'recall': d['rec'],
            'rare_recall': d['rare'],
            'iou': d['iou'],
            'exec_pct': d['exec'],
            'distinct_ops': d['dops'],
            'distinct_codes': d['dcod'],
            'n': d['n'],
        })
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    log = Path(sys.argv[1])
    rows = extract(log)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    fields = ['step', 'bucket', 'modality', 'op_loss_w', 'recall',
              'rare_recall', 'iou', 'exec_pct', 'distinct_ops',
              'distinct_codes', 'n']
    if out_path is None:
        w = csv.DictWriter(sys.stdout, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f'wrote {len(rows)} rows → {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
