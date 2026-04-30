"""Op-metric trajectory across training steps for one or more runs.

Computes per-eval-step (per-bucket):
  - op entropy (Shannon entropy of op presence rates)
  - rare op recall (gt rare ∩ pred rare / gt rare)
  - distinct ops (unique ops in pred set)

Sources:
  - greedy IoU log lines: rare_recall + distinct_ops are ALREADY in the
    training log (`[img/.../recall=...  rare_recall=...  IoU=... distinct_ops=N]`),
    so we extract them directly — no JSONL re-parse needed.
  - op entropy: NOT in log, computed from `predictions/step-NNNNNN.jsonl`.

Op taxonomy (regex patterns + rare set) is loaded from
`configs/eval/op_taxonomy.yaml` so adding/removing rare ops requires no
code change.

Usage:
    uv run python -m scripts.analysis.op_trajectory_metrics \\
        --runs v3:/home/ubuntu/cadrille/logs/v3_clean_20260428_132042.log \\
              v2:/home/ubuntu/cadrille/logs/big_bench_shell_50k_phase2b_20260427_184015.log \\
        --predictions-dir-by-run \\
            v3:/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions \\
        --buckets "BenchCAD val" "DeepCAD test" "Fusion360 test" \\
        --out docs/op_trajectory_2026-04-30.png \\
        [--post-discord]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─── Config loader ────────────────────────────────────────────────────────────

def _load_taxonomy(path: Path = REPO_ROOT / 'configs/eval/op_taxonomy.yaml') -> dict:
    raw = yaml.safe_load(path.read_text())
    patterns = {name: re.compile(p) for name, p in raw['patterns'].items()}
    rare = set(raw['rare'])
    feature = set(raw.get('feature', []))
    # Sanity: every rare op must exist in patterns
    unknown = rare - set(patterns)
    if unknown:
        raise SystemExit(f'rare set has ops not in patterns: {unknown}')
    return {'patterns': patterns, 'rare': rare, 'feature': feature}


def find_ops(code: str, patterns: dict) -> set[str]:
    """Set of op-names whose pattern matches anywhere in code."""
    if not code:
        return set()
    out = set()
    for name, pat in patterns.items():
        if pat.search(code):
            out.add(name)
    return out


# ─── Log scraper for greedy-eval trajectory metrics ──────────────────────────

# Match lines like:
#   [img/BenchCAD val] op_loss_w=0.150  recall=0.714  rare_recall=0.638
#                       IoU=0.643  exec=96.0%  distinct_ops=20  ...  (n=50)
_EVAL_LINE = re.compile(
    r'\[(\w+)/([^\]]+)\]\s+'
    r'(?:op_loss_w=(?P<oploss>[\d.]+)\s+)?'
    r'(?:recall=(?P<recall>[\d.]+)\s+)?'
    r'(?:rare_recall=(?P<rare>[\d.]+)\s+)?'
    r'(?:IoU=(?P<iou>[\d.]+)\s+)?'
    r'(?:exec=(?P<exec>[\d.]+)%\s+)?'
    r'distinct_ops=(?P<distinct>\d+)'
)
_STEP_LINE = re.compile(r'step=(\d+) running IoU eval')


def parse_log_trajectory(log_path: Path, buckets: list[str]) -> dict:
    """Return {bucket: [{step, iou, rare_recall, distinct_ops, exec_rate}, ...]}.
    Empty bucket entries skipped.
    """
    out = defaultdict(list)
    cur_step = None
    for line in log_path.read_text(errors='ignore').splitlines():
        m = _STEP_LINE.search(line)
        if m:
            cur_step = int(m.group(1))
            continue
        m = _EVAL_LINE.search(line)
        if m and cur_step is not None:
            modality, bucket = m.group(1), m.group(2)
            if bucket not in buckets:
                continue
            row = {'step': cur_step, 'modality': modality, 'bucket': bucket}
            for k in ('iou', 'rare', 'distinct', 'exec', 'recall', 'oploss'):
                v = m.group(k)
                if v is not None:
                    row[k] = float(v) if k != 'distinct' else int(v)
            out[bucket].append(row)
    return dict(out)


# ─── Predictions JSONL → op entropy ──────────────────────────────────────────

def compute_op_entropy(jsonl_path: Path, patterns: dict, bucket: str) -> dict:
    """For one (step, bucket), return entropy/distinct/avg_ops_per_code."""
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('bucket') != bucket:
                continue
            rows.append(r)
    if not rows:
        return {}
    op_counts = Counter()
    n_ops_per_code = []
    for r in rows:
        ops = find_ops(r.get('pred_code') or '', patterns)
        op_counts.update(ops)
        n_ops_per_code.append(len(ops))
    total = sum(op_counts.values())
    if total == 0:
        return {'op_entropy': 0.0, 'distinct_ops': 0,
                'avg_ops_per_code': 0.0, 'n_codes': len(rows)}
    p = np.array([c / total for c in op_counts.values()])
    H = float(-(p * np.log(p + 1e-12)).sum())
    return {
        'op_entropy': H,
        'distinct_ops': int(len(op_counts)),
        'avg_ops_per_code': float(np.mean(n_ops_per_code)),
        'n_codes': len(rows),
    }


def parse_predictions_trajectory(pred_dir: Path, buckets: list[str],
                                  patterns: dict) -> dict:
    """Return {bucket: [{step, op_entropy, distinct_ops, ...}, ...]}.
    Iterates step-NNNNNN.jsonl files; ignores .max@K.jsonl.
    """
    out = defaultdict(list)
    pat = re.compile(r'^step-(\d{6})\.jsonl$')
    files = sorted(pred_dir.glob('step-*.jsonl'))
    for p in files:
        m = pat.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        for bucket in buckets:
            metrics = compute_op_entropy(p, patterns, bucket)
            if metrics:
                metrics.update(step=step, bucket=bucket)
                out[bucket].append(metrics)
    return dict(out)


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_trajectory(runs: dict[str, dict], pred_runs: dict[str, dict],
                    buckets: list[str], out_path: Path) -> None:
    """6-panel grid: (rare_recall, op_entropy, distinct_ops) × (BC, DC, Fu)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cmap = {'v3': 'C0', 'v2': 'C1', 'v2b': 'C1', 'curriculum': 'C2',
            'v4': 'C3', 'base': 'C5'}

    fig, axes = plt.subplots(3, len(buckets), figsize=(5 * len(buckets), 11),
                              sharex=True)
    if len(buckets) == 1:
        axes = axes.reshape(-1, 1)

    metric_specs = [
        ('rare', 'rare op recall', (0.0, 1.05)),
        ('op_entropy', 'op entropy (nats)', None),
        ('distinct', 'distinct ops', None),
    ]

    for col, bucket in enumerate(buckets):
        for row, (key, ylabel, ylim) in enumerate(metric_specs):
            ax = axes[row, col]
            for run_name, log_traj in runs.items():
                color = cmap.get(run_name.split('-')[0], f'C{hash(run_name) % 9}')
                if key in ('rare', 'distinct'):
                    series = log_traj.get(bucket, [])
                    xs = [r['step'] for r in series if key in r]
                    ys = [r[key] for r in series if key in r]
                else:  # op_entropy from predictions
                    series = pred_runs.get(run_name, {}).get(bucket, [])
                    xs = [r['step'] for r in series]
                    ys = [r['op_entropy'] for r in series]
                if xs:
                    ax.plot(xs, ys, '-o', label=run_name, color=color,
                            markersize=3, linewidth=1.5, alpha=0.8)
            if row == 0:
                ax.set_title(f'[{bucket}]', fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == len(metric_specs) - 1:
                ax.set_xlabel('training step', fontsize=10)
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)
            # Per-panel legend (was previously only on top-right which sometimes
            # had empty data → "no artists with labels" warning).
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='best', fontsize=8)

    fig.suptitle('Op-metric trajectory across training', fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='List of run_name:log_path entries')
    ap.add_argument('--predictions-dir-by-run', nargs='*', default=[],
                    help='List of run_name:predictions_dir entries '
                         '(only used for op_entropy)')
    ap.add_argument('--buckets', nargs='+',
                    default=['BenchCAD val', 'DeepCAD test', 'Fusion360 test'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--post-discord', action='store_true')
    args = ap.parse_args()

    runs = {}
    for entry in args.runs:
        name, path = entry.split(':', 1)
        runs[name] = parse_log_trajectory(Path(path), args.buckets)
        for bucket, series in runs[name].items():
            print(f'  [{name}] {bucket}: {len(series)} eval points')

    tax = _load_taxonomy()
    pred_runs = {}
    for entry in args.predictions_dir_by_run:
        name, pred_dir = entry.split(':', 1)
        print(f'  [{name}] computing op_entropy from {pred_dir} ...')
        pred_runs[name] = parse_predictions_trajectory(
            Path(pred_dir), args.buckets, tax['patterns'])
        for bucket, series in pred_runs[name].items():
            print(f'    {bucket}: {len(series)} steps')

    plot_trajectory(runs, pred_runs, args.buckets, args.out)
    print(f'Wrote {args.out} ({args.out.stat().st_size // 1024} KB)')

    if args.post_discord:
        import os
        import subprocess
        # Read DISCORD_WEBHOOK_URL from env or .bashrc
        webhook = os.environ.get('DISCORD_WEBHOOK_URL')
        if not webhook:
            try:
                webhook = subprocess.check_output(
                    "grep '^export DISCORD_WEBHOOK_URL' ~/.bashrc | "
                    "sed 's/.*=\"\\?//; s/\"$//'", shell=True, text=True).strip()
            except Exception:
                webhook = None
        if not webhook:
            print('DISCORD_WEBHOOK_URL not set; skipping post')
            return
        import requests
        n_runs = len(runs)
        content = (f'**Op-metric trajectory** ({n_runs} run{"s" if n_runs > 1 else ""}, '
                   f'{len(args.buckets)} buckets) — rare_recall, op_entropy, '
                   f'distinct_ops vs training step.')
        with args.out.open('rb') as f:
            r = requests.post(webhook, data={'payload_json':
                json.dumps({'content': content})},
                files={'files[0]': (args.out.name, f.read())}, timeout=60)
        print(f'Discord HTTP {r.status_code}')


if __name__ == '__main__':
    main()
