"""Compare two model eval runs side by side.

Reads the per-sample IoU/CD CSVs from two evaluate.py runs and the static
failure analysis records from each eval dir, then generates comparison plots.

Usage:
    python viz/compare_evals.py \
        --eval-a work_dirs/eval_hf_baseline   --csv-a work_dirs/results_hf_baseline.csv \
        --eval-b work_dirs/eval_gbmgrb95_mini --csv-b work_dirs/results_gbmgrb95_mini.csv \
        --label-a "HF Baseline" --label-b "Our SFT/RL" \
        --out-dir viz/plots/compare
"""

import os
import sys
import csv
import re
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.parse_cq import parse_cq_script

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print('ERROR: matplotlib not installed.')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {path}')


def load_eval_dir(eval_dir: str):
    """Parse all .py files in eval_dir, return list of feature dicts."""
    from pathlib import Path
    records = []
    for path in sorted(Path(eval_dir).glob('*.py')):
        try:
            code = path.read_text(encoding='utf-8', errors='replace')
            feat = parse_cq_script(code)
            feat['stem'] = path.stem
            feat['code'] = code
            records.append(feat)
        except Exception:
            pass
    return records


def load_csv(csv_path: str):
    """Load evaluate.py results CSV → dict stem → {iou, cd}."""
    results = {}
    if not csv_path or not os.path.exists(csv_path):
        return results
    try:
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                key  = row.get('file_name', '')
                base = key.split('+')[0] if '+' in key else key
                iou  = float(row['iou']) if row.get('iou') not in ('', 'None', None) else None
                cd   = float(row['cd'])  if row.get('cd')  not in ('', 'None', None) else None
                if base not in results or (iou is not None and
                        (results[base]['iou'] is None or iou > results[base]['iou'])):
                    results[base] = {'iou': iou, 'cd': cd}
    except Exception as e:
        print(f'  warning: could not load {csv_path}: {e}')
    return results


# ---------------------------------------------------------------------------
# Plot 1: IoU distribution comparison
# ---------------------------------------------------------------------------

def plot_iou_comparison(ious_a, ious_b, label_a, label_b, out_dir):
    if not ious_a and not ious_b:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(ious_a, bins=bins, alpha=0.65, color='steelblue',
            density=True, label=f'{label_a}  mean={np.mean(ious_a):.3f}  n={len(ious_a)}')
    ax.hist(ious_b, bins=bins, alpha=0.65, color='darkorange',
            density=True, label=f'{label_b}  mean={np.mean(ious_b):.3f}  n={len(ious_b)}')
    for vals, color in [(ious_a, 'steelblue'), (ious_b, 'darkorange')]:
        ax.axvline(np.mean(vals), color=color, linestyle='--', linewidth=1.5)
    ax.set_xlabel('IoU')
    ax.set_ylabel('Density')
    ax.set_title('IoU distribution comparison')
    ax.legend(fontsize=8)

    # Cumulative
    ax = axes[1]
    for vals, color, label in [(ious_a, 'steelblue', label_a), (ious_b, 'darkorange', label_b)]:
        sorted_v = np.sort(vals)
        ax.plot(sorted_v, np.linspace(0, 1, len(sorted_v)),
                color=color, linewidth=2, label=f'{label}')
    ax.set_xlabel('IoU threshold')
    ax.set_ylabel('Fraction of samples ≤ threshold')
    ax.set_title('Cumulative IoU distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '01_iou_comparison.png'))


# ---------------------------------------------------------------------------
# Plot 2: CD distribution comparison
# ---------------------------------------------------------------------------

def plot_cd_comparison(cds_a, cds_b, label_a, label_b, out_dir):
    if not cds_a and not cds_b:
        return

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    cds_a_ms = [c * 1000 for c in cds_a]
    cds_b_ms = [c * 1000 for c in cds_b]
    cap = np.percentile(cds_a_ms + cds_b_ms, 95) * 1.1
    bins = np.linspace(0, cap, 40)
    ax.hist(cds_a_ms, bins=bins, alpha=0.65, color='steelblue', density=True,
            label=f'{label_a}  median={np.median(cds_a_ms):.3f}  n={len(cds_a_ms)}')
    ax.hist(cds_b_ms, bins=bins, alpha=0.65, color='darkorange', density=True,
            label=f'{label_b}  median={np.median(cds_b_ms):.3f}  n={len(cds_b_ms)}')
    for vals, color in [(cds_a_ms, 'steelblue'), (cds_b_ms, 'darkorange')]:
        ax.axvline(np.median(vals), color=color, linestyle='--', linewidth=1.5)
    ax.set_xlabel('Chamfer Distance (×10³)')
    ax.set_ylabel('Density')
    ax.set_title('Chamfer Distance comparison')
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '02_cd_comparison.png'))


# ---------------------------------------------------------------------------
# Plot 3: Metric summary bar (means / medians side by side)
# ---------------------------------------------------------------------------

def plot_metric_summary(ious_a, cds_a, n_total_a,
                        ious_b, cds_b, n_total_b,
                        label_a, label_b, out_dir):
    metrics = {
        'IoU mean':         (np.mean(ious_a) if ious_a else 0, np.mean(ious_b) if ious_b else 0),
        'IoU median':       (np.median(ious_a) if ious_a else 0, np.median(ious_b) if ious_b else 0),
        'CD median (×10³)': (np.median(cds_a)*1000 if cds_a else 0, np.median(cds_b)*1000 if cds_b else 0),
        'Failure rate':     (1 - len(ious_a)/n_total_a if n_total_a else 0,
                             1 - len(ious_b)/n_total_b if n_total_b else 0),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4), facecolor='white')
    for ax, (name, (va, vb)) in zip(axes, metrics.items()):
        higher_is_better = ('CD' not in name and 'Failure' not in name)
        winner_a = (va >= vb) if higher_is_better else (va <= vb)
        ca = '#4CAF50' if winner_a else '#EF5350'
        cb = '#4CAF50' if not winner_a else '#EF5350'
        ax.bar([label_a, label_b], [va, vb], color=[ca, cb], edgecolor='white', width=0.5)
        for i, v in enumerate([va, vb]):
            ax.text(i, v + max(va, vb) * 0.02, f'{v:.3f}', ha='center', fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, max(va, vb) * 1.25 + 1e-6)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Model comparison summary', fontsize=12)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '03_metric_summary.png'))


# ---------------------------------------------------------------------------
# Plot 4: IoU by operation type — both models
# ---------------------------------------------------------------------------

def plot_iou_by_op_compare(recs_a, recs_b, label_a, label_b, out_dir):
    flag_keys = [
        ('has_arc',           'arc'),
        ('has_multi_body',    'multi-body'),
        ('has_cylinder',      'cylinder'),
        ('has_box',           'box'),
        ('has_push',          'push'),
        ('has_subtract_mode', 'sub-mode'),
        ('has_revolve',       'revolve'),
        ('has_fillet',        'fillet'),
    ]

    group_labels, means_a, means_b = ['all'], [], []
    all_iou_a = [r['iou'] for r in recs_a if r.get('iou') is not None]
    all_iou_b = [r['iou'] for r in recs_b if r.get('iou') is not None]
    means_a.append(np.mean(all_iou_a) if all_iou_a else 0)
    means_b.append(np.mean(all_iou_b) if all_iou_b else 0)

    for feat_key, label in flag_keys:
        sub_a = [r['iou'] for r in recs_a if r.get(feat_key) and r.get('iou') is not None]
        sub_b = [r['iou'] for r in recs_b if r.get(feat_key) and r.get('iou') is not None]
        if len(sub_a) + len(sub_b) < 4:
            continue
        group_labels.append(f'{label}\na={len(sub_a)}, b={len(sub_b)}')
        means_a.append(np.mean(sub_a) if sub_a else 0)
        means_b.append(np.mean(sub_b) if sub_b else 0)

    x = np.arange(len(group_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(x) * 1.3), 5), facecolor='white')
    ax.bar(x - width/2, means_a, width, label=label_a, color='steelblue', edgecolor='white')
    ax.bar(x + width/2, means_b, width, label=label_b, color='darkorange', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel('Mean IoU')
    ax.set_ylim(0, 1.05)
    ax.set_title('Mean IoU by operation type — model comparison')
    ax.legend()
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '04_iou_by_op_compare.png'))


# ---------------------------------------------------------------------------
# Plot 5: Operation usage — both models (distribution shift)
# ---------------------------------------------------------------------------

def plot_op_usage_compare(recs_a, recs_b, label_a, label_b, out_dir):
    op_keys = [
        ('segment', 'n_segments'), ('arc', 'n_arcs'), ('circle', 'n_circles'),
        ('rect', 'n_rects'), ('push', 'n_push'), ('subtract_mode', 'n_subtract_mode'),
        ('union', 'n_unions'), ('cut', 'n_cuts'), ('cylinder', 'n_cylinders'),
        ('box', 'n_boxes'), ('revolve', 'n_revolves'), ('fillet', 'n_fillets'),
        ('polygon', 'n_polygons'), ('spline', 'n_splines'),
    ]
    na, nb = len(recs_a), len(recs_b)
    labels, rates_a, rates_b = [], [], []
    for label, key in op_keys:
        ra = sum(1 for r in recs_a if r.get(key, 0) > 0) / na if na else 0
        rb = sum(1 for r in recs_b if r.get(key, 0) > 0) / nb if nb else 0
        labels.append(label)
        rates_a.append(ra)
        rates_b.append(rb)

    order = np.argsort([ra + rb for ra, rb in zip(rates_a, rates_b)])[::-1]
    labels  = [labels[i]  for i in order]
    rates_a = [rates_a[i] for i in order]
    rates_b = [rates_b[i] for i in order]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.bar(x - width/2, rates_a, width, label=label_a, color='steelblue', edgecolor='white')
    ax.bar(x + width/2, rates_b, width, label=label_b, color='darkorange', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel('Fraction of scripts with this op')
    ax.set_title('Operation usage: model A vs model B')
    ax.legend()
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '05_op_usage_compare.png'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare two eval runs side by side')
    parser.add_argument('--eval-a',  required=True, help='Eval dir for model A')
    parser.add_argument('--eval-b',  required=True, help='Eval dir for model B')
    parser.add_argument('--csv-a',   default=None,  help='Results CSV for model A')
    parser.add_argument('--csv-b',   default=None,  help='Results CSV for model B')
    parser.add_argument('--label-a', default='Model A', help='Display name for model A')
    parser.add_argument('--label-b', default='Model B', help='Display name for model B')
    parser.add_argument('--out-dir', default='viz/plots/compare', help='Output dir')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading {args.label_a} from {args.eval_a} ...')
    recs_a = load_eval_dir(args.eval_a)
    print(f'  {len(recs_a)} scripts')

    print(f'Loading {args.label_b} from {args.eval_b} ...')
    recs_b = load_eval_dir(args.eval_b)
    print(f'  {len(recs_b)} scripts')

    map_a = load_csv(args.csv_a)
    map_b = load_csv(args.csv_b)

    # Merge IoU/CD into records
    for rec in recs_a:
        base = rec['stem'].split('+')[0]
        hit  = map_a.get(rec['stem']) or map_a.get(base)
        rec['iou'] = hit['iou'] if hit else None
        rec['cd']  = hit['cd']  if hit else None

    for rec in recs_b:
        base = rec['stem'].split('+')[0]
        hit  = map_b.get(rec['stem']) or map_b.get(base)
        rec['iou'] = hit['iou'] if hit else None
        rec['cd']  = hit['cd']  if hit else None

    ious_a = [r['iou'] for r in recs_a if r.get('iou') is not None]
    ious_b = [r['iou'] for r in recs_b if r.get('iou') is not None]
    cds_a  = [r['cd']  for r in recs_a if r.get('cd')  is not None]
    cds_b  = [r['cd']  for r in recs_b if r.get('cd')  is not None]

    print(f'\n{args.label_a}: IoU mean={np.mean(ious_a):.3f}  CD median={np.median(cds_a)*1000:.3f}×10³  n_iou={len(ious_a)}/{len(recs_a)}')
    print(f'{args.label_b}: IoU mean={np.mean(ious_b):.3f}  CD median={np.median(cds_b)*1000:.3f}×10³  n_iou={len(ious_b)}/{len(recs_b)}')

    print(f'\nGenerating comparison plots → {args.out_dir}/')
    plot_iou_comparison(ious_a, ious_b, args.label_a, args.label_b, args.out_dir)
    plot_cd_comparison(cds_a, cds_b, args.label_a, args.label_b, args.out_dir)
    plot_metric_summary(ious_a, cds_a, len(recs_a), ious_b, cds_b, len(recs_b),
                        args.label_a, args.label_b, args.out_dir)
    plot_iou_by_op_compare(recs_a, recs_b, args.label_a, args.label_b, args.out_dir)
    plot_op_usage_compare(recs_a, recs_b, args.label_a, args.label_b, args.out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
