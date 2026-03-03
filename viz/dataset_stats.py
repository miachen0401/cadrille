"""Training data distribution analysis.

Loads all CadQuery .py files from the training batches, extracts structural
features, and generates plots showing what operations/complexity the model
has been trained on.

Usage:
    python viz/dataset_stats.py
    python viz/dataset_stats.py --data-dir ./data/cad-recode-v1.5/train
    python viz/dataset_stats.py --data-dir ./data/cad-recode-v1.5/train --out-dir viz/plots/dataset_stats
"""

import os
import sys
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz.parse_cq import load_cq_dir

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print('ERROR: matplotlib not installed. Run: pip install matplotlib')
    sys.exit(1)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

STYLE = dict(facecolor='#f8f8f8')

def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {path}')


def _pct_label(ax, bars, total):
    """Annotate each bar with its percentage of total."""
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + total * 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{w / total * 100:.1f}%', va='center', fontsize=8)


# ---------------------------------------------------------------------------
# Plot 1: Operation frequency
# ---------------------------------------------------------------------------

def plot_operation_frequency(records: list, out_dir: str, split_name: str):
    """Horizontal bar chart: how often each CadQuery operation appears."""
    op_keys = [
        ('segment',       'n_segments'),
        ('arc',           'n_arcs'),
        ('circle',        'n_circles'),
        ('rect',          'n_rects'),
        ('polygon',       'n_polygons'),
        ('spline',        'n_splines'),
        ('ellipse',       'n_ellipses'),
        ('extrude',       'n_extrudes'),
        ('revolve',       'n_revolves'),
        ('loft',          'n_lofts'),
        ('sweep',         'n_sweeps'),
        ('union',         'n_unions'),
        ('cut/subtract',  'n_cuts'),
        ('intersect',     'n_intersects'),
        ('cylinder',      'n_cylinders'),
        ('box',           'n_boxes'),
        ('sphere',        'n_spheres'),
        ('fillet',        'n_fillets'),
        ('chamfer',       'n_chamfers'),
        ('shell',         'n_shells'),
        ('push (multi-region)', 'n_push'),
        ('subtract mode', 'n_subtract_mode'),
    ]

    total_scripts = len(records)
    # For each op: count how many scripts contain at least one instance
    counts = []
    for label, key in op_keys:
        n = sum(1 for r in records if r.get(key, 0) > 0)
        counts.append((label, n))

    counts.sort(key=lambda x: x[1])
    labels, vals = zip(*counts)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    bars = ax.barh(labels, vals, color='steelblue', edgecolor='white')
    _pct_label(ax, bars, total_scripts)
    ax.set_xlabel('Number of scripts containing this operation')
    ax.set_title(f'Operation presence in {split_name} ({total_scripts:,} scripts)\n'
                 f'(% = fraction of all scripts)', fontsize=11)
    ax.axvline(total_scripts, color='red', linestyle='--', alpha=0.4, linewidth=1,
               label='total scripts')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, f'01_operation_frequency_{split_name}.png'))


# ---------------------------------------------------------------------------
# Plot 2: Code length distribution
# ---------------------------------------------------------------------------

def plot_code_length(records_train: list, records_val: list, out_dir: str):
    train_lens = [r['code_length'] for r in records_train]
    val_lens   = [r['code_length'] for r in records_val]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    bins = np.linspace(0, max(max(train_lens), max(val_lens)) * 1.05, 60)
    ax.hist(train_lens, bins=bins, alpha=0.7, color='steelblue', label=f'train (n={len(train_lens):,})', density=True)
    ax.hist(val_lens,   bins=bins, alpha=0.7, color='darkorange', label=f'val (n={len(val_lens):,})', density=True)

    for lens, color, name in [(train_lens, 'steelblue', 'train'),
                              (val_lens, 'darkorange', 'val')]:
        ax.axvline(np.median(lens), color=color, linestyle='--', linewidth=1.5,
                   label=f'median {name}: {np.median(lens):.0f}')

    ax.set_xlabel('Script length (characters)')
    ax.set_ylabel('Density')
    ax.set_title('Code length distribution: train vs val')
    ax.legend()
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '02_code_length_distribution.png'))


# ---------------------------------------------------------------------------
# Plot 3: Sketch operations per script
# ---------------------------------------------------------------------------

def plot_sketch_ops(records_train: list, records_val: list, out_dir: str):
    train_ops = [r['n_sketch_ops'] for r in records_train]
    val_ops   = [r['n_sketch_ops'] for r in records_val]

    max_ops = max(max(train_ops), max(val_ops))
    bins = np.arange(0, min(max_ops + 2, 60))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    ax.hist(train_ops, bins=bins, alpha=0.7, color='steelblue',
            label=f'train  median={np.median(train_ops):.0f}', density=True)
    ax.hist(val_ops,   bins=bins, alpha=0.7, color='darkorange',
            label=f'val  median={np.median(val_ops):.0f}', density=True)
    ax.set_xlabel('Sketch operations per script (segments + arcs + circles + …)')
    ax.set_ylabel('Density')
    ax.set_title('Sketch complexity distribution')
    ax.legend()
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '03_sketch_ops_per_script.png'))


# ---------------------------------------------------------------------------
# Plot 4: Plane type distribution
# ---------------------------------------------------------------------------

def plot_plane_types(records: list, out_dir: str, split_name: str):
    counter = collections.Counter()
    for r in records:
        counter.update(r['planes'])

    # Normalize to per-script occurrence frequency
    total = len(records)
    labels = sorted(counter.keys())
    freqs  = [counter[l] / total for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    bars = ax.bar(labels, freqs, color=['steelblue', 'darkorange', 'seagreen'],
                  edgecolor='white', width=0.5)
    for bar, f in zip(bars, freqs):
        ax.text(bar.get_x() + bar.get_width() / 2, f + 0.005,
                f'{f:.2f}', ha='center', fontsize=9)
    ax.set_ylabel('Average occurrences per script')
    ax.set_title(f'Workplane type distribution ({split_name})')
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, f'04_plane_types_{split_name}.png'))


# ---------------------------------------------------------------------------
# Plot 5: Boolean / body count distribution
# ---------------------------------------------------------------------------

def plot_body_count(records: list, out_dir: str, split_name: str):
    body_counts = [r['n_bodies'] for r in records]
    max_bodies  = min(max(body_counts), 10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    # Left: histogram of body count
    ax = axes[0]
    counter = collections.Counter(body_counts)
    xs = list(range(1, max_bodies + 2))
    ys = [counter.get(x, 0) for x in xs]
    bars = ax.bar([str(x) if x <= max_bodies else f'>{max_bodies}' for x in xs],
                  ys, color='steelblue', edgecolor='white')
    total = len(records)
    for bar, y in zip(bars, ys):
        if y > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, y + total * 0.001,
                    f'{y / total * 100:.1f}%', ha='center', fontsize=7)
    ax.set_xlabel('Number of bodies (boolean ops + 1)')
    ax.set_ylabel('Scripts')
    ax.set_title(f'Body count ({split_name})')

    # Right: boolean op type breakdown
    ax = axes[1]
    bool_labels = ['union', 'cut/subtract', 'intersect']
    bool_vals   = [
        sum(r['n_unions']    for r in records),
        sum(r['n_cuts']      for r in records),
        sum(r['n_intersects'] for r in records),
    ]
    explode = [0.05] * 3
    non_zero = [(l, v) for l, v in zip(bool_labels, bool_vals) if v > 0]
    if non_zero:
        labels_nz, vals_nz = zip(*non_zero)
        ax.pie(vals_nz, labels=labels_nz, autopct='%1.1f%%',
               colors=['steelblue', 'tomato', 'seagreen'],
               explode=explode[:len(vals_nz)], startangle=90)
    ax.set_title(f'Boolean operation types ({split_name})')

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, f'05_body_and_boolean_ops_{split_name}.png'))


# ---------------------------------------------------------------------------
# Plot 6: Operation co-occurrence (which ops appear together most)
# ---------------------------------------------------------------------------

def plot_op_cooccurrence(records: list, out_dir: str, split_name: str):
    """Show which operation combinations appear most frequently."""
    flag_keys = [
        'has_arc', 'has_cut', 'has_cylinder', 'has_box',
        'has_push', 'has_subtract_mode', 'has_revolve',
        'has_fillet', 'has_polygon', 'has_spline',
    ]
    short = {
        'has_arc': 'arc',
        'has_cut': 'cut',
        'has_cylinder': 'cylinder',
        'has_box': 'box',
        'has_push': 'push',
        'has_subtract_mode': 'sub_mode',
        'has_revolve': 'revolve',
        'has_fillet': 'fillet',
        'has_polygon': 'polygon',
        'has_spline': 'spline',
    }
    n = len(flag_keys)
    matrix = np.zeros((n, n), dtype=float)
    for r in records:
        for i, ki in enumerate(flag_keys):
            if r.get(ki):
                for j, kj in enumerate(flag_keys):
                    if r.get(kj):
                        matrix[i, j] += 1
    # Normalize by total scripts
    matrix /= len(records)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
    ticks = [short[k] for k in flag_keys]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ticks, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ticks, fontsize=9)
    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if val < 0.5 else 'white')
    plt.colorbar(im, ax=ax, label='Fraction of scripts')
    ax.set_title(f'Operation co-occurrence matrix ({split_name})\n'
                 f'Cell = fraction of scripts where both row AND col appear')
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, f'06_op_cooccurrence_{split_name}.png'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyse training data CadQuery distribution')
    parser.add_argument('--data-dir', default='./data/cad-recode-v1.5/train',
                        help='Root of training data (contains batch_*/)')
    parser.add_argument('--out-dir', default='viz/plots/dataset_stats',
                        help='Directory to save plots')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Limit number of files per batch (for quick runs)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load data ---
    print(f'Loading training scripts from {args.data_dir} ...')

    # Find all batch dirs (batch_00, batch_01, …) excluding batch_val
    import glob as _glob
    batch_dirs = sorted(d for d in _glob.glob(os.path.join(args.data_dir, 'batch_*'))
                        if os.path.isdir(d) and 'val' not in os.path.basename(d))
    val_dirs   = [d for d in _glob.glob(os.path.join(args.data_dir, 'batch_*'))
                  if os.path.isdir(d) and 'val' in os.path.basename(d)]

    records_train = []
    for d in batch_dirs:
        recs = load_cq_dir(d, glob='*.py', max_files=args.max_files)
        records_train.extend(recs)
        print(f'  {os.path.basename(d)}: {len(recs)} scripts')

    records_val = []
    for d in val_dirs:
        recs = load_cq_dir(d, glob='*.py', max_files=args.max_files)
        records_val.extend(recs)
        print(f'  {os.path.basename(d)}: {len(recs)} scripts')

    print(f'\nTotal: {len(records_train):,} train, {len(records_val):,} val')

    if not records_train:
        print('ERROR: no training scripts found. Check --data-dir path.')
        sys.exit(1)

    # --- Summary stats ---
    print('\n=== Summary (train) ===')
    for key in ['code_length', 'n_sketch_ops', 'n_bool_ops', 'n_bodies',
                'n_segments', 'n_arcs', 'n_circles']:
        vals = [r[key] for r in records_train]
        print(f'  {key:20s}  mean={np.mean(vals):.1f}  median={np.median(vals):.1f}'
              f'  p95={np.percentile(vals, 95):.1f}')

    # --- Generate plots ---
    print(f'\nGenerating plots → {args.out_dir}/')
    plot_operation_frequency(records_train, args.out_dir, 'train')
    if records_val:
        plot_operation_frequency(records_val, args.out_dir, 'val')

    if records_val:
        plot_code_length(records_train, records_val, args.out_dir)
        plot_sketch_ops(records_train, records_val, args.out_dir)
    else:
        # Single split: just plot train as histogram
        lengths = [r['code_length'] for r in records_train]
        fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
        ax.hist(lengths, bins=60, color='steelblue', edgecolor='white')
        ax.axvline(np.median(lengths), color='red', linestyle='--', linewidth=1.5,
                   label=f'median: {np.median(lengths):.0f} chars')
        ax.set_xlabel('Script length (characters)')
        ax.set_ylabel('Scripts')
        ax.set_title(f'Code length distribution (train, n={len(lengths):,})')
        ax.legend()
        fig.tight_layout()
        _savefig(fig, os.path.join(args.out_dir, '02_code_length_train.png'))

    plot_plane_types(records_train, args.out_dir, 'train')
    plot_body_count(records_train, args.out_dir, 'train')
    plot_op_cooccurrence(records_train, args.out_dir, 'train')

    print('\nDone.')


if __name__ == '__main__':
    main()
