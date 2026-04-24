"""Failure mode analysis for model-generated CadQuery scripts.

For each generated .py file in an eval directory:
  1. Parse structural features (viz/parse_cq.py)
  2. Execute in a subprocess with timeout, categorise the result
  3. Optionally merge per-sample IoU/CD from evaluate.py --results-csv
  4. Generate plots showing failure rates, error types, and complexity correlations

Error categories
----------------
  success          code ran and produced valid geometry
  syntax_error     Python cannot parse the file
  no_result        code ran but variable 'r' is undefined
  attribute_error  invalid CadQuery method (model hallucinated an API)
  geometry_error   OCC / tessellation failure (degenerate or empty solid)
  timeout          execution exceeded --timeout seconds
  other_error      any other exception

Usage:
    # Basic (no IoU data)
    python viz/failure_analysis.py --eval-dir ./work_dirs/eval_hf_baseline

    # With train data for distribution comparison
    python viz/failure_analysis.py \
        --eval-dir  ./work_dirs/eval_hf_baseline \
        --train-dir ./data/cad-recode-v1.5/train

    # With per-sample IoU (run evaluate.py --results-csv first)
    python viz/failure_analysis.py \
        --eval-dir      ./work_dirs/eval_hf_baseline \
        --results-csv   ./work_dirs/eval_hf_baseline/results.csv \
        --train-dir     ./data/cad-recode-v1.5/train
"""

import os
import sys
import json
import textwrap
import tempfile
import subprocess
import argparse
import collections
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.parse_cq import parse_cq_script, load_cq_dir

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
except ImportError:
    print('ERROR: matplotlib not installed. Run: pip install matplotlib')
    sys.exit(1)

# ---------------------------------------------------------------------------
# Subprocess worker: execute one CadQuery script, return status + error type
# ---------------------------------------------------------------------------

_EXEC_WORKER = textwrap.dedent('''\
    import sys, json, traceback
    code = sys.stdin.read()
    try:
        import cadquery as cq  # noqa
        g = {}
        exec(code, g)
        r = g.get('r')
        if r is None:
            print(json.dumps({'status': 'no_result',
                              'error_type': 'no_result',
                              'detail': "variable 'r' not defined after exec"}))
        else:
            val = r.val()
            import trimesh, numpy as np
            verts, faces = val.tessellate(0.001, 0.1)
            mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
            if len(mesh.faces) < 2:
                print(json.dumps({'status': 'geometry_error',
                                  'error_type': 'geometry_error',
                                  'detail': 'degenerate mesh (<2 faces)'}))
            else:
                print(json.dumps({'status': 'success',
                                  'error_type': None,
                                  'detail': None}))
    except SyntaxError as e:
        print(json.dumps({'status': 'syntax_error', 'error_type': 'syntax_error',
                          'detail': str(e)[:200]}))
    except AttributeError as e:
        msg = str(e)
        print(json.dumps({'status': 'attribute_error', 'error_type': 'attribute_error',
                          'detail': msg[:200]}))
    except Exception as e:
        etype = type(e).__name__
        # Map OCC/OCCT exceptions to geometry_error
        occ_names = ('Standard_ConstructionError', 'Standard_NoSuchObject',
                     'BRep_NotDone', 'StdFail_NotDone', 'TopoDS_UnCompatibleShapes',
                     'Standard_TypeMismatch')
        if etype in occ_names or 'OCC' in etype or 'Standard_' in etype:
            mapped = 'geometry_error'
        elif etype == 'NameError':
            mapped = 'no_result'
        else:
            mapped = 'other_error'
        print(json.dumps({'status': mapped,
                          'error_type': etype,
                          'detail': str(e)[:200]}))
    sys.stdout.flush()
''')

_worker_path: str = None


def _get_worker() -> str:
    global _worker_path
    if _worker_path and os.path.exists(_worker_path):
        return _worker_path
    fd, path = tempfile.mkstemp(suffix='.py', prefix='cq_exec_worker_')
    with os.fdopen(fd, 'w') as f:
        f.write(_EXEC_WORKER)
    _worker_path = path
    return path


def _exec_script(code: str, timeout: float = 15.0) -> dict:
    """Execute code in subprocess, return {status, error_type, detail}."""
    try:
        proc = subprocess.run(
            [sys.executable, _get_worker()],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        return {'status': 'other_error', 'error_type': 'no_output',
                'detail': proc.stderr[:200] if proc.stderr else ''}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'error_type': 'timeout', 'detail': ''}
    except Exception as e:
        return {'status': 'other_error', 'error_type': type(e).__name__,
                'detail': str(e)[:200]}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {path}')


COLORS = {
    'success':         '#4CAF50',
    'syntax_error':    '#F44336',
    'no_result':       '#FF9800',
    'attribute_error': '#9C27B0',
    'geometry_error':  '#2196F3',
    'timeout':         '#795548',
    'other_error':     '#607D8B',
}


# ---------------------------------------------------------------------------
# Plot 1: Failure type breakdown
# ---------------------------------------------------------------------------

def plot_failure_breakdown(records: list, out_dir: str):
    counter = collections.Counter(r['status'] for r in records)
    order   = ['success', 'syntax_error', 'no_result',
               'attribute_error', 'geometry_error', 'timeout', 'other_error']
    labels  = [k for k in order if k in counter]
    vals    = [counter[k] for k in labels]
    colors  = [COLORS[k] for k in labels]
    total   = len(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    # Bar chart
    ax = axes[0]
    bars = ax.barh(labels, vals, color=colors, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(v + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v}  ({v/total*100:.1f}%)', va='center', fontsize=9)
    ax.set_xlabel('Number of scripts')
    ax.set_xlim(0, total * 1.25)
    ax.set_title(f'Failure type breakdown  (n={total})')

    # Pie chart (excluding success for visibility)
    ax = axes[1]
    fail_labels = [k for k in labels if k != 'success']
    fail_vals   = [counter[k] for k in fail_labels]
    fail_colors = [COLORS[k] for k in fail_labels]
    total_fail  = sum(fail_vals)
    if total_fail > 0:
        ax.pie(fail_vals,
               labels=[f'{l}\n({v/total*100:.1f}%)' for l, v in zip(fail_labels, fail_vals)],
               colors=fail_colors, startangle=90, autopct='%1.0f%%',
               pctdistance=0.7)
    ax.set_title(f'Failure breakdown  (fails={total_fail}, {total_fail/total*100:.1f}% of total)')

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '01_failure_type_breakdown.png'))

    return counter


# ---------------------------------------------------------------------------
# Plot 2: Code length — success vs failure
# ---------------------------------------------------------------------------

def plot_length_vs_status(records: list, out_dir: str):
    success = [r['code_length'] for r in records if r['status'] == 'success']
    failure = [r['code_length'] for r in records if r['status'] != 'success']

    if not success or not failure:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    # Overlapping histograms
    ax = axes[0]
    bins = np.linspace(0, max(max(success), max(failure)) * 1.05, 50)
    ax.hist(success, bins=bins, alpha=0.6, color='#4CAF50',
            density=True, label=f'success (n={len(success)})')
    ax.hist(failure, bins=bins, alpha=0.6, color='#F44336',
            density=True, label=f'failure (n={len(failure)})')
    ax.axvline(np.median(success), color='#2E7D32', linestyle='--', linewidth=1.5,
               label=f'median success: {np.median(success):.0f}')
    ax.axvline(np.median(failure), color='#B71C1C', linestyle='--', linewidth=1.5,
               label=f'median failure: {np.median(failure):.0f}')
    ax.set_xlabel('Script length (characters)')
    ax.set_ylabel('Density')
    ax.set_title('Code length: success vs failure')
    ax.legend(fontsize=8)

    # Binned failure rate
    ax = axes[1]
    all_lens  = [r['code_length'] for r in records]
    all_fail  = [r['status'] != 'success' for r in records]
    bin_edges = np.percentile(all_lens, np.linspace(0, 100, 11))  # decile bins
    bin_edges = np.unique(bin_edges)
    midpoints, fail_rates, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = [(lo <= l < hi) for l in all_lens]
        n = sum(mask)
        if n > 0:
            fail_r = sum(f for f, m in zip(all_fail, mask) if m) / n
            midpoints.append((lo + hi) / 2)
            fail_rates.append(fail_r)
            counts.append(n)
    sc = ax.scatter(midpoints, fail_rates, s=[c / 2 for c in counts],
                    c=fail_rates, cmap='RdYlGn_r', vmin=0, vmax=1,
                    edgecolors='black', linewidths=0.5, zorder=3)
    ax.plot(midpoints, fail_rates, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Failure rate')
    ax.set_xlabel('Script length (characters)')
    ax.set_ylabel('Failure rate (by length decile)')
    ax.set_title('Failure rate vs code length\n(bubble size ∝ sample count)')
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '02_code_length_vs_status.png'))


# ---------------------------------------------------------------------------
# Plot 3: Sketch ops — success vs failure (violin / box)
# ---------------------------------------------------------------------------

def plot_sketch_ops_vs_status(records: list, out_dir: str):
    success_ops = [r['n_sketch_ops'] for r in records if r['status'] == 'success']
    failure_ops = [r['n_sketch_ops'] for r in records if r['status'] != 'success']

    if not success_ops or not failure_ops:
        return

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    vp = ax.violinplot([success_ops, failure_ops], positions=[1, 2],
                       showmedians=True, showextrema=True)
    vp['bodies'][0].set_facecolor('#4CAF50')
    vp['bodies'][1].set_facecolor('#F44336')
    for part in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp[part].set_edgecolor('black')
        vp[part].set_linewidth(1.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Success\n(n={len(success_ops)})',
                        f'Failure\n(n={len(failure_ops)})'])
    ax.set_ylabel('Sketch operations per script')
    ax.set_title('Sketch complexity: success vs failure')
    ax.text(1, np.median(success_ops), f' {np.median(success_ops):.1f}',
            va='center', color='darkgreen', fontsize=9)
    ax.text(2, np.median(failure_ops), f' {np.median(failure_ops):.1f}',
            va='center', color='darkred', fontsize=9)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '03_sketch_ops_vs_status.png'))


# ---------------------------------------------------------------------------
# Plot 4: Failure rate by operation type
# ---------------------------------------------------------------------------

def plot_failure_rate_by_op(records: list, out_dir: str):
    """For each operation type, compute failure rate among scripts that use it."""
    flag_features = [
        ('has_arc',           'arc'),
        ('has_cut',           'cut / subtract'),
        ('has_cylinder',      'cylinder'),
        ('has_box',           'box'),
        ('has_push',          'push (multi-region)'),
        ('has_subtract_mode', 'subtract-mode (mode=s)'),
        ('has_revolve',       'revolve'),
        ('has_fillet',        'fillet'),
        ('has_polygon',       'polygon'),
        ('has_spline',        'spline'),
        ('has_sphere',        'sphere'),
        ('has_multi_body',    'any boolean op'),
        ('has_loft',          'loft'),
        ('has_sweep',         'sweep'),
        ('has_chamfer',       'chamfer'),
        ('has_shell',         'shell'),
    ]

    # Baseline failure rate (all scripts)
    total = len(records)
    baseline = sum(1 for r in records if r['status'] != 'success') / total

    rows = []
    for feat_key, label in flag_features:
        subset = [r for r in records if r.get(feat_key)]
        n = len(subset)
        if n < 3:   # skip if too few samples
            continue
        fail_rate = sum(1 for r in subset if r['status'] != 'success') / n
        rows.append((label, fail_rate, n))

    # Sort by failure rate descending
    rows.sort(key=lambda x: x[1], reverse=True)
    labels, fail_rates, counts = zip(*rows) if rows else ([], [], [])

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    colors = ['#F44336' if f > baseline else '#4CAF50' for f in fail_rates]
    bars = ax.barh(labels, fail_rates, color=colors, edgecolor='white')

    # Annotate with n and rate
    for bar, f, n in zip(bars, fail_rates, counts):
        ax.text(f + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{f*100:.1f}%  (n={n})', va='center', fontsize=8)

    ax.axvline(baseline, color='black', linestyle='--', linewidth=1.5,
               label=f'baseline failure rate: {baseline*100:.1f}%')
    ax.set_xlabel('Failure rate')
    ax.set_xlim(0, 1.0)
    ax.set_title('Failure rate by operation type\n'
                 '(red = above baseline, green = below baseline)')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '04_failure_rate_by_op_type.png'))


# ---------------------------------------------------------------------------
# Plot 5: Training data vs generated distribution (distribution shift)
# ---------------------------------------------------------------------------

def plot_distribution_shift(records_eval: list, records_train: list, out_dir: str):
    """Side-by-side bar: op presence rate in train vs generated scripts."""
    op_keys = [
        ('segment',       'n_segments'),
        ('arc',           'n_arcs'),
        ('circle',        'n_circles'),
        ('rect',          'n_rects'),
        ('push',          'n_push'),
        ('subtract_mode', 'n_subtract_mode'),
        ('union',         'n_unions'),
        ('cut',           'n_cuts'),
        ('cylinder',      'n_cylinders'),
        ('box',           'n_boxes'),
        ('revolve',       'n_revolves'),
        ('fillet',        'n_fillets'),
        ('polygon',       'n_polygons'),
        ('spline',        'n_splines'),
    ]

    n_train = len(records_train)
    n_eval  = len(records_eval)

    labels, train_rates, eval_rates = [], [], []
    for label, key in op_keys:
        tr = sum(1 for r in records_train if r.get(key, 0) > 0) / n_train
        ev = sum(1 for r in records_eval  if r.get(key, 0) > 0) / n_eval
        labels.append(label)
        train_rates.append(tr)
        eval_rates.append(ev)

    # Sort by train rate descending
    order = np.argsort(train_rates)[::-1]
    labels      = [labels[i]      for i in order]
    train_rates = [train_rates[i] for i in order]
    eval_rates  = [eval_rates[i]  for i in order]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    b1 = ax.bar(x - width / 2, train_rates, width, label=f'train (n={n_train:,})',
                color='steelblue', edgecolor='white')
    b2 = ax.bar(x + width / 2, eval_rates,  width, label=f'generated (n={n_eval:,})',
                color='darkorange', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Fraction of scripts containing this op')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title('Distribution shift: training data vs model-generated scripts')
    ax.legend()
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '05_distribution_shift.png'))


# ---------------------------------------------------------------------------
# Plots 6-7: IoU correlations (require results CSV)
# ---------------------------------------------------------------------------

def plot_iou_vs_complexity(records: list, out_dir: str):
    """Scatter: IoU vs code length, and IoU vs sketch ops (for successful scripts)."""
    with_iou = [r for r in records if r.get('iou') is not None and r['status'] == 'success']
    if len(with_iou) < 5:
        print('  skipping IoU plots (too few samples with IoU scores)')
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    for ax, xkey, xlabel in [
        (axes[0], 'code_length',  'Script length (chars)'),
        (axes[1], 'n_sketch_ops', 'Sketch operations per script'),
    ]:
        xs   = [r[xkey]  for r in with_iou]
        ious = [r['iou'] for r in with_iou]
        sc   = ax.scatter(xs, ious, alpha=0.5, s=15, c=ious, cmap='RdYlGn',
                          vmin=0, vmax=1, edgecolors='none')
        # Binned mean
        try:
            bin_edges = np.percentile(xs, np.linspace(0, 100, 8))
            bin_edges = np.unique(bin_edges)
            midpts, means = [], []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = [lo <= x < hi for x in xs]
                if sum(mask) > 0:
                    midpts.append((lo + hi) / 2)
                    means.append(np.mean([i for i, m in zip(ious, mask) if m]))
            ax.plot(midpts, means, color='black', linewidth=2, label='binned mean')
        except Exception:
            pass
        plt.colorbar(sc, ax=ax, label='IoU')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('IoU')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'IoU vs {xlabel}')
        ax.legend(fontsize=8)

    fig.suptitle(f'IoU quality vs script complexity  (n={len(with_iou)} successful scripts)',
                 fontsize=11)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '06_iou_vs_complexity.png'))


def plot_iou_by_op_type(records: list, out_dir: str):
    """Box plot: IoU distribution for scripts containing each operation type."""
    with_iou = [r for r in records if r.get('iou') is not None]
    if len(with_iou) < 5:
        return

    flag_keys = [
        ('has_arc',           'arc'),
        ('has_cut',           'cut'),
        ('has_cylinder',      'cylinder'),
        ('has_box',           'box'),
        ('has_push',          'push'),
        ('has_subtract_mode', 'sub-mode'),
        ('has_revolve',       'revolve'),
        ('has_fillet',        'fillet'),
        ('has_multi_body',    'multi-body'),
    ]

    groups, group_labels = [], []
    # Baseline: all scripts with IoU
    groups.append([r['iou'] for r in with_iou])
    group_labels.append(f'all\n(n={len(with_iou)})')

    for feat_key, label in flag_keys:
        subset = [r['iou'] for r in with_iou if r.get(feat_key)]
        if len(subset) >= 3:
            groups.append(subset)
            group_labels.append(f'{label}\n(n={len(subset)})')

    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.2), 5), facecolor='white')
    bp = ax.boxplot(groups, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color='black', linewidth=2))
    colors = ['#90A4AE'] + ['#4CAF50' if np.median(g) >= np.median(groups[0])
                            else '#F44336' for g in groups[1:]]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel('IoU')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('IoU distribution by operation type\n'
                 '(green = median ≥ baseline, red = below baseline)')
    ax.axhline(np.median(groups[0]), color='gray', linestyle='--', linewidth=1,
               label=f'baseline median: {np.median(groups[0]):.3f}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '07_iou_by_op_type.png'))


# ---------------------------------------------------------------------------
# Plot 8: Error message analysis (top errors by type)
# ---------------------------------------------------------------------------

def plot_error_analysis(records: list, out_dir: str):
    """Two panels: (a) top error detail strings; (b) hallucinated API names."""
    failures = [r for r in records if r['status'] != 'success' and r.get('detail')]

    if not failures:
        return

    # --- Panel A: top error messages (cleaned) ---
    import re as _re
    def _clean(detail: str) -> str:
        # Normalise numbers and file paths to reduce noise
        s = _re.sub(r'\b\d+\.\d+\b', 'N', detail)
        s = _re.sub(r'\b\d+\b', 'N', s)
        s = _re.sub(r'/[^ ]+', '<path>', s)
        return s[:120].strip()

    cleaned = collections.Counter(_clean(r['detail']) for r in failures)
    top_errors = cleaned.most_common(12)
    if not top_errors:
        return

    # --- Panel B: hallucinated attribute names ---
    attr_failures = [r for r in failures if r['status'] == 'attribute_error']
    api_names: list = []
    for r in attr_failures:
        m = _re.search(r"attribute '([^']+)'", r.get('detail', ''))
        if m:
            api_names.append(m.group(1))
    attr_counter = collections.Counter(api_names)
    top_attrs = attr_counter.most_common(12)

    n_panels = 1 + (1 if top_attrs else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, max(5, len(top_errors) * 0.45 + 1)),
                             facecolor='white')
    if n_panels == 1:
        axes = [axes]

    # Error messages
    ax = axes[0]
    msgs, cnts = zip(*top_errors) if top_errors else ([], [])
    ax.barh(range(len(msgs)), cnts, color='#EF5350', edgecolor='white')
    ax.set_yticks(range(len(msgs)))
    ax.set_yticklabels([m[:70] for m in msgs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title(f'Top error messages  (failures={len(failures)})')

    # Hallucinated API names
    if top_attrs:
        ax = axes[1]
        anames, acnts = zip(*top_attrs)
        bars = ax.barh(anames, acnts, color='#9C27B0', edgecolor='white')
        for bar, c in zip(bars, acnts):
            ax.text(c + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(c), va='center', fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Count')
        ax.set_title(f'Hallucinated CQ API attributes\n'
                     f'(attribute_errors = {len(attr_failures)})')

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '08_error_analysis.png'))


# ---------------------------------------------------------------------------
# Plot 9: Chamfer Distance correlations (require results CSV)
# ---------------------------------------------------------------------------

def plot_cd_analysis(records: list, out_dir: str):
    """CD vs complexity and CD by op type (mirrors IoU plots 6-7)."""
    with_cd = [r for r in records if r.get('cd') is not None and r['status'] == 'success']
    if len(with_cd) < 5:
        return

    # --- 9a: CD vs complexity ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
    for ax, xkey, xlabel in [
        (axes[0], 'code_length',  'Script length (chars)'),
        (axes[1], 'n_sketch_ops', 'Sketch ops per script'),
    ]:
        xs  = [r[xkey] for r in with_cd]
        cds = [r['cd']  for r in with_cd]
        sc  = ax.scatter(xs, cds, alpha=0.5, s=15, c=cds, cmap='RdYlGn_r',
                         edgecolors='none')
        try:
            bin_edges = np.percentile(xs, np.linspace(0, 100, 8))
            bin_edges = np.unique(bin_edges)
            midpts, means = [], []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = [lo <= x < hi for x in xs]
                if sum(mask) > 0:
                    midpts.append((lo + hi) / 2)
                    means.append(np.mean([c for c, m in zip(cds, mask) if m]))
            ax.plot(midpts, means, color='black', linewidth=2, label='binned mean')
        except Exception:
            pass
        plt.colorbar(sc, ax=ax, label='CD (×10³)', format='%.2f')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Chamfer Distance')
        ax.set_title(f'CD vs {xlabel}')
        ax.legend(fontsize=8)
    fig.suptitle(f'CD quality vs script complexity  (n={len(with_cd)} scripts)', fontsize=11)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '09_cd_vs_complexity.png'))

    # --- 9b: CD by op type ---
    flag_keys = [
        ('has_arc',           'arc'),
        ('has_cut',           'cut'),
        ('has_cylinder',      'cylinder'),
        ('has_box',           'box'),
        ('has_push',          'push'),
        ('has_subtract_mode', 'sub-mode'),
        ('has_revolve',       'revolve'),
        ('has_fillet',        'fillet'),
        ('has_multi_body',    'multi-body'),
    ]
    groups, group_labels = [[r['cd'] for r in with_cd]], [f'all\n(n={len(with_cd)})']
    for feat_key, label in flag_keys:
        subset = [r['cd'] for r in with_cd if r.get(feat_key)]
        if len(subset) >= 3:
            groups.append(subset)
            group_labels.append(f'{label}\n(n={len(subset)})')

    baseline_median = np.median(groups[0])
    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 1.2), 5), facecolor='white')
    bp = ax.boxplot(groups, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color='black', linewidth=2))
    colors = ['#90A4AE'] + ['#F44336' if np.median(g) > baseline_median
                            else '#4CAF50' for g in groups[1:]]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel('Chamfer Distance')
    ax.set_title('CD distribution by operation type\n'
                 '(red = median > baseline, green = below baseline)')
    ax.axhline(baseline_median, color='gray', linestyle='--', linewidth=1,
               label=f'baseline median: {baseline_median*1000:.3f} ×10³')
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '10_cd_by_op_type.png'))


# ---------------------------------------------------------------------------
# Plot 11: IoU/CD joint scatter — quality summary
# ---------------------------------------------------------------------------

def plot_iou_cd_joint(records: list, out_dir: str):
    """Scatter of IoU vs CD per sample, coloured by script complexity."""
    with_both = [r for r in records if r.get('iou') is not None and r.get('cd') is not None]
    if len(with_both) < 5:
        return

    ious       = [r['iou']          for r in with_both]
    cds        = [r['cd']           for r in with_both]
    complexity = [r['n_sketch_ops'] for r in with_both]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    sc = ax.scatter(cds, ious, c=complexity, cmap='viridis', alpha=0.7, s=30, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Sketch ops (complexity)')
    ax.set_xlabel('Chamfer Distance')
    ax.set_ylabel('IoU')
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'IoU vs Chamfer Distance per sample  (n={len(with_both)})\n'
                 f'IoU mean={np.mean(ious):.3f}  CD median={np.median(cds)*1000:.3f}×10³')
    # Annotate top-left (high IoU, low CD = best) and bottom-right (worst)
    ax.text(0.02, 0.97, 'best', transform=ax.transAxes,
            color='green', fontsize=9, va='top')
    ax.text(0.98, 0.03, 'worst', transform=ax.transAxes,
            color='red', fontsize=9, ha='right')
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, '11_iou_cd_joint.png'))


# ---------------------------------------------------------------------------
# Load results CSV (from evaluate.py --results-csv)
# ---------------------------------------------------------------------------

def load_results_csv(csv_path: str) -> dict:
    """Load per-sample IoU/CD from evaluate.py results CSV.

    Returns dict: {file_name: {'iou': float|None, 'cd': float|None}}
    Handles two possible CSV formats:
      - one row per generated .py (file_name includes +index suffix)
      - one row per GT item (best prediction selected)
    """
    results = {}
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get('file_name') or row.get('stem', '')
                # Strip +index suffix if present to match .py filename stems
                base = key.split('+')[0] if '+' in key else key
                iou  = float(row['iou']) if row.get('iou') not in (None, '', 'None') else None
                cd   = float(row['cd'])  if row.get('cd')  not in (None, '', 'None') else None
                # Store by full key first; prefer best iou if multiple rows
                full_key = row.get('file_name', key)
                if full_key not in results or (iou is not None and
                        (results[full_key]['iou'] is None or iou > results[full_key]['iou'])):
                    results[full_key] = {'iou': iou, 'cd': cd}
    except Exception as e:
        print(f'  warning: could not load results CSV ({e})')
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Failure mode analysis for generated CadQuery scripts')
    parser.add_argument('--eval-dir', required=True,
                        help='Directory of generated .py files (from test.py)')
    parser.add_argument('--train-dir', default=None,
                        help='Training data dir for distribution shift comparison')
    parser.add_argument('--results-csv', default=None,
                        help='Per-sample IoU/CD CSV from evaluate.py --results-csv')
    parser.add_argument('--out-dir', default='viz/plots/failure_analysis',
                        help='Directory to save plots')
    parser.add_argument('--timeout', type=float, default=15.0,
                        help='Subprocess execution timeout per script (seconds)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel subprocess workers')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Limit number of eval files (for quick runs)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load and parse eval scripts ---
    print(f'Loading eval scripts from {args.eval_dir} ...')
    from pathlib import Path
    py_paths = sorted(Path(args.eval_dir).glob('*.py'))
    if args.max_files:
        py_paths = py_paths[:args.max_files]
    print(f'  {len(py_paths)} scripts found')

    records = []
    for path in py_paths:
        try:
            code = path.read_text(encoding='utf-8', errors='replace')
            feat = parse_cq_script(code)
            feat['path']  = str(path)
            feat['stem']  = path.stem
            feat['code']  = code
            feat['status']      = None
            feat['error_type']  = None
            feat['detail']      = None
            records.append(feat)
        except Exception:
            pass

    # --- Execute scripts (parallel via ThreadPoolExecutor) ---
    print(f'Executing {len(records)} scripts (timeout={args.timeout}s, workers={args.workers}) ...')
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm as _tqdm

    def _run(rec):
        result = _exec_script(rec['code'], timeout=args.timeout)
        return rec['stem'], result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, r): r for r in records}
        stem_to_result = {}
        for fut in _tqdm(as_completed(futures), total=len(futures), desc='exec'):
            stem, res = fut.result()
            stem_to_result[stem] = res

    for rec in records:
        res = stem_to_result.get(rec['stem'], {})
        rec['status']     = res.get('status', 'other_error')
        rec['error_type'] = res.get('error_type')
        rec['detail']     = res.get('detail')

    # Print summary
    counter = collections.Counter(r['status'] for r in records)
    total   = len(records)
    print('\nExecution results:')
    for status, n in sorted(counter.items(), key=lambda x: -x[1]):
        print(f'  {status:20s} {n:4d}  ({n/total*100:.1f}%)')

    # --- Merge IoU/CD from results CSV (if provided) ---
    if args.results_csv and os.path.exists(args.results_csv):
        print(f'\nMerging IoU scores from {args.results_csv} ...')
        iou_map = load_results_csv(args.results_csv)
        merged  = 0
        for rec in records:
            stem_base = rec['stem'].split('+')[0]  # e.g. 00000093
            # Try exact match, then base match
            hit = iou_map.get(rec['stem']) or iou_map.get(stem_base)
            if hit:
                rec['iou'] = hit['iou']
                rec['cd']  = hit['cd']
                merged += 1
        print(f'  merged IoU scores for {merged}/{total} scripts')
    else:
        for rec in records:
            rec['iou'] = None
            rec['cd']  = None

    # --- Load training data for distribution shift ---
    records_train = []
    if args.train_dir:
        print(f'\nLoading training data from {args.train_dir} ...')
        import glob as _glob
        batch_dirs = sorted(d for d in _glob.glob(os.path.join(args.train_dir, 'batch_*'))
                            if os.path.isdir(d) and 'val' not in os.path.basename(d))
        for d in batch_dirs:
            records_train.extend(load_cq_dir(d, glob='*.py'))
        print(f'  {len(records_train):,} training scripts loaded')

    # --- Generate plots ---
    print(f'\nGenerating plots → {args.out_dir}/')
    plot_failure_breakdown(records, args.out_dir)
    plot_length_vs_status(records, args.out_dir)
    plot_sketch_ops_vs_status(records, args.out_dir)
    plot_failure_rate_by_op(records, args.out_dir)
    if records_train:
        plot_distribution_shift(records, records_train, args.out_dir)
    plot_error_analysis(records, args.out_dir)
    if any(r.get('iou') is not None for r in records):
        plot_iou_vs_complexity(records, args.out_dir)
        plot_iou_by_op_type(records, args.out_dir)
    if any(r.get('cd') is not None for r in records):
        plot_cd_analysis(records, args.out_dir)
    if any(r.get('iou') is not None and r.get('cd') is not None for r in records):
        plot_iou_cd_joint(records, args.out_dir)

    # --- Print text summary for root cause ---
    print('\n=== Root Cause Summary ===')
    failures = [r for r in records if r['status'] != 'success']
    if failures:
        by_type = collections.defaultdict(list)
        for r in failures:
            by_type[r['status']].append(r.get('detail', ''))
        for etype, details in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f'\n  {etype} ({len(details)} failures):')
            import re as _re2
            # hallucinated API extraction
            if etype == 'attribute_error':
                attrs = [_re2.search(r"attribute '([^']+)'", d or '') for d in details]
                names = collections.Counter(m.group(1) for m in attrs if m)
                for name, n in names.most_common(5):
                    print(f'    no attribute "{name}" × {n}')
            else:
                for d in details[:3]:
                    print(f'    {(d or "").strip()[:120]}')
    else:
        print('  No failures detected.')

    print('\nDone.')


if __name__ == '__main__':
    main()
