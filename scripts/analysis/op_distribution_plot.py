"""Op-usage distribution plots across SFT data sources.

Two plots:
  1. op_freq_per_dataset.png — % of codes containing each op (top-20 per dataset)
  2. ops_per_case_distribution.png — overlay histogram of distinct ops per code

Usage:
    uv run python -m scripts.analysis.op_distribution_plot
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.parse_cq import parse_cq_script

# Dataset roots and color codes
DATASETS = [
    ('benchcad',          'data/benchcad',          'tab:red'),
    ('cad_iso_106',       'data/cad-iso-106',       'tab:orange'),
    ('benchcad_simple',   'data/benchcad-simple',   'goldenrod'),
    ('text2cad_bench',    'data/text2cad-bench',    'tab:purple'),
    ('cad_recode_bench',  'data/cad-recode-bench',  'tab:blue'),
    ('cad_recode_20k',    'data/cad-recode-20k',    'tab:cyan'),
]

# Leaf op count keys (skip aggregate keys)
LEAF_OP_KEYS = [
    'n_workplanes', 'n_secondary_wps',
    'n_segments', 'n_arcs', 'n_splines',
    'n_circles', 'n_rects', 'n_polygons', 'n_ellipses',
    'n_extrudes', 'n_revolves', 'n_lofts', 'n_sweeps',
    'n_unions', 'n_cuts', 'n_intersects',
    'n_cylinders', 'n_boxes', 'n_spheres',
    'n_fillets', 'n_chamfers', 'n_shells',
    'n_push', 'n_subtract_mode',
]

# Display names (drop 'n_' prefix and trailing 's' for readability)
DISPLAY = {
    'n_workplanes':    'workplane',
    'n_secondary_wps': 'workplane2',
    'n_segments':      'segment',
    'n_arcs':          'arc',
    'n_splines':       'spline',
    'n_circles':       'circle',
    'n_rects':         'rect',
    'n_polygons':      'polygon',
    'n_ellipses':      'ellipse',
    'n_extrudes':      'extrude',
    'n_revolves':      'revolve',
    'n_lofts':         'loft',
    'n_sweeps':        'sweep',
    'n_unions':        'union',
    'n_cuts':          'cut',
    'n_intersects':    'intersect',
    'n_cylinders':     'cylinder',
    'n_boxes':         'box',
    'n_spheres':       'sphere',
    'n_fillets':       'fillet',
    'n_chamfers':      'chamfer',
    'n_shells':        'shell',
    'n_push':          'push',
    'n_subtract_mode': 'subtract_mode',
}

N_SAMPLE = 500
SEED = 42
OUT_DIR = REPO_ROOT / 'docs/op_distribution_2026-04-29'


def resolve_code(row: dict, root: Path) -> str | None:
    """Return code string for a row, or None if can't load."""
    code = row.get('code')
    if code:
        return code
    py = row.get('py_path')
    if py:
        p = (root / py)
        if p.exists():
            return p.read_text(encoding='utf-8', errors='replace')
    uid = row.get('uid')
    if uid:
        for sub in ('cadquery', '.', 'py'):
            p = root / sub / f'{uid}.py'
            if p.exists():
                return p.read_text(encoding='utf-8', errors='replace')
    return None


def analyze_dataset(label: str, root: Path) -> dict:
    """Sample N items, parse, return per-op presence + distinct ops counts."""
    pkl = root / 'train.pkl'
    rows = pickle.load(open(pkl, 'rb'))
    rng = random.Random(SEED)
    sample = rng.sample(rows, min(N_SAMPLE, len(rows)))

    op_present_count = {k: 0 for k in LEAF_OP_KEYS}
    distinct_per_case: list[int] = []
    n_parsed = 0

    for r in sample:
        code = resolve_code(r, root)
        if not code:
            continue
        try:
            feats = parse_cq_script(code)
        except Exception:
            continue
        n_parsed += 1
        n_distinct = 0
        for k in LEAF_OP_KEYS:
            if feats.get(k, 0) > 0:
                op_present_count[k] += 1
                n_distinct += 1
        distinct_per_case.append(n_distinct)

    op_present_pct = {DISPLAY[k]: 100.0 * v / max(n_parsed, 1)
                      for k, v in op_present_count.items()}
    return {
        'label':          label,
        'n_parsed':       n_parsed,
        'op_present_pct': op_present_pct,
        'distinct_per_case': np.array(distinct_per_case),
    }


def plot_op_freq(stats: list[dict], out_path: Path, colors: list[str]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=130)
    for ax, st, color in zip(axes.flat, stats, colors):
        items = sorted(st['op_present_pct'].items(), key=lambda kv: kv[1])
        items = items[-20:]  # top 20
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        ax.barh(names, vals, color=color, edgecolor='black', linewidth=0.3)
        ax.set_xlim(0, 100)
        ax.set_xlabel('% of codes containing op')
        ax.set_title(f'{st["label"]}  (n={st["n_parsed"]})', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    fig.suptitle('Op presence per dataset (top 20 by frequency)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def plot_ops_per_case(stats: list[dict], out_path: Path, colors: list[str]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=130)
    bins = np.arange(0, 26)
    for st, color in zip(stats, colors):
        d = st['distinct_per_case']
        if len(d) == 0:
            continue
        ax.hist(d, bins=bins, density=True, alpha=0.4, color=color,
                label=f'{st["label"]} (med={np.median(d):.1f}, mean={d.mean():.2f})',
                edgecolor=color, linewidth=1.0)
        # vertical median line
        ax.axvline(np.median(d), color=color, linestyle='--',
                   alpha=0.7, linewidth=1.3)
    ax.set_xlabel('# distinct ops per code')
    ax.set_ylabel('density')
    ax.set_xlim(0, 25)
    ax.set_title(f'Distinct ops per code (n={N_SAMPLE} sampled per source)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = []
    colors = []
    for label, sub, color in DATASETS:
        root = REPO_ROOT / sub
        if not (root / 'train.pkl').exists():
            print(f'  SKIP {label}: no train.pkl', flush=True)
            continue
        print(f'  parsing {label} ...', flush=True)
        st = analyze_dataset(label, root)
        stats.append(st)
        colors.append(color)
        d = st['distinct_per_case']
        print(f'    n_parsed={st["n_parsed"]} '
              f'med={np.median(d):.1f} mean={d.mean():.2f} '
              f'p95={np.percentile(d, 95):.0f}', flush=True)

    p1 = OUT_DIR / 'op_freq_per_dataset.png'
    p2 = OUT_DIR / 'ops_per_case_distribution.png'
    plot_op_freq(stats, p1, colors)
    plot_ops_per_case(stats, p2, colors)

    # Summary table
    print('\n## Summary table\n')
    print(f'| dataset           | n   | ops/case median | ops/case mean | ops/case p95 |')
    print(f'|---|---|---|---|---|')
    for st in stats:
        d = st['distinct_per_case']
        med = np.median(d) if len(d) else 0
        mean = d.mean() if len(d) else 0
        p95 = np.percentile(d, 95) if len(d) else 0
        print(f'| {st["label"]:<18} | {st["n_parsed"]:>3} | '
              f'{med:>15.1f} | {mean:>13.2f} | {p95:>12.0f} |')

    print('\n## Plots\n')
    print(p1)
    print(p2)


if __name__ == '__main__':
    main()
