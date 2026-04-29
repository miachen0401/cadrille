"""Comprehensive audit of all training-data sources used in the big_bench_shell_50k run.

For each source: items count, op coverage (with confidence interval), code-length
distribution, family distribution. Combined: weighted mix expected per-step
op exposure.

Outputs:
  eval_outputs/training_data_audit/
    op_coverage.png            heatmap of op presence by dataset
    op_coverage_weighted.png   weighted by mix weights
    code_len_dist.png          length histogram per dataset
    family_distribution.png    top families per dataset
    summary.md                 markdown report

Usage:
  uv run python -m scripts.analysis.training_data_audit \
      --out eval_outputs/training_data_audit \
      --sample-per 5000
"""
import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Op vocabulary (matches online_eval._OPS exactly)
OPS_PATTERNS = {
    'box':         re.compile(r'\.box\b'),
    'cylinder':    re.compile(r'\.cylinder\b'),
    'sphere':      re.compile(r'\.sphere\b'),
    'cone':        re.compile(r'\.cone\b'),
    'circle':      re.compile(r'\.circle\b'),
    'rect':        re.compile(r'\.rect\b'),
    'polygon':     re.compile(r'\.polygon\b'),
    'polyline':    re.compile(r'\.polyline\b'),
    'segment':     re.compile(r'\.segment\b'),
    'arc':         re.compile(r'\.(threePointArc|radiusArc|tangentArc|arc)\b'),
    'spline':      re.compile(r'\.spline\b'),
    'extrude':     re.compile(r'\.extrude\b'),
    'revolve':     re.compile(r'\.revolve\b'),
    'sweep':       re.compile(r'\.sweep\b'),
    'loft':        re.compile(r'\.loft\b'),
    'cut':         re.compile(r'\.cut\b'),
    'union':       re.compile(r'\.union\b'),
    'intersect':   re.compile(r'\.intersect\b'),
    'hole':        re.compile(r'\.hole\b'),
    'cbore':       re.compile(r'\.cboreHole\b'),
    'csk':         re.compile(r'\.cskHole\b'),
    'fillet':      re.compile(r'\.fillet\b'),
    'chamfer':     re.compile(r'\.chamfer\b'),
    'shell':       re.compile(r'\.shell\b'),
    'mirror':      re.compile(r'\.mirror\b'),
    'workplane':   re.compile(r'\.workplane\b'),
    'transformed': re.compile(r'\.transformed\b'),
    'sketch':      re.compile(r'\.sketch\b'),
    'rotate':      re.compile(r'\.rotate\b'),
    'translate':   re.compile(r'\.translate\b'),
}
OP_NAMES = list(OPS_PATTERNS.keys())

# Mix used in big_bench_shell_50k.yaml
MIX_WEIGHTS = {
    'benchcad':         4,
    'cad-iso-106':      4,
    'benchcad-simple':  3,
    'cad-recode-bench': 2,
    'text2cad-bench':   1,
}

# Source path mapping
SOURCE_PATHS = {
    'benchcad':         '/home/ubuntu/cadrille/data/benchcad',
    'cad-iso-106':      '/home/ubuntu/cadrille/data/cad-iso-106',
    'benchcad-simple':  '/home/ubuntu/cadrille/data/benchcad-simple',
    'cad-recode-bench': '/home/ubuntu/cadrille/data/cad-recode-bench',
    'text2cad-bench':   '/home/ubuntu/cadrille/data/text2cad-bench',
}


def sample_codes_from_pkl(root: str, n: int) -> tuple[list[str], list[str], int]:
    """Return (codes, families, total_count) sampled from train.pkl."""
    pkl_path = Path(root) / 'train.pkl'
    if not pkl_path.exists():
        return [], [], 0
    with pkl_path.open('rb') as f:
        rows = pickle.load(f)
    total = len(rows)
    rng = np.random.default_rng(42)
    indices = rng.choice(total, size=min(n, total), replace=False)
    codes, families = [], []
    for i in indices:
        r = rows[int(i)]
        # text2cad-bench has description not py_path
        if 'py_path' in r:
            full = Path(root) / r['py_path']
        elif 'cadquery_py_path' in r:
            full = Path(root) / r['cadquery_py_path']
        elif Path(root, 'cadquery', f"{r.get('uid','')}.py").exists():
            full = Path(root) / 'cadquery' / f"{r['uid']}.py"
        else:
            full = None
        if full and full.exists():
            try:
                codes.append(full.read_text())
            except Exception:
                pass
        # Family from uid
        uid = r.get('uid', '')
        if uid.startswith('synth_'):
            parts = uid.split('_')
            fam = '_'.join(parts[1:3]) if len(parts) >= 3 else parts[1]
        elif uid.startswith('simple_'):
            parts = uid.rsplit('_', 1)
            fam = parts[0]
        elif '_' in uid:
            fam = uid.rsplit('_', 1)[0]
        else:
            fam = uid
        families.append(fam)
    return codes, families, total


def code_stats(codes: list[str]) -> dict:
    """Return op_freq, multi_line_frac, len_p10/50/90."""
    if not codes:
        return {'n_sample': 0}
    lens = [len(c) for c in codes if c]
    multi_line = sum(1 for c in codes if c and 'result = (' in c) / len(codes)
    op_freq = {}
    for op, pat in OPS_PATTERNS.items():
        op_freq[op] = sum(1 for c in codes if pat.search(c or '')) / len(codes)
    lens_sorted = sorted(lens)
    return {
        'n_sample': len(codes),
        'multi_line_frac': multi_line,
        'len_p10': lens_sorted[int(0.1 * len(lens_sorted))] if lens_sorted else 0,
        'len_p50': lens_sorted[int(0.5 * len(lens_sorted))] if lens_sorted else 0,
        'len_p90': lens_sorted[int(0.9 * len(lens_sorted))] if lens_sorted else 0,
        'op_freq': op_freq,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='eval_outputs/training_data_audit')
    ap.add_argument('--sample-per', type=int, default=5000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print('='*80, flush=True)
    print('TRAINING DATA AUDIT', flush=True)
    print('='*80, flush=True)
    results = {}
    for name, path in SOURCE_PATHS.items():
        print(f'\n[{name}] sampling {args.sample_per} from {path} ...', flush=True)
        codes, families, total = sample_codes_from_pkl(path, args.sample_per)
        if not codes:
            print(f'  ⚠️ no codes found at {path}')
            continue
        s = code_stats(codes)
        s['total_items'] = total
        s['top_families'] = Counter(families).most_common(10)
        s['family_count'] = len(set(families))
        results[name] = s
        print(f'  total={total}  sampled={s["n_sample"]}  multi_line={s["multi_line_frac"]:.0%}  '
              f'len p50={s["len_p50"]}', flush=True)

    # -------- Plot 1: op coverage heatmap (raw, per-source) --------
    fig, ax = plt.subplots(figsize=(10, max(6, len(OP_NAMES) * 0.3)))
    M = np.array([[results[n]['op_freq'][op] if n in results else 0 for n in MIX_WEIGHTS] for op in OP_NAMES])
    im = ax.imshow(M, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(MIX_WEIGHTS)))
    ax.set_xticklabels([f'{n}\n(w={w}, {results[n]["total_items"]/1000:.0f}k)' if n in results else n
                        for n, w in MIX_WEIGHTS.items()], rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(len(OP_NAMES))); ax.set_yticklabels(OP_NAMES)
    for i in range(len(OP_NAMES)):
        for j in range(len(MIX_WEIGHTS)):
            v = M[i, j]
            if v > 0.001:
                ax.text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=8,
                         color='white' if v > 0.5 else 'black')
    ax.set_title('Op coverage per dataset (% of items containing op)')
    fig.colorbar(im, ax=ax, label='fraction')
    fig.tight_layout()
    fig.savefig(out / 'op_coverage.png', dpi=130)
    plt.close(fig)
    print(f'  → op_coverage.png')

    # -------- Plot 2: weighted op exposure --------
    # Per-step probability of seeing op = sum_i (weight_i / total_w) * op_freq_i
    total_w = sum(MIX_WEIGHTS.values())
    weighted = {}
    for op in OP_NAMES:
        weighted[op] = sum(
            (MIX_WEIGHTS[n] / total_w) * results[n]['op_freq'][op]
            for n in MIX_WEIGHTS if n in results
        )
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_ops = sorted(OP_NAMES, key=lambda o: -weighted[o])
    vals = [weighted[o] for o in sorted_ops]
    colors = ['#d62728' if v < 0.05 else '#ff7f0e' if v < 0.20 else '#2ca02c' for v in vals]
    ax.barh(range(len(sorted_ops)), vals, color=colors)
    ax.set_yticks(range(len(sorted_ops))); ax.set_yticklabels(sorted_ops)
    ax.set_xlabel('per-sample probability (weighted by mix)')
    ax.set_title(f'Weighted op exposure — big_bench_shell_50k mix\n'
                  f'red <5% (rare/no-data), orange <20% (low), green ≥20% (well-covered)')
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / 'op_coverage_weighted.png', dpi=130)
    plt.close(fig)
    print(f'  → op_coverage_weighted.png')

    # -------- Plot 3: code-length distribution --------
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4), sharey=True)
    for i, (n, s) in enumerate(results.items()):
        ax = axes[i] if len(results) > 1 else axes
        # Resample lengths
        codes, _, _ = sample_codes_from_pkl(SOURCE_PATHS[n], 2000)
        lens = [len(c) for c in codes if c]
        ax.hist(lens, bins=40, color='#1f77b4', alpha=0.8)
        ax.axvline(1000, color='red', ls='--', alpha=0.5, label='max_code_len')
        ax.set_title(f'{n}\n(p50={s["len_p50"]}, p90={s["len_p90"]})', fontsize=9)
        ax.set_xlabel('code length (chars)')
        ax.legend(fontsize=8)
    fig.suptitle('Code length distribution per dataset (red line = max_code_len=1000 cutoff)')
    fig.tight_layout()
    fig.savefig(out / 'code_len_dist.png', dpi=130)
    plt.close(fig)
    print(f'  → code_len_dist.png')

    # -------- Plot 4: family diversity --------
    fig, axes = plt.subplots(1, len(results), figsize=(18, 5))
    for i, (n, s) in enumerate(results.items()):
        ax = axes[i] if len(results) > 1 else axes
        top = s['top_families']
        if top:
            fams = [f[0][:20] for f in top]
            counts = [f[1] for f in top]
            ax.barh(range(len(fams)), counts, color='#2ca02c')
            ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=8)
            ax.set_title(f'{n}\n({s["family_count"]} unique families)', fontsize=9)
            ax.invert_yaxis()
    fig.suptitle('Top 10 families per dataset')
    fig.tight_layout()
    fig.savefig(out / 'family_distribution.png', dpi=130)
    plt.close(fig)
    print(f'  → family_distribution.png')

    # -------- Markdown summary --------
    md = ['# Training Data Audit (big_bench_shell_50k mix)\n\n']
    md.append('## Dataset Inventory\n\n')
    md.append('| dataset | items | weight | step % | sampled | p50 len | multi-line | families |\n')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for n, w in MIX_WEIGHTS.items():
        if n not in results: continue
        s = results[n]
        pct = w / total_w * 100
        md.append(f'| **{n}** | {s["total_items"]:,} | {w} | {pct:.1f}% | '
                  f'{s["n_sample"]} | {s["len_p50"]} | {s["multi_line_frac"]:.0%} | '
                  f'{s["family_count"]} |\n')

    md.append('\n## Weighted Op Exposure (per training step)\n\n')
    md.append('Probability an op token appears in a randomly drawn training sample.\n\n')
    md.append('| op | weighted % | concern level |\n|---|---:|---|\n')
    for op in sorted(OP_NAMES, key=lambda o: -weighted[o]):
        v = weighted[op]
        if v > 0.20: level = '🟢 well-covered'
        elif v > 0.05: level = '🟡 low'
        elif v > 0.001: level = '🟠 very rare'
        else: level = '🔴 NO DATA — model will never produce'
        md.append(f'| **{op}** | {v:.2%} | {level} |\n')

    md.append('\n## Critical findings\n\n')
    no_data_ops = [op for op in OP_NAMES if weighted[op] < 0.001]
    md.append(f'**🔴 Ops with ZERO training data ({len(no_data_ops)}):** '
              f'{", ".join(no_data_ops)}\n\n')
    md.append('  → Model will have **0% recall on these ops** in evaluation.\n')
    md.append('  → To fix: synthesize new training data with these ops, '
              'or accept these gaps.\n\n')

    rare_ops = [op for op in OP_NAMES if 0.001 <= weighted[op] < 0.05]
    md.append(f'**🟠 Rare ops (<5% per-step exposure, {len(rare_ops)}):** '
              f'{", ".join(rare_ops)}\n\n')
    md.append('  → Likely <20% recall during eval unless weighted CE loss applied (T11).\n\n')

    md.append('## Per-source op deltas (where each dataset is unique)\n\n')
    md.append('| op | only-in-source(s) |\n|---|---|\n')
    for op in OP_NAMES:
        sources_with = [n for n in results if results[n]['op_freq'][op] >= 0.05]
        if 1 <= len(sources_with) <= 2:  # only 1-2 sources have this op
            md.append(f'| {op} | {", ".join(sources_with)} ({weighted[op]:.1%} weighted) |\n')

    md.append('\n## Figures\n\n')
    md.append('![op coverage per dataset](op_coverage.png)\n\n')
    md.append('![weighted op exposure](op_coverage_weighted.png)\n\n')
    md.append('![code length](code_len_dist.png)\n\n')
    md.append('![family distribution](family_distribution.png)\n\n')

    (out / 'summary.md').write_text(''.join(md))
    print(f'\n→ summary.md saved at {out / "summary.md"}', flush=True)
    print(f'\nDONE — see {out}/', flush=True)


if __name__ == '__main__':
    main()
