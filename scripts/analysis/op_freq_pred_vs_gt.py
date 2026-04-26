"""Compare predicted op frequencies vs GT for the 100-case sweep.

For each of 7 ckpts × 33 BenchCAD val anchors, count which ops appear in the
predicted code. Compare against GT op distribution. Highlights where the
model under-/over-predicts. For DeepCAD/Fusion (no GT code), still report
predicted op freq trajectory.

Usage:
  uv run python -m scripts.analysis.op_freq_pred_vs_gt \
      --codes eval_outputs/trajectory_100case_curriculum/codes.jsonl \
      --anchors eval_outputs/trajectory_100case_curriculum/anchors.jsonl \
      --out    eval_outputs/op_freq_pred_vs_gt
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Match online_eval._OPS exactly (keep in lockstep)
OPS = {
    'box':         re.compile(r'\.box\b'),
    'cylinder':    re.compile(r'\.cylinder\b'),
    'sphere':      re.compile(r'\.sphere\b'),
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
    'moveTo':      re.compile(r'\.moveTo\b'),
    'translate':   re.compile(r'\.translate\b'),
    'rotate':      re.compile(r'\.rotate\b'),
    'sketch':      re.compile(r'\.sketch\b'),
}
OP_NAMES = list(OPS.keys())


def ops_present(code: str) -> set[str]:
    return {op for op, p in OPS.items() if p.search(code or '')}


def freq_in_corpus(codes: list[str]) -> dict[str, float]:
    """Fraction of codes that contain each op."""
    n = len(codes)
    if n == 0:
        return {op: 0.0 for op in OP_NAMES}
    out = {op: 0 for op in OP_NAMES}
    for c in codes:
        ops = ops_present(c)
        for op in ops:
            out[op] += 1
    return {op: cnt / n for op, cnt in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', required=True, help='codes.jsonl from 100-case sweep')
    ap.add_argument('--anchors', required=True, help='anchors.jsonl with gt_code')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 1. Load
    rows = [json.loads(l) for l in Path(args.codes).read_text().splitlines()]
    anchors = [json.loads(l) for l in Path(args.anchors).read_text().splitlines()]
    case_idx_to_anchor = {a['_case_idx']: a for a in anchors}
    print(f'Loaded {len(rows)} pred rows, {len(anchors)} anchors')

    # 2. Per-source: predicted ops freq per ckpt
    by_ds = defaultdict(lambda: defaultdict(list))  # dataset -> step -> list[code]
    for r in rows:
        by_ds[r['dataset']][r['step']].append(r['code'])
    steps = sorted({r['step'] for r in rows})
    print(f'Steps: {steps}')

    # 3. GT freq (BenchCAD val only — DeepCAD/Fusion have no GT code)
    bc_gt_codes = [a['gt_code'] for a in anchors
                    if a['dataset'] == 'benchcad_val' and a.get('gt_code')]
    gt_freq = freq_in_corpus(bc_gt_codes)
    print(f'BenchCAD GT freq computed over {len(bc_gt_codes)} cases')

    # 4. Predicted freq per ckpt × dataset
    pred_freq = {}  # (dataset, step) -> {op: freq}
    for ds in by_ds:
        for s in steps:
            pred_freq[(ds, s)] = freq_in_corpus(by_ds[ds][s])

    # 5. Build table for BenchCAD val: pred at each ckpt vs GT
    md = ['# Op frequency: predicted vs GT (BenchCAD val n=33)\n\n']
    md.append('Each cell = fraction of the 33 BenchCAD val cases that contain '
              'this op. Δ = pred - GT (positive → model over-uses; negative → '
              'model under-uses; lighter color = closer to GT).\n\n')

    # Sort ops by GT frequency (most common first)
    ops_sorted = sorted(OP_NAMES, key=lambda op: -gt_freq.get(op, 0))

    md.append('| op | GT |')
    for s in steps:
        md.append(f' step {s} | Δ{s} |')
    md.append('\n|---|---:|')
    for s in steps:
        md.append('---:|---:|')
    md.append('\n')
    for op in ops_sorted:
        gt = gt_freq.get(op, 0)
        row = f'| **{op}** | {gt:.2%} |'
        for s in steps:
            p = pred_freq.get(('benchcad_val', s), {}).get(op, 0)
            d = p - gt
            sign = '+' if d > 0 else ''
            row += f' {p:.2%} | {sign}{d:.2%} |'
        md.append(row + '\n')
    md.append('\n')

    # 6. Highlight under/over-predicted ops (compared at step 20000)
    final_step = steps[-1]
    deltas = [(op, pred_freq[('benchcad_val', final_step)].get(op, 0) - gt_freq.get(op, 0))
              for op in OP_NAMES]
    deltas.sort(key=lambda x: x[1])
    md.append(f'## Top 5 under-predicted ops at step {final_step} (model misses)\n\n')
    md.append('| op | GT freq | pred freq | Δ |\n|---|---:|---:|---:|\n')
    for op, d in deltas[:5]:
        gt = gt_freq[op]; p = pred_freq[('benchcad_val', final_step)].get(op, 0)
        md.append(f'| **{op}** | {gt:.2%} | {p:.2%} | {d:.2%} |\n')

    md.append(f'\n## Top 5 over-predicted ops at step {final_step} (model hallucinates)\n\n')
    md.append('| op | GT freq | pred freq | Δ |\n|---|---:|---:|---:|\n')
    for op, d in deltas[-5:][::-1]:
        gt = gt_freq[op]; p = pred_freq[('benchcad_val', final_step)].get(op, 0)
        md.append(f'| **{op}** | {gt:.2%} | {p:.2%} | +{d:.2%} |\n')

    # 7. Predicted-only freq trajectory for DeepCAD + Fusion
    md.append('\n# Predicted op freq trajectory — DeepCAD + Fusion (no GT code, n=33+34)\n\n')
    md.append('Trends: which ops the model produces on cad-recode-style inputs.\n\n')

    for ds, ds_label in [('deepcad_test', 'DeepCAD test n=33'),
                           ('fusion360_test', 'Fusion360 test n=34')]:
        md.append(f'## {ds_label}\n\n')
        md.append('| op |')
        for s in steps:
            md.append(f' step {s} |')
        md.append('\n|---|')
        for s in steps:
            md.append('---:|')
        md.append('\n')
        # Sort by max freq across ckpts (most-used ops first)
        ops_order = sorted(OP_NAMES,
                           key=lambda op: -max(pred_freq.get((ds, s), {}).get(op, 0) for s in steps))
        for op in ops_order:
            row = f'| {op} |'
            for s in steps:
                p = pred_freq.get((ds, s), {}).get(op, 0)
                row += f' {p:.2%} |'
            md.append(row + '\n')
        md.append('\n')

    # 8. Plot: bar chart, GT vs pred at step 20000 for BenchCAD val
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(ops_sorted))
    gt_vals = [gt_freq[op] for op in ops_sorted]
    p20_vals = [pred_freq[('benchcad_val', final_step)].get(op, 0) for op in ops_sorted]
    p11_vals = [pred_freq[('benchcad_val', 11000)].get(op, 0) for op in ops_sorted] if 11000 in steps else None
    width = 0.28
    ax.bar(x - width, gt_vals, width, label='GT (n=33)', color='#1f77b4')
    if p11_vals:
        ax.bar(x, p11_vals, width, label='pred @ step 11k (curriculum P3 start)',
                color='#ff7f0e', alpha=0.85)
    ax.bar(x + width, p20_vals, width, label=f'pred @ step {final_step}',
            color='#2ca02c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ops_sorted, rotation=60, ha='right', fontsize=9)
    ax.set_ylabel('frequency (fraction of 33 cases)')
    ax.set_title('BenchCAD val: GT op freq vs predicted op freq (curriculum 11k & 20k)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'pred_vs_gt_benchcad.png', dpi=130)
    plt.close(fig)
    print(f'  → pred_vs_gt_benchcad.png')

    # 9. Plot: per-ckpt heatmap of (op × step) freq for each dataset
    for ds, ds_label in [('benchcad_val', 'BenchCAD val'),
                          ('deepcad_test', 'DeepCAD test'),
                          ('fusion360_test', 'Fusion360 test')]:
        # Restrict to ops with non-trivial freq somewhere
        active = [op for op in OP_NAMES
                  if max(pred_freq.get((ds, s), {}).get(op, 0) for s in steps) > 0.05]
        if not active: continue
        mat = np.array([[pred_freq.get((ds, s), {}).get(op, 0) for s in steps] for op in active])
        fig, ax = plt.subplots(figsize=(8, max(4, len(active) * 0.32)))
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(steps))); ax.set_xticklabels([str(s) for s in steps])
        ax.set_yticks(range(len(active))); ax.set_yticklabels(active)
        ax.set_xlabel('step'); ax.set_title(f'{ds_label} — predicted op freq per ckpt')
        for i in range(len(active)):
            for j in range(len(steps)):
                v = mat[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                         color='white' if v < 0.4 else 'black', fontsize=7)
        fig.colorbar(im, ax=ax, label='freq')
        fig.tight_layout()
        fig.savefig(out / f'op_freq_heatmap_{ds}.png', dpi=130)
        plt.close(fig)
        print(f'  → op_freq_heatmap_{ds}.png')

    md.append('\n# Figures\n\n')
    md.append('![pred vs GT BenchCAD](pred_vs_gt_benchcad.png)\n\n')
    md.append('## Heatmaps per dataset\n\n')
    md.append('![BenchCAD heatmap](op_freq_heatmap_benchcad_val.png)\n\n')
    md.append('![DeepCAD heatmap](op_freq_heatmap_deepcad_test.png)\n\n')
    md.append('![Fusion360 heatmap](op_freq_heatmap_fusion360_test.png)\n\n')

    (out / 'report.md').write_text(''.join(md))
    print(f'  → report.md')

    # CSV export
    import csv
    with (out / 'op_freq.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'step', 'op', 'pred_freq', 'gt_freq', 'delta'])
        for ds in ('benchcad_val', 'deepcad_test', 'fusion360_test'):
            for s in steps:
                for op in OP_NAMES:
                    p = pred_freq.get((ds, s), {}).get(op, 0)
                    g = gt_freq.get(op, 0) if ds == 'benchcad_val' else None
                    d = (p - g) if g is not None else None
                    w.writerow([ds, s, op, f'{p:.4f}',
                                f'{g:.4f}' if g is not None else '',
                                f'{d:+.4f}' if d is not None else ''])
    print(f'  → op_freq.csv')


if __name__ == '__main__':
    main()
