"""§7 main + appendix figures — single source of truth.

Produces 8 single-panel 4-line figures from per-step prediction JSONLs:

  Main:
    fig_7_4line_ess_pass.png   §7.a  BC val OOD essential_pass
    fig_7_ood_iou_4line.png    §7.b  BC val OOD IoU

  Appendix:
    fig_app_ood_exec.png             BC val OOD exec rate
    fig_app_iid_ess_pass.png         BC val IID essential_pass
    fig_app_iid_iou.png              BC val IID IoU
    fig_app_iid_exec.png             BC val IID exec rate
    fig_app_deepcad_iou.png          DeepCAD test IoU
    fig_app_deepcad_exec.png         DeepCAD test exec rate
    fig_app_fusion360_iou.png        Fusion360 test IoU
    fig_app_fusion360_exec.png       Fusion360 test exec rate

Reads 4 trained / training runs:
  v3                 — IID ceiling (saw all families)
  ood_enhance (v4)   — holdout + benchcad-easy supplement
  baseline           — HQ only (no benchcad-style)
  ood                — holdout, no benchcad-easy   (loads when ckpt dir appears)
  iid                — no holdout, full data       (loads when ckpt dir appears)

Usage:
    uv run python -m scripts.analysis.plot_main_appendix
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.holdout import HOLDOUT_FAMILIES as HOLDOUT
from common.essential_ops import ESSENTIAL_BY_FAMILY as ESS_SPEC

# ------------------------------------------------------------------
# Run registry — add a new dict entry when a new chain run lands its
# first prediction JSONL. Order here drives legend order in figures.
# ------------------------------------------------------------------
RUNS = [
    {
        # v3 trained on all 106 BenchCAD families, so on the held-out families
        # it IS the IID line (those families are in-distribution for v3).
        # No separate matched-recipe iid run is needed — saves a 24h chain step.
        'key':   'iid_v3',
        'label': '(1) iid — v3 baseline (trained on all 106 families)',
        'dir':   '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions',
        'plot':  dict(color='C2', linewidth=2.0, marker='', linestyle='-', alpha=0.85),
    },
    {
        'key':   'ood_enhance',
        'label': '(3) ood_enhance — holdout + benchcad-easy (v4)',
        'dir':   '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/predictions',
        'plot':  dict(color='C0', linewidth=2.0, marker='s', markersize=5, linestyle='-', alpha=0.9),
    },
    {
        'key':   'baseline',
        'label': '(4) baseline — HQ only (no bench-stack)',
        'dir':   '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0501-0629/predictions',
        'plot':  dict(color='C4', linewidth=2.0, marker='^', markersize=5, linestyle='-', alpha=0.9),
    },
    {
        'key':   'ood',
        'label': '(2) ood — holdout, no benchcad-easy',
        'dir':   '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0501-1753/predictions',
        'plot':  dict(color='C3', linewidth=2.0, marker='D', markersize=5, linestyle='-', alpha=0.9),
    },
]

# Bucket spec: each fig is parameterised by which rows count for the metric.
BUCKETS = {
    'BC val OOD':     dict(bucket_prefix='BenchCAD val', family_filter='ood'),
    'BC val IID':     dict(bucket_prefix='BenchCAD val', family_filter='iid'),
    'DeepCAD test':   dict(bucket_exact='DeepCAD test'),
    'Fusion360 test': dict(bucket_exact='Fusion360 test'),
}


def load_setup():
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    tax = yaml.safe_load(open(REPO_ROOT / 'configs/eval/op_taxonomy.yaml'))
    patterns = {n: re.compile(p) for n, p in tax['patterns'].items()}
    return uid2fam, patterns


def find_ops(code, patterns):
    if not code:
        return set()
    out = {n for n, p in patterns.items() if p.search(code)}
    if 'sweep' in out and 'helix' in out:
        out.add('sweep+helix')
    return out


def ess_pass(family, ops):
    spec = ESS_SPEC.get(family)
    if not spec:
        return None
    for elem in spec:
        if isinstance(elem, str):
            if elem not in ops:
                return False
        else:
            if not any(o in ops for o in elem):
                return False
    return True


def select_rows(rows, bucket_cfg, uid2fam):
    out = []
    for r in rows:
        b = r.get('bucket') or ''
        if 'bucket_exact' in bucket_cfg:
            if b != bucket_cfg['bucket_exact']:
                continue
        else:
            if not b.startswith(bucket_cfg['bucket_prefix']):
                continue
            fam = uid2fam.get(r['uid'])
            if bucket_cfg['family_filter'] == 'ood' and fam not in HOLDOUT:
                continue
            if bucket_cfg['family_filter'] == 'iid' and (fam is None or fam in HOLDOUT):
                continue
        out.append(r)
    return out


def metric_value(rows, metric, uid2fam, patterns):
    """Compute a single scalar for one step + bucket. Returns None if no data."""
    if not rows:
        return None
    if metric == 'iou':
        vals = [max(r.get('iou') or 0, 0) for r in rows]
        return float(np.mean(vals))
    if metric == 'exec':
        vals = [(r.get('iou') is not None and r['iou'] >= 0) for r in rows]
        return float(np.mean(vals))
    if metric == 'ess_pass':
        vals = []
        for r in rows:
            fam = uid2fam.get(r['uid'])
            if fam is None:
                continue
            ops = find_ops(r.get('pred_code') or '', patterns)
            e = ess_pass(fam, ops)
            if e is not None:
                vals.append(1 if e else 0)
        return float(np.mean(vals)) if vals else None
    raise ValueError(f'unknown metric {metric!r}')


def per_step(pred_dir, bucket_cfg, metric, uid2fam, patterns):
    out = {}
    p = Path(pred_dir)
    if not p.is_dir():
        return out
    for f in sorted(p.glob('step-*.jsonl')):
        if '.max@' in f.name:
            continue
        try:
            step = int(f.stem.replace('step-', ''))
        except ValueError:
            continue
        if step % 1000 != 0 or step == 0:
            continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        sub = select_rows(rows, bucket_cfg, uid2fam)
        v = metric_value(sub, metric, uid2fam, patterns)
        if v is not None:
            out[step] = v
    return out


# ------------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------------

def plot_one(bucket_name, metric, ylabel, title, outpath, ymax_hint=None,
             uid2fam=None, patterns=None):
    bucket_cfg = BUCKETS[bucket_name]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    any_data = False
    max_step = 0
    for run in RUNS:
        d = per_step(run['dir'], bucket_cfg, metric, uid2fam, patterns)
        if not d:
            continue
        any_data = True
        steps = sorted(d)
        max_step = max(max_step, max(steps))
        ax.plot(steps, [d[s] for s in steps],
                label=f'{run["label"]} (n={len(steps)})',
                **run['plot'])

    placeholder_x = np.arange(1000, max(max_step, 50000) + 1, 1000)
    for run in RUNS:
        d = per_step(run['dir'], bucket_cfg, metric, uid2fam, patterns)
        if d:
            continue  # already plotted real data
        ax.plot(placeholder_x, [0] * len(placeholder_x),
                color=run['plot']['color'], linestyle='--', linewidth=1.2,
                alpha=0.4, label=f'{run["label"]} [TBD]')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('training step', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if ymax_hint:
        ax.set_ylim(-0.02, ymax_hint)
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    if metric in ('ess_pass', 'exec'):
        ax.axhline(1.0, color='gray', lw=0.5, linestyle=':')
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    if any_data:
        print(f'  wrote {outpath.name} ({outpath.stat().st_size//1024} KB)')
    else:
        print(f'  wrote {outpath.name} ({outpath.stat().st_size//1024} KB) — NO real data yet')


def main():
    uid2fam, patterns = load_setup()
    fig_dir = REPO_ROOT / 'paper/figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print('=== Main figures ===')
    plot_one('BC val OOD', 'ess_pass', 'essential_pass rate (mean)',
             '§7.a essential_pass rate on held-out families (BC val, OOD)',
             fig_dir / 'fig_7_4line_ess_pass.png',
             uid2fam=uid2fam, patterns=patterns)
    plot_one('BC val OOD', 'iou', 'IoU (mean over OOD samples)',
             '§7.b OOD IoU vs training step (4-line comparison)',
             fig_dir / 'fig_7_ood_iou_4line.png',
             uid2fam=uid2fam, patterns=patterns)

    print('=== Appendix — OOD complement ===')
    plot_one('BC val OOD', 'exec', 'exec rate',
             'Appendix — OOD exec rate (BC val, held-out families)',
             fig_dir / 'fig_app_ood_exec.png',
             uid2fam=uid2fam, patterns=patterns)

    print('=== Appendix — IID control ===')
    plot_one('BC val IID', 'ess_pass', 'essential_pass rate (mean)',
             'Appendix — IID essential_pass control (BC val, seen families)',
             fig_dir / 'fig_app_iid_ess_pass.png',
             uid2fam=uid2fam, patterns=patterns)
    plot_one('BC val IID', 'iou', 'IoU (mean over IID samples)',
             'Appendix — IID IoU control (BC val, seen families)',
             fig_dir / 'fig_app_iid_iou.png',
             uid2fam=uid2fam, patterns=patterns)
    plot_one('BC val IID', 'exec', 'exec rate',
             'Appendix — IID exec rate (BC val, seen families)',
             fig_dir / 'fig_app_iid_exec.png',
             uid2fam=uid2fam, patterns=patterns)

    print('=== Appendix — external benchmarks ===')
    plot_one('DeepCAD test', 'iou', 'IoU (mean over n=50)',
             'Appendix — DeepCAD test IoU',
             fig_dir / 'fig_app_deepcad_iou.png',
             uid2fam=uid2fam, patterns=patterns)
    plot_one('DeepCAD test', 'exec', 'exec rate',
             'Appendix — DeepCAD test exec rate',
             fig_dir / 'fig_app_deepcad_exec.png',
             uid2fam=uid2fam, patterns=patterns)
    plot_one('Fusion360 test', 'iou', 'IoU (mean over n=50)',
             'Appendix — Fusion360 test IoU',
             fig_dir / 'fig_app_fusion360_iou.png',
             uid2fam=uid2fam, patterns=patterns)
    plot_one('Fusion360 test', 'exec', 'exec rate',
             'Appendix — Fusion360 test exec rate',
             fig_dir / 'fig_app_fusion360_exec.png',
             uid2fam=uid2fam, patterns=patterns)


if __name__ == '__main__':
    main()
