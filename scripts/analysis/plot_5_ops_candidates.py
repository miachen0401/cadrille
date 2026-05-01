"""5 candidate ops-metric figures for §7 main Figure 1 — user picks one.

Each figure: 4-line plot (recipe variants) on OOD samples vs training step.
Lines: (1) IID ceiling, (2) OOD plain [TBD], (3) OOD+easy (current), (4) no-bench [TBD]

Candidates:
  A: rare_recall (OOD)
  B: essential_pass rate (OOD)
  C: rare_recall - essential_pass gap (OOD)  [recommended]
  D: op_entropy (OOD)
  E: feature_F1 (OOD)

Usage:
    uv run python -m scripts.analysis.plot_5_ops_candidates
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

V3_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions'
V4_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/predictions'

HOLDOUT = {'tapered_boss', 'taper_pin', 'venturi_tube', 'bucket', 'dome_cap',
           'nozzle', 'enclosure', 'waffle_plate', 'bolt', 'duct_elbow'}


def setup():
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    tax = yaml.safe_load(open(REPO_ROOT / 'configs/eval/op_taxonomy.yaml'))
    patterns = {n: re.compile(p) for n, p in tax['patterns'].items()}
    rare = set(tax['rare'])
    feat = set(tax['feature'])
    ess_spec = yaml.safe_load(open(REPO_ROOT / 'configs/eval/canonical_ops.yaml'))
    return uid2fam, patterns, rare, feat, ess_spec


def find_ops(code, patterns):
    if not code: return set()
    out = {n for n, p in patterns.items() if p.search(code)}
    if 'sweep' in out and 'helix' in out:
        out.add('sweep+helix')
    return out


def ess_pass(family, ops, spec):
    s = spec.get(family)
    if not s: return None
    for elem in s:
        if isinstance(elem, str):
            if elem not in ops: return False
        else:
            if not any(o in ops for o in elem): return False
    return True


def feature_f1(p, g, features):
    pf = p & features; gf = g & features
    if not gf and not pf: return 1.0
    if not gf or not pf: return 0.0
    tp = len(pf & gf); fp = len(pf - gf); fn = len(gf - pf)
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    return 2*pr*rc/(pr+rc) if pr+rc else 0


def op_entropy(rows, patterns):
    if not rows: return 0.0
    op_names = list(patterns); counts = np.zeros(len(op_names))
    for r in rows:
        ops = find_ops(r.get('pred_code') or '', patterns)
        for i, n in enumerate(op_names):
            if n in ops: counts[i] += 1
    if counts.sum() == 0: return 0.0
    p = counts/counts.sum(); p = p[p>0]
    return float(-(p * np.log(p)).sum())


def metrics_per_step(pred_dir, uid2fam, patterns, rare, feat, ess_spec, target_holdout):
    """Compute all 5 candidate metrics per step on the target subset."""
    out = {}
    for f in sorted(Path(pred_dir).glob('step-*.jsonl')):
        if '.max@' in f.name: continue
        step = int(f.stem.replace('step-', ''))
        if step % 1000 != 0 or step == 0: continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        if target_holdout:
            sub = [r for r in rows if r.get('bucket') == 'BenchCAD val'
                   and uid2fam.get(r['uid']) in HOLDOUT]
        else:
            sub = [r for r in rows if r.get('bucket') == 'BenchCAD val'
                   and uid2fam.get(r['uid']) not in HOLDOUT]
        if not sub: continue
        rare_recall, ess_pass_list, ff1 = [], [], []
        for r in sub:
            po = find_ops(r.get('pred_code') or '', patterns)
            go = find_ops(r.get('gt_code') or '', patterns)
            if go:
                gr = go & rare
                if gr:
                    rare_recall.append(len(gr & po) / len(gr))
            fam = uid2fam.get(r['uid'])
            e = ess_pass(fam, po, ess_spec) if fam else None
            if e is not None: ess_pass_list.append(1 if e else 0)
            ff1.append(feature_f1(po, go, feat))
        out[step] = {
            'rare_recall': float(np.mean(rare_recall)) if rare_recall else 0,
            'ess_pass':    float(np.mean(ess_pass_list)) if ess_pass_list else 0,
            'feat_f1':     float(np.mean(ff1)),
            'op_entropy':  op_entropy(sub, patterns),
        }
    return out


def main():
    uid2fam, patterns, rare, feat, ess_spec = setup()

    print('parsing v3 OOD ...')
    v3_ood = metrics_per_step(V3_DIR, uid2fam, patterns, rare, feat, ess_spec, target_holdout=True)
    print('parsing v3 IID ...')
    v3_iid = metrics_per_step(V3_DIR, uid2fam, patterns, rare, feat, ess_spec, target_holdout=False)
    print('parsing v4 OOD ...')
    v4_ood = metrics_per_step(V4_DIR, uid2fam, patterns, rare, feat, ess_spec, target_holdout=True)

    steps_v3 = sorted(v3_iid)
    steps_v4 = sorted(v4_ood)
    max_step = max(max(steps_v3), max(steps_v4))
    placeholder_x = np.arange(1000, max_step + 1, 1000)

    candidates = [
        ('A', 'rare_recall', 'rare op recall (OOD families)', (0, 1.05)),
        ('B', 'ess_pass', 'essential_pass rate (OOD families)', (0, 1.05)),
        ('C', 'recall_minus_ess', 'rare_recall - essential_pass gap (OOD families)', (-0.5, 1)),
        ('D', 'op_entropy', 'op entropy in nats (OOD families)', None),
        ('E', 'feat_f1', 'feature F1 (chamfer/fillet/hole, OOD families)', (0, 1.05)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax_idx, (letter, key, ylabel, ylim) in enumerate(candidates):
        ax = axes[ax_idx]

        if key == 'recall_minus_ess':
            v3_iid_y = [v3_iid[s]['rare_recall'] - v3_iid[s]['ess_pass'] for s in steps_v3]
            v4_y = [v4_ood[s]['rare_recall'] - v4_ood[s]['ess_pass'] for s in steps_v4]
            v3_ood_y = [v3_ood[s]['rare_recall'] - v3_ood[s]['ess_pass'] for s in sorted(v3_ood)]
        else:
            v3_iid_y = [v3_iid[s][key] for s in steps_v3]
            v4_y = [v4_ood[s][key] for s in steps_v4]
            v3_ood_y = [v3_ood[s][key] for s in sorted(v3_ood)]

        # 4-line + reference
        ax.plot(steps_v3, v3_iid_y, '-', color='C2', lw=2,
                label='(1) IID ceiling — v3', alpha=0.85)
        ax.plot(placeholder_x, [0]*len(placeholder_x), '--', color='C3', lw=1.3,
                label='(2) OOD plain — v4-holdout-noeasy [TBD]', alpha=0.5)
        ax.plot(steps_v4, v4_y, '-s', color='C0', lw=2, markersize=4,
                label='(3) OOD + easy — v4-holdout', alpha=0.9)
        ax.plot(placeholder_x, [0]*len(placeholder_x), '--', color='C4', lw=1.3,
                label='(4) no-bench — v4-hq-only [TBD]', alpha=0.5)
        ax.plot(sorted(v3_ood), v3_ood_y, ':', color='C1', lw=1.5,
                label='ref — v3 on holdout (saw them)', alpha=0.65)

        ax.set_title(f'Candidate {letter}: {ylabel}', fontsize=11, fontweight='bold')
        ax.set_xlabel('training step')
        ax.set_ylabel(ylabel.split('(')[0].strip())
        if ylim: ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        if key == 'recall_minus_ess':
            ax.axhline(0, color='gray', lw=0.5)
            ax.text(0.02, 0.95, 'higher = more "knows but can\'t compose"',
                    transform=ax.transAxes, fontsize=9, color='darkred')
        ax.legend(fontsize=7, loc='best')

    # Hide 6th subplot
    axes[-1].axis('off')
    axes[-1].text(0.05, 0.6, 'Pick which candidate goes into §7 main Figure 1.\n\n'
                              'A: rare_recall — "knows ops"\n'
                              'B: essential_pass — "composes correctly"\n'
                              'C: gap (A - B) — direct "recall ≠ composition" measure\n'
                              'D: op_entropy — diversity\n'
                              'E: feature_F1 — chamfer/fillet/hole accuracy\n\n'
                              'Recommended: C (single panel tells the whole story)',
                  fontsize=11, family='sans-serif')

    fig.suptitle('§7 Figure 1 candidates — pick one as main figure',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = REPO_ROOT / 'paper/figures/fig_7_ops_metric_candidates.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'wrote {out} ({out.stat().st_size//1024} KB)')


if __name__ == '__main__':
    main()
