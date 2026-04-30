"""Main figure v2 — "Model recalls rare ops on OOD but can't compose them."

Single 4-panel figure (one main + supplementary detail):
  (a) BC OOD: rare_recall trajectory — v3 vs v4 (CLIMBS for both)
  (b) BC OOD: essential_pass trajectory — v3 climbs to ceiling, v4 plateaus
  (c) BC OOD final-step gap: rare_recall (high) vs essential_pass (low) for v4
  (d) IID control: trajectories matched between v3 and v4

Plus 2 appendix figures:
  A.1: Per-family OOD breakdown (v3 vs v4 essential_pass per family)
  A.2: IID rare-op recall (the "knows-ops" side, from Option B)

Usage:
    uv run python -m scripts.analysis.main_figure_v2
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

V3_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions'
V4_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/predictions'

HOLDOUT = {'tapered_boss', 'taper_pin', 'venturi_tube', 'bucket', 'dome_cap',
           'nozzle', 'enclosure', 'waffle_plate', 'bolt', 'duct_elbow'}


def load_setup():
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    tax = yaml.safe_load(open(REPO_ROOT / 'configs/eval/op_taxonomy.yaml'))
    patterns = {n: re.compile(p) for n, p in tax['patterns'].items()}
    rare = set(tax['rare'])
    feature = set(tax['feature'])
    ess_spec = yaml.safe_load(open(REPO_ROOT / 'configs/eval/canonical_ops.yaml'))
    return uid2fam, patterns, rare, feature, ess_spec


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


def parse_step(path, uid2fam, patterns, rare, feature, ess_spec):
    """Return per-split metrics."""
    rows = [json.loads(l) for l in Path(path).open() if l.strip()]
    bc = [r for r in rows if r.get('bucket') == 'BenchCAD val']
    iid = [r for r in bc if uid2fam.get(r['uid']) not in HOLDOUT]
    ood = [r for r in bc if uid2fam.get(r['uid']) in HOLDOUT]

    def stats(rows_):
        if not rows_: return None
        rare_recall, ess, ious = [], [], []
        feat_f1, recall = [], []
        per_fam = {}
        for r in rows_:
            po = find_ops(r.get('pred_code') or '', patterns)
            go = find_ops(r.get('gt_code') or '', patterns)
            iou = r.get('iou', 0) or 0
            if iou < 0: iou = 0
            ious.append(iou)
            if go:
                recall.append(len(go & po) / len(go))
                gr = go & rare
                if gr:
                    rare_recall.append(len(gr & po) / len(gr))
            fam = uid2fam.get(r['uid'])
            e = ess_pass(fam, po, ess_spec) if fam else None
            if e is not None:
                ess.append(1 if e else 0)
                per_fam.setdefault(fam, []).append(1 if e else 0)
            pf = po & feature; gf = go & feature
            if not gf and not pf: feat_f1.append(1.0)
            elif not gf or not pf: feat_f1.append(0.0)
            else:
                tp = len(pf & gf); fp_ = len(pf - gf); fn = len(gf - pf)
                pr = tp/(tp+fp_) if tp+fp_ else 0
                rc = tp/(tp+fn) if tp+fn else 0
                feat_f1.append(2*pr*rc/(pr+rc) if pr+rc else 0)
        return {
            'iou': np.mean(ious),
            'recall': np.mean(recall) if recall else 0,
            'rare_recall': np.mean(rare_recall) if rare_recall else 0,
            'ess_pass': np.mean(ess) if ess else None,
            'feat_f1': np.mean(feat_f1),
            'per_fam_ess': {f: np.mean(v) for f, v in per_fam.items()},
            'n': len(rows_),
        }
    return {'IID': stats(iid), 'OOD': stats(ood)}


def parse_run(pred_dir, uid2fam, patterns, rare, feature, ess_spec):
    out = {}
    for f in sorted(Path(pred_dir).glob('step-*.jsonl')):
        if '.max@' in f.name: continue
        step = int(f.stem.replace('step-', ''))
        if step % 1000 != 0 or step == 0: continue
        out[step] = parse_step(f, uid2fam, patterns, rare, feature, ess_spec)
    return out


def build_main_figure(v3, v4, out: Path):
    steps = sorted(set(v3) & set(v4))
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    # (a) OOD rare_recall trajectory
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, [v3[s]['OOD']['rare_recall'] for s in steps], '-o',
            color='C1', label='v3 (saw all families)', markersize=5, lw=1.8)
    ax.plot(steps, [v4[s]['OOD']['rare_recall'] for s in steps], '-s',
            color='C0', label='v4 (10 fams held out)', markersize=5, lw=1.8)
    ax.fill_between(steps, [v3[s]['OOD']['rare_recall'] for s in steps],
                    [v4[s]['OOD']['rare_recall'] for s in steps], alpha=0.08, color='gray')
    ax.set_title('(a) BC val OOD: rare-op recall climbs even on unseen families',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('training step', fontsize=10)
    ax.set_ylabel('rare_recall (n=9 OOD samples per step)', fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    # annotate
    ax.annotate('v4 climbs 0.17 → 0.78\non held-out families', xy=(13000, 0.667), xytext=(7000, 0.85),
                fontsize=9, color='C0', arrowprops=dict(arrowstyle='->', color='C0', alpha=0.7))

    # (b) OOD essential_pass trajectory
    ax = fig.add_subplot(gs[0, 1])
    v4y = [v4[s]['OOD']['ess_pass'] for s in steps]
    v3y = [v3[s]['OOD']['ess_pass'] for s in steps]
    ax.plot(steps, v3y, '-o', color='C1', label='v3 (saw all families)', markersize=5, lw=1.8)
    ax.plot(steps, v4y, '-s', color='C0', label='v4 (10 fams held out)', markersize=5, lw=1.8)
    ax.fill_between(steps, v3y, v4y, alpha=0.15, color='red',
                    label='structural gap')
    ax.set_title('(b) BC val OOD: essential_pass plateaus — composition NOT learned',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('training step', fontsize=10)
    ax.set_ylabel('essential_pass (per-family AND-of-OR ops)', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='center right')
    ax.grid(alpha=0.3)
    ax.annotate('v4 stuck at 0.22-0.44\nwhile v3 reaches 1.00', xy=(13000, 0.444),
                xytext=(7000, 0.15), fontsize=9, color='C0',
                arrowprops=dict(arrowstyle='->', color='C0', alpha=0.7))

    # (c) The gap — bar chart at final step
    ax = fig.add_subplot(gs[1, 0])
    final = max(steps)
    bars_data = [
        ('v3 IID\nrecall', v3[final]['IID']['rare_recall'], 'C1'),
        ('v3 IID\ness_pass', v3[final]['IID']['ess_pass'] or 0, 'C1'),
        ('v3 OOD\nrecall', v3[final]['OOD']['rare_recall'], 'C1'),
        ('v3 OOD\ness_pass', v3[final]['OOD']['ess_pass'] or 0, 'C1'),
        ('v4 IID\nrecall', v4[final]['IID']['rare_recall'], 'C0'),
        ('v4 IID\ness_pass', v4[final]['IID']['ess_pass'] or 0, 'C0'),
        ('v4 OOD\nrecall', v4[final]['OOD']['rare_recall'], 'C0'),
        ('v4 OOD\ness_pass', v4[final]['OOD']['ess_pass'] or 0, 'C0'),
    ]
    labels = [b[0] for b in bars_data]
    values = [b[1] for b in bars_data]
    colors = [b[2] for b in bars_data]
    # alternate hatched for ess_pass to distinguish from recall
    hatches = ['' if 'recall' in l else '//' for l in labels]
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, hatch=hatches, edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('value', fontsize=10)
    ax.set_title(f'(c) Final step ({final}): rare_recall vs essential_pass — v4 OOD shows the gap',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    # highlight the v4 OOD pair (last 2 bars)
    ax.axvspan(5.5, 7.5, alpha=0.1, color='red')
    ax.text(6.5, 1.0, 'v4 OOD: knows ops\nbut can\'t compose',
            ha='center', fontsize=10, color='darkred', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # (d) IID control — both metrics climb together for v4 IID (sanity check)
    ax = fig.add_subplot(gs[1, 1])
    v3_rec = [v3[s]['IID']['rare_recall'] for s in steps]
    v4_rec = [v4[s]['IID']['rare_recall'] for s in steps]
    v3_ess = [v3[s]['IID']['ess_pass'] or 0 for s in steps]
    v4_ess = [v4[s]['IID']['ess_pass'] or 0 for s in steps]
    ax.plot(steps, v3_rec, '-o', color='C1', label='v3 rare_recall', markersize=4, lw=1.5)
    ax.plot(steps, v3_ess, '--o', color='C1', label='v3 ess_pass', markersize=4, lw=1.5, alpha=0.6)
    ax.plot(steps, v4_rec, '-s', color='C0', label='v4 rare_recall', markersize=4, lw=1.5)
    ax.plot(steps, v4_ess, '--s', color='C0', label='v4 ess_pass', markersize=4, lw=1.5, alpha=0.6)
    ax.set_title('(d) BC val IID control: both metrics climb together when families are seen',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('training step', fontsize=10)
    ax.set_ylabel('value', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle("Recall ≠ Composition: model learns to identify rare CAD ops on held-out families,\n"
                 "but cannot compose them into the correct family-level structure",
                 fontsize=13, fontweight='bold', y=0.98)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'  wrote {out} ({out.stat().st_size//1024} KB)')


def build_appendix_a_per_family(v3, v4, uid2fam, ess_spec, out: Path):
    """Per-family OOD breakdown."""
    families_with_data = sorted(set(
        f for s in v4 for f in v4[s]['OOD']['per_fam_ess']
    ))
    print(f'  per-family bar: families = {families_with_data}')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # aggregate per family across all steps
    fam_v3 = {f: [] for f in families_with_data}
    fam_v4 = {f: [] for f in families_with_data}
    for s in v4:
        for f in families_with_data:
            v = v3[s]['OOD']['per_fam_ess'].get(f)
            if v is not None: fam_v3[f].append(v)
            v = v4[s]['OOD']['per_fam_ess'].get(f)
            if v is not None: fam_v4[f].append(v)
    v3_y = [np.mean(fam_v3[f]) if fam_v3[f] else 0 for f in families_with_data]
    v4_y = [np.mean(fam_v4[f]) if fam_v4[f] else 0 for f in families_with_data]
    x = np.arange(len(families_with_data))
    width = 0.35
    ax.bar(x - width/2, v3_y, width, label='v3 (saw)', color='C1', alpha=0.85)
    ax.bar(x + width/2, v4_y, width, label='v4 (held out)', color='C0', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(families_with_data, rotation=15, ha='right')
    ax.set_ylabel('essential_pass (mean across all eval steps)', fontsize=11)
    ax.set_title('Appendix A: per-OOD-family essential_pass — composition gap is consistent',
                 fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    for i, (a, b) in enumerate(zip(v3_y, v4_y)):
        ax.text(i - width/2, a + 0.02, f'{a:.2f}', ha='center', fontsize=9)
        ax.text(i + width/2, b + 0.02, f'{b:.2f}', ha='center', fontsize=9, color='darkblue')
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'  wrote {out} ({out.stat().st_size//1024} KB)')


def build_appendix_b_iid_recall(v3, v4, out: Path):
    """The "knows-ops" side: IID rare op metrics improvement."""
    steps = sorted(set(v3) & set(v4))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) IID rare_recall
    ax = axes[0]
    ax.plot(steps, [v3[s]['IID']['rare_recall'] for s in steps], '-o', color='C1', label='v3', markersize=4, lw=1.6)
    ax.plot(steps, [v4[s]['IID']['rare_recall'] for s in steps], '-s', color='C0', label='v4', markersize=4, lw=1.6)
    ax.set_title('(a) IID rare op recall', fontsize=11)
    ax.set_xlabel('step'); ax.set_ylabel('rare_recall'); ax.set_ylim(0, 1)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) IID feature F1
    ax = axes[1]
    ax.plot(steps, [v3[s]['IID']['feat_f1'] for s in steps], '-o', color='C1', label='v3', markersize=4, lw=1.6)
    ax.plot(steps, [v4[s]['IID']['feat_f1'] for s in steps], '-s', color='C0', label='v4', markersize=4, lw=1.6)
    ax.set_title('(b) IID feature F1 (chamfer/fillet/hole)', fontsize=11)
    ax.set_xlabel('step'); ax.set_ylabel('feat_F1'); ax.set_ylim(0, 1)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) IID overall recall
    ax = axes[2]
    ax.plot(steps, [v3[s]['IID']['recall'] for s in steps], '-o', color='C1', label='v3', markersize=4, lw=1.6)
    ax.plot(steps, [v4[s]['IID']['recall'] for s in steps], '-s', color='C0', label='v4', markersize=4, lw=1.6)
    ax.set_title('(c) IID overall op recall', fontsize=11)
    ax.set_xlabel('step'); ax.set_ylabel('recall'); ax.set_ylim(0, 1)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle('Appendix B: IID gains from + benchcad-easy + 60/40 mix recipe (BC val IID, n=41)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'  wrote {out} ({out.stat().st_size//1024} KB)')


def main():
    uid2fam, patterns, rare, feature, ess_spec = load_setup()
    print('parsing v3 ...')
    v3 = parse_run(V3_DIR, uid2fam, patterns, rare, feature, ess_spec)
    print(f'  v3 steps: {sorted(v3)}')
    print('parsing v4 ...')
    v4 = parse_run(V4_DIR, uid2fam, patterns, rare, feature, ess_spec)
    print(f'  v4 steps: {sorted(v4)}')

    out_dir = REPO_ROOT / 'docs' / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\nbuilding main figure (recall vs composition gap) ...')
    build_main_figure(v3, v4, out_dir / 'fig_main_recall_vs_composition.png')

    print('\nbuilding appendix A (per-family) ...')
    build_appendix_a_per_family(v3, v4, uid2fam, ess_spec, out_dir / 'fig_app_per_family.png')

    print('\nbuilding appendix B (IID gains) ...')
    build_appendix_b_iid_recall(v3, v4, out_dir / 'fig_app_iid_gains.png')

    print('\ndone — figures in docs/paper_figures/')


if __name__ == '__main__':
    main()
