"""Main figure for §4 dataset training probe — ops recall storyline.

5-panel figure:
  (a-c) trajectories: rare_recall / distinct_ops / feature_F1 (BC IID)
  (d)   per-rare-op recall bar chart at final step, v3 vs v4

Plus two appendix figures:
  A.1: OOD limitation (essential_pass trajectory + per-family bar)
  A.2: dataset op-combo scatter (BC vs DeepCAD vs Fusion360)

Usage:
    uv run python -m scripts.analysis.main_figure
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

V4_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/predictions'
V3_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions'

HOLDOUT = {'tapered_boss', 'taper_pin', 'venturi_tube', 'bucket', 'dome_cap',
           'nozzle', 'enclosure', 'waffle_plate', 'bolt', 'duct_elbow'}


def load_tax():
    tax = yaml.safe_load(open(REPO_ROOT / 'configs/eval/op_taxonomy.yaml'))
    return {
        'patterns': {n: re.compile(p) for n, p in tax['patterns'].items()},
        'rare': list(tax['rare']),  # ordered list for consistent plotting
        'feature': set(tax['feature']),
    }


def load_essentials():
    return yaml.safe_load(open(REPO_ROOT / 'configs/eval/canonical_ops.yaml'))


def find_ops(code, patterns):
    if not code: return set()
    out = {n for n, p in patterns.items() if p.search(code)}
    if 'sweep' in out and 'helix' in out:
        out.add('sweep+helix')
    return out


def essential_pass(family, ops, spec):
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


def parse_run(pred_dir, uid2fam, tax, ess_spec, max_step=None):
    """Per-step IID metrics aggregated."""
    out = {}  # step -> {iou, recall, rare_recall, ess_pass, feat_f1, distinct_ops, per_op_recall}
    rare_ops = tax['rare']
    for f in sorted(Path(pred_dir).glob('step-*.jsonl')):
        if '.max@' in f.name: continue
        step = int(f.stem.replace('step-', ''))
        if step % 1000 != 0 or step == 0: continue
        if max_step and step > max_step: continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        bc = [r for r in rows if r.get('bucket') == 'BenchCAD val']
        iid = [r for r in bc if uid2fam.get(r['uid']) not in HOLDOUT]
        ood = [r for r in bc if uid2fam.get(r['uid']) in HOLDOUT]

        def stats(rows_, label):
            if not rows_: return None
            ious, exec_n = [], 0
            recalls, rare_recalls = [], []
            ess_passes, feat_f1s = [], []
            distinct = set()
            # per-rare-op recall: for each rare op, when GT has it, did pred get it?
            per_op = {ro: {'gt_have': 0, 'pred_have': 0} for ro in rare_ops}
            for r in rows_:
                iou = r.get('iou', 0) or 0
                if iou < 0: iou = 0
                ious.append(iou)
                if r.get('has_iou') and (r.get('iou') or -1) >= 0: exec_n += 1
                po = find_ops(r.get('pred_code') or '', tax['patterns'])
                go = find_ops(r.get('gt_code') or '', tax['patterns'])
                distinct.update(po)
                if go:
                    recalls.append(len(go & po) / len(go))
                    gr = go & set(rare_ops)
                    if gr:
                        rare_recalls.append(len(gr & po) / len(gr))
                fam = uid2fam.get(r['uid'])
                ess = essential_pass(fam, po, ess_spec) if fam else None
                if ess is not None: ess_passes.append(1 if ess else 0)
                feat_f1s.append(feature_f1(po, go, tax['feature']))
                # per-op recall
                for ro in rare_ops:
                    if ro in go:
                        per_op[ro]['gt_have'] += 1
                        if ro in po:
                            per_op[ro]['pred_have'] += 1
            return {
                'iou': np.mean(ious),
                'exec_rate': exec_n / len(rows_) * 100,
                'recall': np.mean(recalls) if recalls else 0,
                'rare_recall': np.mean(rare_recalls) if rare_recalls else 0,
                'ess_pass': np.mean(ess_passes) if ess_passes else None,
                'feat_f1': np.mean(feat_f1s),
                'distinct_ops': len(distinct),
                'per_op_recall': {k: (v['pred_have']/v['gt_have'] if v['gt_have'] else None) for k, v in per_op.items()},
                'per_op_gt_n': {k: v['gt_have'] for k, v in per_op.items()},
                'n': len(rows_),
            }
        out[step] = {'IID': stats(iid, 'IID'), 'OOD': stats(ood, 'OOD')}
    return out


def build_main_figure(v3, v4, rare_ops, out: Path):
    """5-panel main figure."""
    steps = sorted(set(v3) & set(v4))
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.4], hspace=0.4, wspace=0.3)

    # (a) rare_recall trajectory
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, [v3[s]['IID']['rare_recall'] for s in steps], '-o', color='C1', label='v3 (baseline)', markersize=4, lw=1.6)
    ax.plot(steps, [v4[s]['IID']['rare_recall'] for s in steps], '-s', color='C0', label='v4 (+ rare-op data)', markersize=4, lw=1.6)
    ax.set_title('(a) rare op recall', fontsize=12)
    ax.set_xlabel('training step'); ax.set_ylabel('rare_recall (mean)', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (b) distinct_ops trajectory
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, [v3[s]['IID']['distinct_ops'] for s in steps], '-o', color='C1', label='v3', markersize=4, lw=1.6)
    ax.plot(steps, [v4[s]['IID']['distinct_ops'] for s in steps], '-s', color='C0', label='v4', markersize=4, lw=1.6)
    ax.set_title('(b) distinct ops in predictions', fontsize=12)
    ax.set_xlabel('training step'); ax.set_ylabel('distinct ops', fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (c) feature_F1 trajectory
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(steps, [v3[s]['IID']['feat_f1'] for s in steps], '-o', color='C1', label='v3', markersize=4, lw=1.6)
    ax.plot(steps, [v4[s]['IID']['feat_f1'] for s in steps], '-s', color='C0', label='v4', markersize=4, lw=1.6)
    ax.set_title('(c) feature F1 (chamfer / fillet / hole)', fontsize=12)
    ax.set_xlabel('training step'); ax.set_ylabel('feature F1', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (d) essential_pass trajectory (added — important to show no IID degradation)
    ax = fig.add_subplot(gs[1, 0])
    v3_ess = [v3[s]['IID']['ess_pass'] if v3[s]['IID']['ess_pass'] is not None else None for s in steps]
    v4_ess = [v4[s]['IID']['ess_pass'] if v4[s]['IID']['ess_pass'] is not None else None for s in steps]
    ax.plot(steps, v3_ess, '-o', color='C1', label='v3', markersize=4, lw=1.6)
    ax.plot(steps, v4_ess, '-s', color='C0', label='v4', markersize=4, lw=1.6)
    ax.set_title('(d) essential_pass IID (no degradation)', fontsize=12)
    ax.set_xlabel('training step'); ax.set_ylabel('essential_pass', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (e) IoU trajectory
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, [v3[s]['IID']['iou'] for s in steps], '-o', color='C1', label='v3', markersize=4, lw=1.6)
    ax.plot(steps, [v4[s]['IID']['iou'] for s in steps], '-s', color='C0', label='v4', markersize=4, lw=1.6)
    ax.set_title('(e) IoU IID (matched)', fontsize=12)
    ax.set_xlabel('training step'); ax.set_ylabel('IoU', fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (f) overall recall trajectory
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(steps, [v3[s]['IID']['recall'] for s in steps], '-o', color='C1', label='v3', markersize=4, lw=1.6)
    ax.plot(steps, [v4[s]['IID']['recall'] for s in steps], '-s', color='C0', label='v4', markersize=4, lw=1.6)
    ax.set_title('(f) overall op recall', fontsize=12)
    ax.set_xlabel('training step'); ax.set_ylabel('op recall', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (g) per-rare-op recall bar chart at final step
    ax = fig.add_subplot(gs[2, :])
    final_step = max(steps)
    v3_per = v3[final_step]['IID']['per_op_recall']
    v4_per = v4[final_step]['IID']['per_op_recall']
    n_gt = v3[final_step]['IID']['per_op_gt_n']
    # filter to ops with at least 2 GT samples
    visible = [op for op in rare_ops if n_gt.get(op, 0) >= 2]
    visible.sort(key=lambda o: -(n_gt.get(o, 0)))  # most-common first
    x = np.arange(len(visible))
    width = 0.4
    v3_y = [v3_per.get(o, 0) or 0 for o in visible]
    v4_y = [v4_per.get(o, 0) or 0 for o in visible]
    bars1 = ax.bar(x - width/2, v3_y, width, label='v3', color='C1', alpha=0.85)
    bars2 = ax.bar(x + width/2, v4_y, width, label='v4', color='C0', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(visible, rotation=20, ha='right')
    ax.set_ylabel('per-rare-op recall (BC IID)', fontsize=11)
    ax.set_title(f'(g) per-rare-op recall at step {final_step:,} — n_GT shown above each op', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=10, loc='upper right')
    # n_GT annotation above
    for i, op in enumerate(visible):
        n = n_gt.get(op, 0)
        ax.text(i, 1.02, f'n={n}', ha='center', fontsize=8, color='gray')
        # delta annotation
        delta = v4_y[i] - v3_y[i]
        if abs(delta) > 0.01:
            color = 'green' if delta > 0 else 'red'
            ax.text(i, max(v3_y[i], v4_y[i]) + 0.05, f'{delta:+.2f}', ha='center', fontsize=9, color=color, fontweight='bold')

    fig.suptitle('Op recall and diversity gains from rare-op-combination data — v4-holdout vs v3', fontsize=14, y=0.995)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'  wrote {out} ({out.stat().st_size//1024} KB)')


def build_appendix_a1(v3, v4, uid2fam, ess_spec, out: Path):
    """OOD limitation: trajectory + per-family bar."""
    steps = sorted(set(v3) & set(v4))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) OOD essential_pass trajectory
    ax = axes[0]
    v3y = [v3[s]['OOD']['ess_pass'] for s in steps]
    v4y = [v4[s]['OOD']['ess_pass'] for s in steps]
    ax.plot(steps, v3y, '-o', color='C1', label='v3 (saw all families)', markersize=4, lw=1.6)
    ax.plot(steps, v4y, '-s', color='C0', label='v4 (10 families held out)', markersize=4, lw=1.6)
    ax.set_title('(a) BC val OOD essential_pass over training', fontsize=12)
    ax.set_xlabel('training step'); ax.set_ylabel('essential_pass (mean over n=9 OOD samples)', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)

    # (b) per-family OOD breakdown at final step
    ax = axes[1]
    final_step = max(steps)
    # Aggregate OOD across all steps per family for stability
    families = sorted(HOLDOUT)
    def agg_ood(parsed_run):
        # collect per-family iou from JSONL
        agg = {f: [] for f in families}
        return agg
    # Re-parse to per-family — simpler: load last step's predictions, group by family
    fam_iou_v3 = {f: [] for f in families}
    fam_iou_v4 = {f: [] for f in families}
    fam_ess_v3 = {f: [] for f in families}
    fam_ess_v4 = {f: [] for f in families}
    for label, src_dir, fam_iou, fam_ess in [('v3', V3_DIR, fam_iou_v3, fam_ess_v3),
                                              ('v4', V4_DIR, fam_iou_v4, fam_ess_v4)]:
        for f in sorted(Path(src_dir).glob('step-*.jsonl')):
            if '.max@' in f.name: continue
            step = int(f.stem.replace('step-', ''))
            if step % 1000 != 0 or step == 0: continue
            for line in f.open():
                if not line.strip(): continue
                r = json.loads(line)
                if r.get('bucket') != 'BenchCAD val': continue
                fam = uid2fam.get(r['uid'])
                if fam not in HOLDOUT: continue
                iou = r.get('iou', 0) or 0
                if iou < 0: iou = 0
                fam_iou[fam].append(iou)
                tax = load_tax()
                po = find_ops(r.get('pred_code') or '', tax['patterns'])
                ess = essential_pass(fam, po, ess_spec)
                if ess is not None:
                    fam_ess[fam].append(1 if ess else 0)

    visible_fams = [f for f in families if fam_iou_v4[f] or fam_iou_v3[f]]
    x = np.arange(len(visible_fams))
    width = 0.35
    v3_ess_y = [np.mean(fam_ess_v3[f]) if fam_ess_v3[f] else 0 for f in visible_fams]
    v4_ess_y = [np.mean(fam_ess_v4[f]) if fam_ess_v4[f] else 0 for f in visible_fams]
    ax.bar(x - width/2, v3_ess_y, width, label='v3', color='C1', alpha=0.85)
    ax.bar(x + width/2, v4_ess_y, width, label='v4', color='C0', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(visible_fams, rotation=45, ha='right')
    ax.set_ylabel('essential_pass (mean across all eval steps)', fontsize=10)
    ax.set_title('(b) per-OOD-family essential_pass — only 5 of 10 holdout families landed in n=50 sample', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=10)

    fig.suptitle('Appendix A.1: Family-level transfer remains limited', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'  wrote {out} ({out.stat().st_size//1024} KB)')


def build_appendix_a2(out: Path):
    """Op combination scatter: BC vs DeepCAD vs Fusion360.
    Each point = one family avg op profile."""
    tax = load_tax()
    rare = set(tax['rare'])
    patterns = tax['patterns']

    def family_profile(rows, get_code, get_family):
        """Return dict family -> (avg distinct_ops, avg rare_op_count, n)."""
        agg = {}  # fam -> [(distinct, rare_count), ...]
        for r in rows:
            code = get_code(r)
            fam = get_family(r)
            if not code or not fam: continue
            ops = find_ops(code, patterns)
            agg.setdefault(fam, []).append((len(ops), len(ops & rare)))
        out = {}
        for fam, samples in agg.items():
            out[fam] = (np.mean([s[0] for s in samples]),
                        np.mean([s[1] for s in samples]),
                        len(samples))
        return out

    # BenchCAD val
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    bc_prof = family_profile(
        bc_val,
        get_code=lambda r: open(REPO_ROOT / 'data/benchcad' / r['py_path']).read() if (REPO_ROOT / 'data/benchcad' / r['py_path']).exists() else '',
        get_family=lambda r: r.get('family'),
    )

    # DeepCAD test - read codes from test set
    # We don't have GT codes for DeepCAD/Fusion360 readily — use predictions JSONL gt_code as proxy (it's the GT)
    def parse_test(pred_dir, bucket):
        """Pull gt_code from any step JSONL."""
        f = next(iter(sorted(Path(pred_dir).glob('step-*.jsonl'))))
        rows = []
        for line in f.open():
            if not line.strip(): continue
            r = json.loads(line)
            if r.get('bucket') == bucket:
                rows.append({'gt': r.get('gt_code') or '', 'uid': r.get('uid', '')})
        return rows
    dc_rows = parse_test(V3_DIR, 'DeepCAD test')
    fu_rows = parse_test(V3_DIR, 'Fusion360 test')

    def simple_family(uid):
        # DeepCAD/Fusion360 don't have family field; bucket all into one synthetic "family" per dataset
        return 'all'

    def per_sample_profile(rows, get_code):
        """Each sample becomes a (distinct, rare_count) point."""
        out = []
        for r in rows:
            code = get_code(r)
            if not code: continue
            ops = find_ops(code, patterns)
            out.append((len(ops), len(ops & rare)))
        return out

    dc_pts = per_sample_profile(dc_rows, lambda r: r['gt'])
    fu_pts = per_sample_profile(fu_rows, lambda r: r['gt'])

    fig, ax = plt.subplots(figsize=(10, 7))

    # BC: each family one point (avg)
    bc_x = [p[0] for p in bc_prof.values()]
    bc_y = [p[1] for p in bc_prof.values()]
    bc_n = [p[2] for p in bc_prof.values()]
    sizes = [min(120, max(12, n / 5)) for n in bc_n]
    ax.scatter(bc_x, bc_y, s=sizes, color='C0', alpha=0.7, edgecolors='black', linewidths=0.5,
               label=f'BenchCAD ({len(bc_x)} families, point size ~ family size)')

    # DeepCAD: per-sample scatter
    dc_x = [p[0] + np.random.normal(0, 0.15) for p in dc_pts]
    dc_y = [p[1] + np.random.normal(0, 0.10) for p in dc_pts]
    ax.scatter(dc_x, dc_y, s=12, color='C1', alpha=0.5, label=f'DeepCAD test ({len(dc_pts)} samples)')

    # Fusion360
    fu_x = [p[0] + np.random.normal(0, 0.15) for p in fu_pts]
    fu_y = [p[1] + np.random.normal(0, 0.10) for p in fu_pts]
    ax.scatter(fu_x, fu_y, s=12, color='C2', alpha=0.5, label=f'Fusion360 test ({len(fu_pts)} samples)')

    ax.set_xlabel('distinct ops per sample (mean for BC families)', fontsize=11)
    ax.set_ylabel('rare ops per sample (mean for BC families)', fontsize=11)
    ax.set_title('Appendix A.2: Op-combination diversity — BC families spread across rare-op space\n'
                 'while DeepCAD/Fusion360 cluster at low-rare regions', fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.3, 5)

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'  wrote {out} ({out.stat().st_size//1024} KB)')


def main():
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    tax = load_tax()
    ess_spec = load_essentials()

    print('parsing v4 ...')
    v4 = parse_run(V4_DIR, uid2fam, tax, ess_spec)
    print(f'  v4 steps: {sorted(v4)}')
    print('parsing v3 ...')
    v3 = parse_run(V3_DIR, uid2fam, tax, ess_spec, max_step=max(v4) if v4 else None)
    print(f'  v3 steps: {sorted(v3)}')

    out_dir = REPO_ROOT / 'docs' / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('\nbuilding main figure ...')
    build_main_figure(v3, v4, tax['rare'], out_dir / 'fig_main_ops_recall.png')
    print('\nbuilding appendix A.1 ...')
    build_appendix_a1(v3, v4, uid2fam, ess_spec, out_dir / 'fig_app_a1_ood_limitation.png')
    print('\nbuilding appendix A.2 ...')
    build_appendix_a2(out_dir / 'fig_app_a2_op_combo_scatter.png')

    print('\ndone — figures in docs/paper_figures/')


if __name__ == '__main__':
    main()
