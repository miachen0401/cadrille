"""§7 essential_pass=1 rate vs step — 4-line plot per the paper architecture.

Lines:
  (1) IID ceiling      — v3 trained on all families, eval on the 10 OOD-target families (which v3 saw)
  (2) OOD plain        — v4-holdout-noeasy (held out + no benchcad-easy supplement)  [placeholder zeros]
  (3) OOD + bench-easy — v4-holdout (held out + benchcad-easy) [real data]
  (4) no-bench         — v4-hq-only (text2cad + recode_bench only) [placeholder zeros]

Same plot is generated for IID ceiling on left, OOD on right.

Usage:
    uv run python -m scripts.analysis.plot_4line_ess
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


def load_setup():
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    tax = yaml.safe_load(open(REPO_ROOT / 'configs/eval/op_taxonomy.yaml'))
    patterns = {n: re.compile(p) for n, p in tax['patterns'].items()}
    ess_spec = yaml.safe_load(open(REPO_ROOT / 'configs/eval/canonical_ops.yaml'))
    return uid2fam, patterns, ess_spec


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


def metrics_per_step(pred_dir, uid2fam, patterns, ess_spec, target_holdout=True):
    """Returns {step: ess_pass_rate} aggregated over OOD or IID samples."""
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
        ess = []
        for r in sub:
            po = find_ops(r.get('pred_code') or '', patterns)
            fam = uid2fam.get(r['uid'])
            e = ess_pass(fam, po, ess_spec) if fam else None
            if e is not None:
                ess.append(1 if e else 0)
        if ess:
            out[step] = float(np.mean(ess))
    return out


def main():
    uid2fam, patterns, ess_spec = load_setup()

    print('parsing v3 OOD ess_pass ...')
    v3_ood = metrics_per_step(V3_DIR, uid2fam, patterns, ess_spec, target_holdout=True)
    print(f'  steps: {len(v3_ood)}')
    print('parsing v3 IID ess_pass ...')
    v3_iid = metrics_per_step(V3_DIR, uid2fam, patterns, ess_spec, target_holdout=False)
    print('parsing v4-holdout OOD ess_pass ...')
    v4_ood = metrics_per_step(V4_DIR, uid2fam, patterns, ess_spec, target_holdout=True)
    print('parsing v4-holdout IID ess_pass ...')
    v4_iid = metrics_per_step(V4_DIR, uid2fam, patterns, ess_spec, target_holdout=False)

    steps_v4 = sorted(v4_ood)
    steps_v3 = sorted(v3_ood)
    steps_all = sorted(set(steps_v3) | set(steps_v4))
    max_step = max(steps_all) if steps_all else 50000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: OOD ess_pass = 1 rate
    ax = axes[0]
    # (1) IID ceiling = v3 IID (model saw ALL families, eval on IID)
    iid_steps = sorted(v3_iid)
    ax.plot(iid_steps, [v3_iid[s] for s in iid_steps], '-', color='C2', lw=2,
            label='(1) IID ceiling — v3 saw all families', alpha=0.8)
    # (2) OOD plain — v4-holdout-noeasy (NOT YET TRAINED, placeholder zeros)
    placeholder_x = np.arange(1000, max_step + 1, 1000)
    ax.plot(placeholder_x, [0]*len(placeholder_x), '--', color='C3', lw=1.5,
            label='(2) OOD plain — v4-holdout-noeasy [PLACEHOLDER, not yet trained]', alpha=0.6)
    # (3) OOD + bench-easy — v4-holdout (real data)
    ax.plot(steps_v4, [v4_ood[s] for s in steps_v4], '-s', color='C0', lw=2, markersize=5,
            label='(3) OOD + bench-easy — v4-holdout (current)', alpha=0.9)
    # (4) no-bench — v4-hq-only (NOT YET TRAINED, placeholder zeros)
    ax.plot(placeholder_x, [0]*len(placeholder_x), '--', color='C4', lw=1.5,
            label='(4) no-bench — v4-hq-only [PLACEHOLDER, not yet trained]', alpha=0.6)
    # Reference: v3 OOD (model saw those families, eval on them) — informative
    ax.plot(steps_v3, [v3_ood[s] for s in steps_v3], ':', color='C1', lw=1.5,
            label='ref — v3 evaluated on holdout families (saw them in train)', alpha=0.6)

    ax.set_title('§7.c essential_pass = 1 rate (BC val, OOD families) vs training step',
                 fontsize=11)
    ax.set_xlabel('training step'); ax.set_ylabel('essential_pass rate (mean)')
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    ax.axhline(1.0, color='gray', lw=0.5, linestyle=':')

    # Right: IID ess_pass = 1 rate (sanity check)
    ax = axes[1]
    ax.plot(sorted(v3_iid), [v3_iid[s] for s in sorted(v3_iid)], '-', color='C2', lw=2,
            label='v3 IID', alpha=0.85)
    ax.plot(sorted(v4_iid), [v4_iid[s] for s in sorted(v4_iid)], '-s', color='C0', lw=2, markersize=4,
            label='v4-holdout IID (current)', alpha=0.85)
    ax.plot(placeholder_x, [0]*len(placeholder_x), '--', color='C3', lw=1.5,
            label='v4-holdout-noeasy IID [TBD]', alpha=0.6)
    ax.plot(placeholder_x, [0]*len(placeholder_x), '--', color='C4', lw=1.5,
            label='v4-hq-only IID [TBD]', alpha=0.6)
    ax.set_title('§7.b essential_pass = 1 rate (BC val, IID families)', fontsize=11)
    ax.set_xlabel('training step'); ax.set_ylabel('essential_pass rate (mean)')
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    ax.axhline(1.0, color='gray', lw=0.5, linestyle=':')

    fig.suptitle('Paper §7 4-line plot — essential_pass rate vs training step\n'
                 '(placeholder zeros = configs not yet trained)', fontsize=12)
    fig.tight_layout()
    out = REPO_ROOT / 'paper/figures/fig_7_4line_ess_pass.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'wrote {out} ({out.stat().st_size//1024} KB)')


if __name__ == '__main__':
    main()
