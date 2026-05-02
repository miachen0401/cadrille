"""§7 OOD IoU vs step — 4-line plot.

Lines:
  (1) IID ceiling      — v3, eval IID
  (2) OOD plain        — v4-holdout-noeasy [placeholder]
  (3) OOD + bench-easy — v4-holdout (real)
  (4) no-bench         — v4-hq-only [placeholder]

Companion to §7.a ops-metric plot.

Usage:
    uv run python -m scripts.analysis.plot_ood_iou_4line
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V3_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions'
V4_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/predictions'
BASELINE_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0501-0629/predictions'

from common.holdout import HOLDOUT_FAMILIES as HOLDOUT


def iou_per_step(pred_dir, uid2fam, target_holdout):
    out = {}
    for f in sorted(Path(pred_dir).glob('step-*.jsonl')):
        if '.max@' in f.name: continue
        step = int(f.stem.replace('step-', ''))
        if step % 1000 != 0 or step == 0: continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        # Accept legacy single 'BenchCAD val' AND new IID/OOD split forms.
        if target_holdout:
            sub = [r for r in rows if (r.get('bucket') or '').startswith('BenchCAD val')
                   and uid2fam.get(r['uid']) in HOLDOUT]
        else:
            sub = [r for r in rows if (r.get('bucket') or '').startswith('BenchCAD val')
                   and uid2fam.get(r['uid']) not in HOLDOUT]
        if not sub: continue
        ious = []
        for r in sub:
            iou = r.get('iou', 0) or 0
            if iou < 0: iou = 0
            ious.append(iou)
        out[step] = float(np.mean(ious))
    return out


def main():
    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}

    print('parsing v3 OOD IoU ...')
    v3_ood = iou_per_step(V3_DIR, uid2fam, target_holdout=True)
    print(f'  steps: {len(v3_ood)}')
    print('parsing v3 IID IoU ...')
    v3_iid = iou_per_step(V3_DIR, uid2fam, target_holdout=False)
    print('parsing v4 OOD IoU ...')
    v4_ood = iou_per_step(V4_DIR, uid2fam, target_holdout=True)
    print('parsing baseline OOD IoU ...')
    base_ood = iou_per_step(BASELINE_DIR, uid2fam, target_holdout=True)
    print(f'  steps: {sorted(base_ood)}')

    steps_v4 = sorted(v4_ood)
    steps_iid = sorted(v3_iid)
    steps_v3_ood = sorted(v3_ood)
    steps_base = sorted(base_ood)
    max_step = max(max(steps_v4, default=0), max(steps_iid, default=0), max(steps_base, default=0))

    fig, ax = plt.subplots(figsize=(10, 6))

    # (1) IID ceiling = v3 IID
    ax.plot(steps_iid, [v3_iid[s] for s in steps_iid], '-', color='C2', lw=2.2,
            label='(1) IID ceiling — v3 saw these families')
    # (2) OOD plain — v4-holdout-noeasy [placeholder zeros]
    placeholder_x = np.arange(1000, max_step + 1, 1000)
    ax.plot(placeholder_x, [0]*len(placeholder_x), '--', color='C3', lw=1.5,
            label='(2) OOD plain — v4-holdout-noeasy [TBD]', alpha=0.5)
    # (3) OOD + bench-easy — v4-holdout (real)
    ax.plot(steps_v4, [v4_ood[s] for s in steps_v4], '-s', color='C0', lw=2.2, markersize=5,
            label='(3) OOD + bench-easy — v4-holdout (current)')
    # (4) no-bench — baseline (real, accumulating)
    if steps_base:
        ax.plot(steps_base, [base_ood[s] for s in steps_base], '-^', color='C4', lw=2.2, markersize=5,
                label=f'(4) no-bench — baseline (current, n={len(steps_base)} steps)')
    else:
        ax.plot(placeholder_x, [0]*len(placeholder_x), '--', color='C4', lw=1.5,
                label='(4) no-bench — baseline [TBD]', alpha=0.5)

    # ref: v3 OOD (model saw → it's actually IID for v3)
    ax.plot(steps_v3_ood, [v3_ood[s] for s in steps_v3_ood], ':', color='C1', lw=1.5,
            label='ref — v3 evaluated on holdout families (v3 saw them, so IID for v3)', alpha=0.7)

    ax.set_xlabel('training step', fontsize=11)
    ax.set_ylabel('IoU (mean over OOD samples, n=9)', fontsize=11)
    ax.set_ylim(-0.02, 0.95)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title('§7.b OOD IoU vs training step — 4-line comparison',
                 fontsize=12)

    fig.tight_layout()
    out = REPO_ROOT / 'paper/figures/fig_7_ood_iou_4line.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'wrote {out} ({out.stat().st_size//1024} KB)')


if __name__ == '__main__':
    main()
