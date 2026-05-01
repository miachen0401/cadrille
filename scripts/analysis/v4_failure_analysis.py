"""§8 Failure analysis figures — single script producing the full set.

Outputs to docs/:
  v4_failure_per_family_<date>.png   per-family OOD breakdown bar chart
  v4_failure_op_confusion_<date>.png op confusion on OOD (GT op vs pred ops)
  v4_failure_code_length_<date>.png  code length / distinct_codes histograms
  v4_failure_exec_modes_<date>.png   exec failure rate over training

Aggregates predictions across all available step-NNNNNN.jsonl files.

Usage:
    uv run python -m scripts.analysis.v4_failure_analysis
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V4_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/predictions'
V3_DIR = '/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/predictions'

from common.holdout import HOLDOUT_FAMILIES as HOLDOUT


def _load_taxonomy():
    tax = yaml.safe_load(open(REPO_ROOT / 'configs/eval/op_taxonomy.yaml'))
    return {
        'patterns': {n: re.compile(p) for n, p in tax['patterns'].items()},
        'rare': set(tax['rare']),
        'feature': set(tax['feature']),
    }


from common.essential_ops import ESSENTIAL_BY_FAMILY
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


def parse_run(pred_dir, uid2fam, tax, ess_spec):
    records = []
    for f in sorted(Path(pred_dir).glob('step-*.jsonl')):
        if '.max@' in f.name: continue
        step = int(f.stem.replace('step-', ''))
        if step % 1000 != 0 or step == 0: continue
        for line in f.open():
            if not line.strip(): continue
            r = json.loads(line)
            if r.get('bucket') != 'BenchCAD val': continue
            fam = uid2fam.get(r['uid'])
            po = find_ops(r.get('pred_code') or '', tax['patterns'])
            go = find_ops(r.get('gt_code') or '', tax['patterns'])
            iou = r.get('iou', 0)
            execed = (r.get('has_iou') and iou is not None and iou >= 0)
            iou = max(iou or 0, 0)
            ess = essential_pass(fam, po, ess_spec)
            records.append({
                'step': step, 'uid': r['uid'], 'family': fam,
                'pred_code': r.get('pred_code') or '',
                'gt_code': r.get('gt_code') or '',
                'pred_ops': po, 'gt_ops': go,
                'iou': iou, 'execed': execed, 'ess_pass': ess,
                'is_ood': fam in HOLDOUT,
                'pred_len': len(r.get('pred_code') or ''),
                'gt_len': len(r.get('gt_code') or ''),
            })
    return records


def fig1_per_family(v3_records, v4_records, out: Path):
    families_seen = sorted(set(r['family'] for r in v4_records if r['is_ood']))
    print(f'OOD families with predictions: {families_seen}')

    def agg(records, families):
        out = {}
        for fam in families:
            fr = [r for r in records if r['family'] == fam]
            if not fr: continue
            out[fam] = {
                'iou': np.mean([r['iou'] for r in fr]),
                'exec': np.mean([r['execed'] for r in fr]) * 100,
                'ess_pass': np.mean([r['ess_pass'] for r in fr if r['ess_pass'] is not None])
                            if any(r['ess_pass'] is not None for r in fr) else None,
            }
        return out

    v3_agg = agg(v3_records, families_seen)
    v4_agg = agg(v4_records, families_seen)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_specs = [('iou', 'IoU', (0, 1)), ('exec', 'exec %', (0, 105)),
                    ('ess_pass', 'essential_pass', (0, 1.05))]
    x = np.arange(len(families_seen))
    width = 0.4

    for ax, (key, label, ylim) in zip(axes, metric_specs):
        v3_y = [v3_agg.get(f, {}).get(key, 0) or 0 for f in families_seen]
        v4_y = [v4_agg.get(f, {}).get(key, 0) or 0 for f in families_seen]
        ax.bar(x - width/2, v3_y, width, label='v3 (saw all families)', color='C1', alpha=0.85)
        ax.bar(x + width/2, v4_y, width, label='v4 (held out)', color='C0', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(families_seen, rotation=45, ha='right')
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} per OOD family (mean across all steps)', fontsize=11)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        for i, (a, b) in enumerate(zip(v3_y, v4_y)):
            ax.text(i - width/2, a + ylim[1]*0.01, f'{a:.2f}', ha='center', fontsize=8)
            ax.text(i + width/2, b + ylim[1]*0.01, f'{b:.2f}', ha='center', fontsize=8)

    fig.suptitle('§8.1 Per-family OOD failure breakdown — v3 vs v4-holdout', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'  wrote {out}')


def fig2_op_confusion(v4_records, out: Path):
    families_in_spec = sorted(HOLDOUT)
    family_req = {
        'tapered_boss': 'loft|taper=', 'taper_pin': 'revolve|sweep',
        'venturi_tube': 'revolve|sweep', 'bucket': 'revolve|sweep',
        'dome_cap': 'revolve|sweep', 'nozzle': 'revolve|sweep',
        'enclosure': 'shell', 'waffle_plate': 'rarray',
        'bolt': 'polygon', 'duct_elbow': 'sweep',
    }
    rows = []
    for fam in families_in_spec:
        fr = [r for r in v4_records if r['family'] == fam]
        if not fr:
            rows.append((fam, family_req[fam], 'no samples', '—', 0))
            continue
        op_counter = Counter()
        for r in fr:
            for o in r['pred_ops']:
                op_counter[o] += 1
        top5 = ', '.join(f'{n}({c})' for n, c in op_counter.most_common(5))
        req = family_req[fam]
        req_ops = req.split('|')
        hits = sum(1 for r in fr if any(o in r['pred_ops'] for o in req_ops))
        rate = hits / len(fr)
        rows.append((fam, req, top5, f'{hits}/{len(fr)}', rate))

    fig, ax = plt.subplots(figsize=(15, 7))
    cols = ['family', 'required (GT)', 'v4 top-5 emitted ops', 'req hits', 'hit rate']
    table_data = [[r[0], r[1], r[2][:60] + ('...' if len(r[2]) > 60 else ''),
                   r[3], f'{r[4]:.0%}'] for r in rows]
    table = ax.table(cellText=table_data, colLabels=cols, loc='center',
                     cellLoc='left', colLoc='center',
                     colWidths=[0.13, 0.12, 0.45, 0.10, 0.08])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.6)
    for i, r in enumerate(rows):
        rate = r[4]
        c = (0.85, 1, 0.85) if rate > 0.7 else (1, 0.95, 0.7) if rate > 0.4 else (1, 0.85, 0.85)
        table[(i+1, 4)].set_facecolor(c)
    ax.axis('off')
    ax.set_title('§8.2 OOD op confusion — does v4 emit the GT-required op on held-out families?\n'
                 '(aggregated over all eval steps)', fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'  wrote {out}')


def fig3_code_length(v3_records, v4_records, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0, 1500, 30)
    splits = [('IID', lambda r: not r['is_ood']),
              ('OOD', lambda r: r['is_ood'])]

    for ax, (label, pred) in zip(axes, splits):
        v3_lens = [r['pred_len'] for r in v3_records if pred(r)]
        v4_lens = [r['pred_len'] for r in v4_records if pred(r)]
        gt_lens = [r['gt_len'] for r in v4_records if pred(r) and r['gt_len']]
        ax.hist(v3_lens, bins=bins, alpha=0.5, label=f'v3 pred (n={len(v3_lens)})', color='C1')
        ax.hist(v4_lens, bins=bins, alpha=0.5, label=f'v4 pred (n={len(v4_lens)})', color='C0')
        ax.hist(gt_lens, bins=bins, alpha=0.3, label=f'GT (n={len(gt_lens)})',
                color='gray', histtype='step', linewidth=2)
        ax.set_xlabel('predicted code length (chars)'); ax.set_ylabel('count')
        ax.set_title(f'BC val {label} — code length distribution')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle('§8.3 Code length: predicted vs GT, IID vs OOD', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'  wrote {out}')


def fig4_exec_modes(v3_records, v4_records, out: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    splits = [('v3 IID', v3_records, False, 'C1', '-'),
              ('v3 OOD', v3_records, True,  'C1', '--'),
              ('v4 IID', v4_records, False, 'C0', '-'),
              ('v4 OOD', v4_records, True,  'C0', '--')]
    for label, records, ood, color, ls in splits:
        by_step = defaultdict(list)
        for r in records:
            if r['is_ood'] != ood: continue
            by_step[r['step']].append(0 if r['execed'] else 1)
        steps_x = sorted(by_step)
        rates = [np.mean(by_step[s]) * 100 for s in steps_x]
        ax.plot(steps_x, rates, marker='o', label=label, color=color, linestyle=ls, markersize=4)
    ax.set_xlabel('training step'); ax.set_ylabel('exec failure rate (%)')
    ax.set_title('§8.4 Exec failure rate over training — IID vs OOD, v3 vs v4')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 50)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'  wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'docs')
    args = ap.parse_args()

    bc_val = pickle.load(open(REPO_ROOT / 'data/benchcad/val.pkl', 'rb'))
    uid2fam = {r['uid']: r['family'] for r in bc_val}
    tax = _load_taxonomy()
    ess_spec = ESSENTIAL_BY_FAMILY

    print('parsing v4 ...')
    v4 = parse_run(V4_DIR, uid2fam, tax, ess_spec)
    print(f'  v4 records: {len(v4)}')
    print('parsing v3 ...')
    v3 = parse_run(V3_DIR, uid2fam, tax, ess_spec)
    print(f'  v3 records: {len(v3)}')

    today = date.today().isoformat()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    fig1_per_family(v3, v4, args.out_dir / f'v4_failure_per_family_{today}.png')
    fig2_op_confusion(v4, args.out_dir / f'v4_failure_op_confusion_{today}.png')
    fig3_code_length(v3, v4, args.out_dir / f'v4_failure_code_length_{today}.png')
    fig4_exec_modes(v3, v4, args.out_dir / f'v4_failure_exec_modes_{today}.png')

    print('\nAll figures written:')
    for p in sorted(args.out_dir.glob(f'v4_failure_*_{today}.png')):
        print(f'  {p.relative_to(REPO_ROOT)} ({p.stat().st_size//1024} KB)')


if __name__ == '__main__':
    main()
