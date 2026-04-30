"""4-panel comparison chart for cad_bench_722 + OOD after CADEvolve v3 fix.

Layout:
  Panel A: Headline mean-IoU bar chart (cad_bench / DeepCAD / Fusion360 × 4 models)
  Panel B: Essential-ops + feature-F1 (coverage-weighted) on cad_bench
  Panel C: Per-difficulty mean IoU on cad_bench (easy / medium / hard × 4 models)
  Panel D: Op-vocabulary frequency vs GT (top 14 ops, grouped bars)

Plus a second figure (per-family wins) showing top-12 families × 4 models heatmap.

Posts both to Discord.

Usage:
    set -a; source .env; set +a
    uv run python research/repro_official/build_v3_charts.py --discord
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

MODELS = [
    ('cadrille_rl_repro',    'Cadrille-rl', '#888888'),
    ('cadevolve_rl1',        'CADEvolve-rl1 v3', '#E15F41'),
    ('cadrille_qwen3vl_v3',  'Cadrille-Q3VL-v3 (ours)', '#3D9A3D'),
    ('qwen25vl_3b_zs',       'Qwen2.5-VL-3B zs', '#5DADE2'),
]
EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'


def _load_jsonl(p):
    if p is None or not Path(p).exists():
        return []
    return [json.loads(l) for l in open(p)]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def _summary(rs):
    ok = [r for r in rs if r.get('error_type') == 'success']
    ious = [r['iou'] for r in ok if r.get('iou') is not None]
    return _mean(ious)


def _summary_24(rs24):
    ok = [r for r in rs24 if r.get('error_type') == 'success']
    iou_24s = [r['iou_24'] for r in ok if r.get('iou_24') is not None]
    return _mean(iou_24s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(EVAL_ROOT))
    ap.add_argument('--discord', action='store_true')
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    # ── Load data ──────────────────────────────────────────────────────────
    cb_iou, cb_iou24 = {}, {}
    cb_per_diff = {slug: {} for slug, _, _ in MODELS}
    for slug, _, _ in MODELS:
        meta = EVAL_ROOT / slug / 'metadata.jsonl'
        # repro_official path for cadrille_rl_repro
        if slug == 'cadrille_rl_repro':
            meta = REPO / 'eval_outputs/repro_official/cad_bench_722_full/metadata.jsonl'
        rs = _load_jsonl(meta)
        cb_iou[slug] = _summary(rs)
        meta_24 = EVAL_ROOT / slug / 'metadata_24.jsonl'
        rs24 = _load_jsonl(meta_24)
        cb_iou24[slug] = _summary_24(rs24)
        # per-difficulty
        for diff in ('easy', 'medium', 'hard'):
            sub = [r['iou'] for r in rs
                   if r.get('error_type') == 'success'
                   and r.get('difficulty') == diff
                   and r.get('iou') is not None]
            cb_per_diff[slug][diff] = _mean(sub)

    # OOD
    ood_iou = {ds: {} for ds in ('deepcad', 'fusion360')}
    ood_paths = {
        'deepcad': {
            'cadevolve_rl1':       REPO / 'eval_outputs/repro_official/deepcad_n300/cadevolve_v3/metadata.jsonl',
            'cadrille_qwen3vl_v3': REPO / 'eval_outputs/deepcad_n300/cadrille_qwen3vl_v3/metadata.jsonl',
            'qwen25vl_3b_zs':      REPO / 'eval_outputs/deepcad_n300/qwen25vl_3b_zs/metadata.jsonl',
            'cadrille_rl_repro':   None,  # value parsed from score.txt below
        },
        'fusion360': {
            'cadevolve_rl1':       REPO / 'eval_outputs/repro_official/fusion360_n300/cadevolve_v3/metadata.jsonl',
            'cadrille_qwen3vl_v3': REPO / 'eval_outputs/fusion360_n300/cadrille_qwen3vl_v3/metadata.jsonl',
            'qwen25vl_3b_zs':      REPO / 'eval_outputs/fusion360_n300/qwen25vl_3b_zs/metadata.jsonl',
            'cadrille_rl_repro':   None,
        },
    }
    for ds, paths in ood_paths.items():
        for slug, p in paths.items():
            if p is None:
                # Cadrille-rl repro from score.txt
                txt_p = (REPO / 'eval_outputs/repro_official' /
                         f'{("deepcad_test_mesh" if ds=="deepcad" else "fusion360_test_mesh")}_n300' /
                         'score.txt')
                if txt_p.exists():
                    for line in txt_p.read_text().splitlines():
                        if 'mean iou' in line.lower():
                            try:
                                ood_iou[ds][slug] = float(line.split(':')[-1].strip().split()[0])
                            except Exception:
                                ood_iou[ds][slug] = float('nan')
                            break
                    else:
                        ood_iou[ds][slug] = float('nan')
                else:
                    ood_iou[ds][slug] = float('nan')
                continue
            ood_iou[ds][slug] = _summary(_load_jsonl(p))

    # essential_ops
    ess = json.loads((EVAL_ROOT / 'essential_ops.json').read_text())
    ess_cw = {slug: ess['models'].get(slug, {}).get('pct_essential_pass_cw', 0)
              for slug, _, _ in MODELS}
    f1_cw  = {slug: ess['models'].get(slug, {}).get('mean_feature_f1_cw', 0)
              for slug, _, _ in MODELS}

    # Op-vocab frequency from per_case
    gt_op_freq = Counter()
    n_gt = 0
    if 'cadrille_qwen3vl_v3' in ess['models']:
        for case in ess['models']['cadrille_qwen3vl_v3']['per_case']:
            n_gt += 1
            for op in case.get('gt_ops') or []:
                gt_op_freq[op] += 1
    model_op_freq = {}
    for slug, _, _ in MODELS:
        c = Counter()
        n = 0
        if slug in ess['models']:
            for case in ess['models'][slug]['per_case']:
                n += 1
                for op in case.get('gen_ops') or []:
                    c[op] += 1
        model_op_freq[slug] = (c, n)

    # ── Figure 1: 4-panel headline ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes[0, 0].set_title('A. mean IoU across 3 datasets',
                         fontsize=14, fontweight='bold')
    datasets = ['cad_bench_722', 'DeepCAD-300', 'Fusion360-300']
    x = np.arange(len(datasets))
    bw = 0.2
    for i, (slug, label, color) in enumerate(MODELS):
        vals = [cb_iou[slug],
                ood_iou['deepcad'].get(slug, float('nan')),
                ood_iou['fusion360'].get(slug, float('nan'))]
        bars = axes[0, 0].bar(x + i * bw - 1.5 * bw, vals, bw,
                              label=label, color=color, edgecolor='black', linewidth=0.4)
        for b, v in zip(bars, vals):
            if v == v:  # not nan
                axes[0, 0].text(b.get_x() + b.get_width() / 2, v + 0.01,
                                f'{v:.2f}', ha='center', fontsize=9)
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].set_ylim(0, 1.05); axes[0, 0].set_ylabel('mean IoU')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].legend(loc='upper right', fontsize=9, framealpha=0.95)

    # B: ess_cw + F1_cw on cad_bench
    axes[0, 1].set_title('B. Essential-ops + F1 (coverage-weighted, cad_bench_722)',
                         fontsize=14, fontweight='bold')
    metrics = ['ess_cw', 'F1_cw']
    x = np.arange(len(metrics))
    for i, (slug, label, color) in enumerate(MODELS):
        vals = [ess_cw[slug], f1_cw[slug]]
        bars = axes[0, 1].bar(x + i * bw - 1.5 * bw, vals, bw,
                              label=label, color=color, edgecolor='black', linewidth=0.4)
        for b, v in zip(bars, vals):
            axes[0, 1].text(b.get_x() + b.get_width() / 2, v + 0.01,
                            f'{v:.2f}', ha='center', fontsize=9)
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(['ess_cw (n_pass/720)', 'mean F1_cw'])
    axes[0, 1].set_ylim(0, 1.0); axes[0, 1].set_ylabel('coverage-weighted score')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # C: per-difficulty IoU on cad_bench
    axes[1, 0].set_title('C. Per-difficulty mean IoU (cad_bench_722)',
                         fontsize=14, fontweight='bold')
    diffs = ['easy', 'medium', 'hard']
    x = np.arange(len(diffs))
    for i, (slug, label, color) in enumerate(MODELS):
        vals = [cb_per_diff[slug].get(d, float('nan')) for d in diffs]
        bars = axes[1, 0].bar(x + i * bw - 1.5 * bw, vals, bw,
                              label=label, color=color, edgecolor='black', linewidth=0.4)
        for b, v in zip(bars, vals):
            if v == v:
                axes[1, 0].text(b.get_x() + b.get_width() / 2, v + 0.01,
                                f'{v:.2f}', ha='center', fontsize=8)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(diffs)
    axes[1, 0].set_ylim(0, 1.05); axes[1, 0].set_ylabel('mean IoU')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # D: op-vocab freq (top 14 ops by GT freq)
    axes[1, 1].set_title('D. Op-vocabulary frequency vs GT (cad_bench_722)',
                         fontsize=14, fontweight='bold')
    top_ops = [op for op, _ in gt_op_freq.most_common(14)]
    x = np.arange(len(top_ops))
    bw2 = 0.16
    # GT bars (gray)
    gt_pct = [gt_op_freq[op] / n_gt * 100 for op in top_ops]
    axes[1, 1].bar(x - 2 * bw2, gt_pct, bw2, label='GT', color='#444444',
                   edgecolor='black', linewidth=0.4)
    for i, (slug, label, color) in enumerate(MODELS):
        c, n = model_op_freq[slug]
        if n == 0:
            continue
        vals = [c[op] / n * 100 for op in top_ops]
        axes[1, 1].bar(x + (i - 1) * bw2, vals, bw2, label=label, color=color,
                       edgecolor='black', linewidth=0.4)
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(top_ops, rotation=40, ha='right')
    axes[1, 1].set_ylim(0, 65); axes[1, 1].set_ylabel('% of preds using op')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].legend(loc='upper right', fontsize=8, framealpha=0.95)

    fig.suptitle('cad_bench_722 + OOD — CADEvolve v3 fixed-setup vs all baselines',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    out_main = Path(args.out_dir) / 'v3_comparison_panels.png'
    fig.savefig(out_main, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  → {out_main}')

    # ── Figure 2: per-family heatmap ──────────────────────────────────────
    # Top 16 families by sample count among those with essential spec
    fam_count = Counter()
    if 'cadrille_qwen3vl_v3' in ess['models']:
        for case in ess['models']['cadrille_qwen3vl_v3']['per_case']:
            fam = case.get('family')
            ep  = case.get('essential_pass')
            if fam and ep is not None:  # has spec
                fam_count[fam] += 1
    top_fams = [f for f, _ in fam_count.most_common(16)]

    fig2, ax = plt.subplots(figsize=(12, 7))
    matrix = np.full((len(MODELS), len(top_fams)), np.nan)
    for i, (slug, _, _) in enumerate(MODELS):
        if slug not in ess['models']:
            continue
        per_fam = {f: [] for f in top_fams}
        for case in ess['models'][slug]['per_case']:
            fam = case.get('family')
            ep  = case.get('essential_pass')
            if fam in per_fam and ep is not None:
                per_fam[fam].append(1 if ep else 0)
        for j, fam in enumerate(top_fams):
            xs = per_fam[fam]
            if xs:
                matrix[i, j] = sum(xs) / len(xs) * 100

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[1] for m in MODELS])
    ax.set_xticks(range(len(top_fams)))
    ax.set_xticklabels(top_fams, rotation=45, ha='right')
    for i in range(len(MODELS)):
        for j in range(len(top_fams)):
            if not np.isnan(matrix[i, j]):
                txt = f'{matrix[i, j]:.0f}'
                color = 'white' if matrix[i, j] < 30 or matrix[i, j] > 70 else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                        color=color, fontweight='bold')
    cbar = fig2.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('essential pass-rate (%)', rotation=270, labelpad=15)
    ax.set_title('Per-family essential pass-rate (cad_bench_722, top 16 families)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_heatmap = Path(args.out_dir) / 'v3_per_family_heatmap.png'
    fig2.savefig(out_heatmap, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f'  → {out_heatmap}')

    # ── Discord ────────────────────────────────────────────────────────────
    if args.discord:
        url = os.environ.get('DISCORD_WEBHOOK_URL')
        if not url:
            print('  (no DISCORD_WEBHOOK_URL — skip)')
            return
        msg = ('📊 **cad_bench_722 + OOD — comparison charts (after CADEvolve v3 fix)**\n'
               '\n'
               '**Panel A**: mean IoU on cad_bench / DeepCAD / Fusion360 — '
               'CADEvolve dominates on geometry (0.81 / 0.93 / 0.87) once setup is correct.\n'
               '**Panel B**: essential-ops + F1 (coverage-weighted) on cad_bench — '
               '**ours dominates** (ess_cw 67.5% / F1_cw 0.73 vs CADEvolve 14.7% / 0.36).\n'
               '**Panel C**: per-difficulty IoU on cad_bench — CADEvolve flat 0.78–0.83 '
               'across easy/medium/hard; ours 0.71→0.66→0.59.\n'
               '**Panel D**: op-vocab frequency — Q3VL-v3 (ours, green) tracks GT '
               'almost 1:1 across all 14 canonical ops; CADEvolve (red) skips revolve / '
               'lineTo / threePointArc / rarray entirely and rarely uses chamfer/fillet.\n'
               '\n'
               'Heatmap: per-family essential pass-rate. Ball_knob, capsule, bellows, '
               'grommet → **ours 100% vs CADEvolve 0–25%** (revolve/sphere vocab gap).\n')
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        def w(s): body.write(s.encode())
        w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n')
        w('Content-Type: application/json\r\n\r\n')
        w(json.dumps({'content': msg}) + '\r\n')
        for i, p in enumerate([out_main, out_heatmap]):
            w(f'--{boundary}\r\nContent-Disposition: form-data; '
              f'name="files[{i}]"; filename="{p.name}"\r\n')
            w('Content-Type: image/png\r\n\r\n')
            body.write(p.read_bytes()); w('\r\n')
        w(f'--{boundary}--\r\n')
        req = urllib.request.Request(url, data=body.getvalue(), headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'User-Agent': 'cadevolve-v3-charts/1.0',
        })
        try:
            urllib.request.urlopen(req, timeout=30).read()
            print('  Discord ✓')
        except Exception as e:
            print(f'  Discord failed: {e}')


if __name__ == '__main__':
    main()
