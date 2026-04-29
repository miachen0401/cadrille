"""IoU vs IoU-24 analysis on cad_bench_722.

Question: how much of each baseline's IoU loss is *orientation drift* —
i.e. the prediction is the right shape but rotated by a multiple of 90°?

For each (model, case) where the model exec'd successfully we have both
`iou` (no rotation) and `iou_24` (max over the 24 cube rotations) plus
`rot_idx` (which rotation won, 0 = identity). The Δ = iou_24 − iou is the
"rotation rescue" amount; the fraction of cases with rot_idx > 0 is the
"oriented-wrong rate".

Outputs (eval_outputs/cad_bench_722/iou_vs_iou24/):
  - report.md          — per-model, per-difficulty, per-family tables
  - scatter.png        — 3-panel scatter (iou x, iou_24 y) per model
  - histogram.png      — 3-panel histogram of iou vs iou_24 per model
  - rotation_dist.png  — bar chart of which rotation idx wins per model

Usage:
    set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
    uv run python research/3d_similarity/analyze_iou_vs_iou24.py --discord
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

EVAL_ROOT = REPO / 'eval_outputs' / 'cad_bench_722'
OUT_DIR   = EVAL_ROOT / 'iou_vs_iou24'

MODELS = [
    ('cadrille_rl',     'Cadrille-rl'),
    ('cadevolve_rl1',   'CADEvolve-rl1'),
    ('qwen25vl_3b_zs',  'Qwen2.5-VL-3B-zs'),
]
MODEL_COLORS = {
    'cadrille_rl':    '#4C72B0',
    'cadevolve_rl1':  '#55A868',
    'qwen25vl_3b_zs': '#C44E52',
}


def load_per_model() -> dict[str, list[dict]]:
    """Return slug → list of per-case records (only those with both iou and iou_24)."""
    out: dict[str, list[dict]] = {}
    for slug, _ in MODELS:
        rows = []
        with open(EVAL_ROOT / slug / 'metadata_24.jsonl') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get('iou') is None or r.get('iou_24') is None: continue
                rows.append(r)
        out[slug] = rows
    return out


def per_group_table(rows: list[dict], key_fn, label: str) -> str:
    """Build a markdown table grouping `rows` by `key_fn(row)`."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = [f'| {label:<22} | {"n":>5} | {"mean_iou":>9} | {"mean_iou_24":>11} '
           f'| {"Δ":>7} | {"pct_rot_win":>11} |',
           f'|{"-"*24}|{"-"*7}|{"-"*11}|{"-"*13}|{"-"*9}|{"-"*13}|']
    rows_sorted = sorted(groups.items(),
                         key=lambda kv: (-len(kv[1]), kv[0]))
    for k, grp in rows_sorted:
        n = len(grp)
        ious  = np.asarray([r['iou']    for r in grp])
        i24s  = np.asarray([r['iou_24'] for r in grp])
        rots  = np.asarray([r.get('rot_idx', -1) for r in grp])
        delta = float((i24s - ious).mean())
        pct_rot = float((rots > 0).mean()) * 100.0
        out.append(f'| {k[:22]:<22} | {n:>5} | {ious.mean():>9.4f} '
                   f'| {i24s.mean():>11.4f} | {delta:>+7.4f} | {pct_rot:>10.1f}% |')
    return '\n'.join(out)


def render_figures(per_model: dict[str, list[dict]]) -> dict[str, Path]:
    """Build scatter / histogram / rotation-distribution PNGs.
    Returns map of fig name → path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, Path] = {}

    # --- Scatter: iou (x) vs iou_24 (y), 1 panel per model ---
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 5), sharex=True, sharey=True)
    for ax, (slug, label) in zip(axes, MODELS):
        rows = per_model[slug]
        x = np.asarray([r['iou']    for r in rows])
        y = np.asarray([r['iou_24'] for r in rows])
        rot = np.asarray([r.get('rot_idx', -1) for r in rows])
        # color by whether identity (rot=0) or non-identity rotation won
        colors = ['#888' if ri == 0 else MODEL_COLORS[slug] for ri in rot]
        ax.scatter(x, y, c=colors, s=14, alpha=0.6, edgecolor='none')
        ax.plot([0, 1], [0, 1], '--', color='#333', alpha=0.5, linewidth=1)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('IoU (no rotation)')
        ax.set_aspect('equal')
        # annotate stats
        delta = (y - x).mean()
        n_rot_win = int((rot > 0).sum())
        ax.set_title(f'{label}\nn={len(rows)}  mean Δ={delta:+.3f}  '
                     f'rot_win={n_rot_win}/{len(rows)} '
                     f'({n_rot_win/len(rows)*100:.0f}%)', fontsize=10)
    axes[0].set_ylabel('IoU-24 (max over 24 rotations)')
    fig.suptitle('IoU vs IoU-24 per case — coloured points beat the y=x line by '
                 'using a non-identity rotation', fontsize=11)
    plt.tight_layout()
    p = OUT_DIR / 'scatter.png'
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    out_paths['scatter'] = p

    # --- Histogram: distribution of iou vs iou_24 per model ---
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.5), sharey=True)
    bins = np.linspace(0, 1, 31)
    for ax, (slug, label) in zip(axes, MODELS):
        rows = per_model[slug]
        ious  = [r['iou']    for r in rows]
        i24s  = [r['iou_24'] for r in rows]
        ax.hist(ious, bins=bins, alpha=0.55, label='IoU',
                color='#999', edgecolor='black', linewidth=0.3)
        ax.hist(i24s, bins=bins, alpha=0.6,  label='IoU-24',
                color=MODEL_COLORS[slug], edgecolor='black', linewidth=0.3)
        ax.set_xlabel('IoU')
        ax.set_title(f'{label} (n={len(rows)})', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
    axes[0].set_ylabel('# cases')
    fig.suptitle('IoU vs IoU-24 distribution per model', fontsize=11)
    plt.tight_layout()
    p = OUT_DIR / 'histogram.png'
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    out_paths['histogram'] = p

    # --- Rotation idx winning distribution ---
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4), sharey=True)
    for ax, (slug, label) in zip(axes, MODELS):
        rows = per_model[slug]
        rots = [r.get('rot_idx', -1) for r in rows if r.get('rot_idx', -1) >= 0]
        counts = Counter(rots)
        xs = list(range(24))
        ys = [counts.get(i, 0) for i in xs]
        bars = ax.bar(xs, ys, color=MODEL_COLORS[slug], alpha=0.85, edgecolor='black',
                      linewidth=0.3)
        # highlight identity differently
        bars[0].set_color('#bbb')
        bars[0].set_edgecolor('black')
        ax.set_xlabel('rotation idx (0=identity)')
        ax.set_xticks([0, 5, 10, 15, 20, 23])
        ax.set_title(f'{label}: top winning rotations', fontsize=10)
        # annotate top-3 non-identity
        non_id = [(i, c) for i, c in counts.items() if i > 0]
        non_id.sort(key=lambda x: -x[1])
        for i, c in non_id[:3]:
            ax.annotate(f'r{i}', xy=(i, c), xytext=(0, 4),
                        textcoords='offset points', ha='center', fontsize=8)
    axes[0].set_ylabel('# cases winning at this rotation')
    fig.suptitle('Which of the 24 rotations wins most often (per model)', fontsize=11)
    plt.tight_layout()
    p = OUT_DIR / 'rotation_dist.png'
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    out_paths['rotation_dist'] = p

    return out_paths


def post_to_discord(text: str, files: list[Path]) -> None:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('  no DISCORD_WEBHOOK_URL — skipping ping'); return
    import urllib.request
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    def w(s): body.write(s.encode())
    w(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n')
    w('Content-Type: application/json\r\n\r\n')
    w(json.dumps({'content': text}) + '\r\n')
    for i, p in enumerate(files):
        w(f'--{boundary}\r\nContent-Disposition: form-data; name="files[{i}]"; '
          f'filename="{p.name}"\r\nContent-Type: image/png\r\n\r\n')
        body.write(p.read_bytes()); w('\r\n')
    w(f'--{boundary}--\r\n')
    req = urllib.request.Request(url, data=body.getvalue(), headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'cad-bench-722-iouvsiou24/1.0'})
    try:
        urllib.request.urlopen(req, timeout=60).read()
        print(f'  posted to Discord ({len(files)} files) ✓')
    except Exception as e:
        print(f'  Discord post failed: {e}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--discord', action='store_true')
    ap.add_argument('--top-families', type=int, default=15)
    args = ap.parse_args()

    print('Loading per-model metadata_24.jsonl …', flush=True)
    per_model = load_per_model()
    for slug, _ in MODELS:
        print(f'  {slug}: {len(per_model[slug])} cases with both iou and iou_24',
              flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build markdown report ───────────────────────────────────────────
    lines = ['# IoU vs IoU-24 — `cad_bench_722` rotation-rescue analysis',
             '',
             'Source: per-model `metadata_24.jsonl`. We exclude cases where the '
             'model failed to exec (no pred mesh = no IoU to compare).',
             '',
             '`Δ = mean(iou_24 − iou)` over included cases. '
             '`pct_rot_win` = fraction of cases where a non-identity rotation '
             'beat the identity, i.e. the prediction is correct shape but '
             'oriented wrong by a multiple of 90°.',
             '']

    # Per-model summary
    lines.append('## Per-model')
    lines.append('')
    lines.append(f'| {"model":<22} | {"n":>5} | {"mean_iou":>9} | '
                 f'{"mean_iou_24":>11} | {"Δ":>7} | {"pct_rot_win":>11} |')
    lines.append(f'|{"-"*24}|{"-"*7}|{"-"*11}|{"-"*13}|{"-"*9}|{"-"*13}|')
    for slug, label in MODELS:
        rows = per_model[slug]
        ious = np.asarray([r['iou']    for r in rows])
        i24s = np.asarray([r['iou_24'] for r in rows])
        rots = np.asarray([r.get('rot_idx', -1) for r in rows])
        delta   = float((i24s - ious).mean())
        pct_rot = float((rots > 0).mean()) * 100.0
        lines.append(f'| {label:<22} | {len(rows):>5} | {ious.mean():>9.4f} '
                     f'| {i24s.mean():>11.4f} | {delta:>+7.4f} '
                     f'| {pct_rot:>10.1f}% |')

    # Per-difficulty
    lines.append('')
    lines.append('## Per-difficulty (per model)')
    for slug, label in MODELS:
        lines.append(f'\n### {label}')
        lines.append('')
        lines.append(per_group_table(per_model[slug],
                                     key_fn=lambda r: r.get('difficulty', '?'),
                                     label='difficulty'))

    # Per-family — top N by sample count
    lines.append('')
    lines.append(f'## Per-family — top {args.top_families} families by # cases (per model)')
    for slug, label in MODELS:
        lines.append(f'\n### {label}')
        lines.append('')
        # Build family stats then keep top N
        fam_stats = defaultdict(list)
        for r in per_model[slug]:
            fam_stats[r.get('family', '?')].append(r)
        top = sorted(fam_stats.items(), key=lambda kv: -len(kv[1]))[:args.top_families]
        sub_rows = [r for _, grp in top for r in grp]
        lines.append(per_group_table(sub_rows,
                                     key_fn=lambda r: r.get('family', '?'),
                                     label='family'))

    # Top examples — biggest rescues per model
    lines.append('')
    lines.append('## Biggest IoU-24 rescues (top 5 per model)')
    for slug, label in MODELS:
        rows = per_model[slug]
        gains = sorted(rows, key=lambda r: -(r['iou_24'] - r['iou']))[:5]
        lines.append(f'\n### {label}')
        lines.append('')
        lines.append(f'| {"stem":<48} | {"family":<22} | {"diff":<6} | '
                     f'{"iou":>6} | {"iou_24":>7} | {"Δ":>7} | {"r_idx":>5} |')
        lines.append(f'|{"-"*50}|{"-"*24}|{"-"*8}|{"-"*8}|{"-"*9}|{"-"*9}|{"-"*7}|')
        for r in gains:
            lines.append(f'| {r["stem"][:48]:<48} | '
                         f'{r.get("family", "?")[:22]:<22} | '
                         f'{r.get("difficulty", "?")[:6]:<6} | '
                         f'{r["iou"]:>6.3f} | {r["iou_24"]:>7.3f} | '
                         f'{r["iou_24"] - r["iou"]:>+7.3f} | '
                         f'{r.get("rot_idx", -1):>5} |')

    report_md = '\n'.join(lines)
    (OUT_DIR / 'report.md').write_text(report_md)
    print(f'\nWrote {OUT_DIR / "report.md"} ({len(report_md)} chars)', flush=True)

    # ── Render figures ──────────────────────────────────────────────────
    print('Rendering figures …', flush=True)
    figs = render_figures(per_model)
    for name, p in figs.items():
        sz = p.stat().st_size / 1024
        print(f'  {name}: {p}  ({sz:.0f} kB)', flush=True)

    # ── Discord ─────────────────────────────────────────────────────────
    if args.discord:
        # Build a compact per-model summary for the message body
        msg_lines = ['📐 **cad_bench_722 — IoU vs IoU-24 rotation-rescue analysis**',
                     '```',
                     f'{"model":<22} {"n":>5} {"iou":>7} {"iou_24":>8} {"Δ":>7} {"rot_win":>9}',
                     '-' * 62]
        for slug, label in MODELS:
            rows = per_model[slug]
            ious = np.asarray([r['iou']    for r in rows])
            i24s = np.asarray([r['iou_24'] for r in rows])
            rots = np.asarray([r.get('rot_idx', -1) for r in rows])
            delta   = float((i24s - ious).mean())
            pct_rot = float((rots > 0).mean()) * 100.0
            msg_lines.append(f'{label:<22} {len(rows):>5} {ious.mean():>7.4f} '
                             f'{i24s.mean():>8.4f} {delta:>+7.4f} '
                             f'{pct_rot:>8.1f}%')
        msg_lines.append('```')
        msg_lines.append('Read: Δ = mean iou_24 − mean iou over paired cases. '
                         'rot_win = fraction whose best rotation ≠ identity '
                         '(i.e. correct shape but oriented 90°/180°/270° off).')
        msg_lines.append('Attached: scatter, histogram, rotation-distribution. '
                         'Full markdown table at '
                         '`eval_outputs/cad_bench_722/iou_vs_iou24/report.md`.')
        post_to_discord('\n'.join(msg_lines),
                        [figs['scatter'], figs['histogram'], figs['rotation_dist']])


if __name__ == '__main__':
    main()
