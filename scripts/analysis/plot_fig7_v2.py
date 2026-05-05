"""Live §7 v2 fig — IoU + ess_pass vs step, 4 buckets × 5 configs.

Parses training logs for the 5 v2 configs (ood_enhanced_v2 / ood_v2 /
iid_enhanced_v2 / iid_v2 / baseline_v2), extracts every eval tick, plots
IoU and ess_pass curves with one line per config × 4 val buckets.

Designed to be re-run on every eval tick — idempotent, fast (~1s).

Output:
  /tmp/fig7_v2_iou.png        4 panels: BC IID, BC OOD, iso IID, iso OOD
  /tmp/fig7_v2_ess.png        same layout, ess_pass instead of IoU
  /tmp/fig7_v2_metrics.csv    long-format CSV: run_name,config,step,bucket,IoU,...

Usage:
    uv run python -m scripts.analysis.plot_fig7_v2          # plot only
    uv run python -m scripts.analysis.plot_fig7_v2 --post   # plot + dc post
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = REPO / 'logs'

CONFIGS = [
    'ood_enhanced_v2',
    # ood_v2 was a partial re-run on this HPC (stopped at step 2k); excluded
    # from the paper figure since the friend HPC owns the canonical ood line.
    # 'ood_v2',
    # iid_enhanced_v2 skipped per option A (redundant with iid_v2 by symmetry).
    # 'iid_enhanced_v2',
    'iid_v2',
    'baseline_v2',
]
# Display labels for the legend (paper-friendly names).
LABELS = {
    'iid_v2':          'iid',
    'ood_enhanced_v2': 'ood',     # 'ood' in paper === ood_enhanced_v2 (with simple)
    'baseline_v2':     'baseline',
}
COLORS = {
    'iid_v2':          '#2ca02c',  # green
    'ood_enhanced_v2': '#d62728',  # red
    'baseline_v2':     '#7f7f7f',  # gray
}
BUCKETS = ['BenchCAD val IID', 'BenchCAD val OOD', 'iso val IID', 'iso val OOD']

# Match a single eval-bucket line in the training log:
#   [img/BenchCAD val IID] op_loss_w=0.493  recall=0.383  rare_recall=0.340  IoU=0.088  exec=94.0%  ess_pass=0.488(n=41)  ...
LINE_RX = re.compile(
    r'\[img/(?P<bucket>[^\]]+)\] '
    r'op_loss_w=(?P<op_loss>[\d.]+)\s+'
    r'recall=(?P<recall>[\d.]+)\s+'
    r'rare_recall=(?P<rare_recall>[\d.]+)\s+'
    r'IoU=(?P<iou>[\d.]+)\s+'
    r'exec=(?P<exec>[\d.]+)%\s+'
    r'ess_pass=(?P<ess>[\d.]+)\((?:n=(?P<ess_n>\d+))?\)'
)
# Match the step marker that precedes a block of bucket lines.
STEP_RX = re.compile(r'step=(?P<step>\d+) running IoU eval')


def parse_log(path: Path) -> list[dict]:
    """Return a list of {step, bucket, IoU, ess_pass, exec, op_loss, recall,
    rare_recall, ess_n} dicts from a single training log."""
    rows: list[dict] = []
    text = path.read_text(errors='ignore')
    cur_step: int | None = None
    for line in text.splitlines():
        m_step = STEP_RX.search(line)
        if m_step:
            cur_step = int(m_step.group('step'))
            continue
        m = LINE_RX.search(line)
        if m and cur_step is not None and m.group('bucket') in BUCKETS:
            rows.append({
                'step':        cur_step,
                'bucket':      m.group('bucket'),
                'iou':         float(m.group('iou')),
                'ess':         float(m.group('ess')),
                'exec':        float(m.group('exec')),
                'op_loss':     float(m.group('op_loss')),
                'recall':      float(m.group('recall')),
                'rare_recall': float(m.group('rare_recall')),
                'ess_n':       int(m.group('ess_n') or 0),
            })
    return rows


_CHAIN_START_RX = re.compile(r'\[(?P<ts>[\d\-T:]+)\] === chain start')
_LOG_FILENAME_RX = re.compile(r'(?P<cfg>\w+)_(?P<ts>\d{8}_\d{6})\.log$')


def _current_chain_start() -> str | None:
    """Read logs/launch_chain_v2.log and return the last '=== chain start' ts
    as 'YYYYMMDD_HHMMSS' (matches log filename ts format). None if no chain log."""
    p = LOGS_DIR / 'launch_chain_v2.log'
    if not p.exists():
        return None
    matches = _CHAIN_START_RX.findall(p.read_text(errors='ignore'))
    if not matches:
        return None
    last_ts = matches[-1]  # ISO format '2026-05-03T09:10:08'
    # Strip non-digits to compare with filename ts
    return last_ts.replace('-', '').replace('T', '_').replace(':', '')


def collect_all(chain_start_ts: str | None = None) -> dict[str, list[dict]]:
    """Walk logs/, pick up all per-config rows.

    For each config, includes:
      * the LATEST log that has substantive eval data (>4 rows, post-step-0)
      * earlier logs with the same config name that have substantive data
        (handles resumed runs and chain-run-across-sessions)

    Excludes failed-start logs (only step=0, ≤ 4 rows) — those are aborted
    runs that never moved past eval-on-start.

    `chain_start_ts` is no longer used as a hard cutoff (was too aggressive
    — would drop completed prior-chain runs whose logs predate the current
    chain start). The 'failed-start' filter alone now handles noise.

    When multiple logs exist for the same config, dedupe by (step, bucket)
    — later log wins (handles resume overlap cleanly).
    """
    by_config: dict[str, dict[tuple[int, str], dict]] = {c: {} for c in CONFIGS}
    for cfg in CONFIGS:
        log_files = sorted(LOGS_DIR.glob(f'{cfg}_*.log'))
        for lf in log_files:
            m = _LOG_FILENAME_RX.search(lf.name)
            if not m:
                continue
            rows = parse_log(lf)
            # Skip failed-start logs (only step=0, never advanced)
            if {r['step'] for r in rows} == {0} and len(rows) <= 4:
                continue
            if not rows:
                continue
            for r in rows:
                key = (r['step'], r['bucket'])
                by_config[cfg][key] = r
    out: dict[str, list[dict]] = {}
    for cfg, d in by_config.items():
        out[cfg] = sorted(d.values(), key=lambda r: (r['step'], r['bucket']))
    return out


def write_csv(by_config: dict[str, list[dict]], path: Path) -> None:
    fields = ['config', 'step', 'bucket', 'iou', 'ess', 'exec',
              'op_loss', 'recall', 'rare_recall', 'ess_n']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cfg, rows in by_config.items():
            for r in rows:
                w.writerow({'config': cfg, **r})


def plot_single_panel(by_config: dict[str, list[dict]],
                      bucket: str, metric: str,
                      title: str, ylabel: str,
                      out_path: Path) -> int:
    """Single-panel fig: one bucket × one metric, lines per config.

    Matches the §7 paper figure layout (each metric/bucket gets its own fig).
    Paper-style emphasis: thick lines, large fonts, simple labels.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    n_lines = 0
    # Plot in order of CONFIGS so legend ordering is deterministic.
    for cfg in CONFIGS:
        rows = by_config.get(cfg, [])
        if not rows or cfg not in COLORS:
            continue
        xs = [r['step'] for r in rows if r['bucket'] == bucket]
        ys = [r[metric] for r in rows if r['bucket'] == bucket]
        if not xs:
            continue
        ax.plot(xs, ys, '-o', color=COLORS[cfg], label=LABELS.get(cfg, cfg),
                markersize=7, linewidth=3.5, alpha=0.95)
        n_lines += 1
    ax.set_xlabel('training step', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=17)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis='both', labelsize=14)
    if n_lines > 0:
        ax.legend(loc='best', fontsize=15, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return n_lines


def post_to_discord(fig_paths: list[Path],
                    by_config: dict[str, list[dict]]) -> None:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('DISCORD_WEBHOOK_URL not set — skipping post')
        return
    # Build the latest-step summary on BC val (paper-headline source)
    summary_lines = ['📊 **§7 v2 — IID/OOD × IoU/ess (BC val)**', '', '```']
    summary_lines.append(f'{"config":<18}{"step":>6}  {"IID IoU":>8}{"IID ess":>9}  {"OOD IoU":>8}{"OOD ess":>9}')
    for cfg in CONFIGS:
        rows = by_config.get(cfg, [])
        if not rows:
            summary_lines.append(f'{cfg:<18}{"—":>6}    no data')
            continue
        latest = max(r['step'] for r in rows)
        latest_rows = [r for r in rows if r['step'] == latest]
        bc_iid = next((r for r in latest_rows if r['bucket'] == 'BenchCAD val IID'), None)
        bc_ood = next((r for r in latest_rows if r['bucket'] == 'BenchCAD val OOD'), None)
        if bc_iid and bc_ood:
            summary_lines.append(
                f'{cfg:<18}{latest:>6}  {bc_iid["iou"]:>8.3f}{bc_iid["ess"]:>9.3f}'
                f'  {bc_ood["iou"]:>8.3f}{bc_ood["ess"]:>9.3f}'
            )
    summary_lines.append('```')
    content = '\n'.join(summary_lines)

    # Multi-part form upload (Discord webhooks support attachments via multipart/form-data)
    boundary = '----cadrille-fig7-' + os.urandom(8).hex()
    body = io.BytesIO()
    def _write(s):
        if isinstance(s, str):
            s = s.encode()
        body.write(s)
    _write(f'--{boundary}\r\n')
    _write('Content-Disposition: form-data; name="payload_json"\r\n')
    _write('Content-Type: application/json\r\n\r\n')
    _write(json.dumps({'content': content}))
    _write('\r\n')
    for i, path in enumerate(fig_paths):
        _write(f'--{boundary}\r\n')
        _write(f'Content-Disposition: form-data; name="files[{i}]"; filename="{path.name}"\r\n')
        _write('Content-Type: image/png\r\n\r\n')
        _write(path.read_bytes())
        _write('\r\n')
    _write(f'--{boundary}--\r\n')
    req = urllib.request.Request(
        url, data=body.getvalue(),
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'User-Agent': 'cadrille-trainer/1.0',
        },
        method='POST',
    )
    try:
        resp = urllib.request.urlopen(req)
        print(f'discord post: HTTP {resp.status}')
    except Exception as e:
        print(f'discord post failed: {e!r}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--post', action='store_true', help='Post to Discord')
    ap.add_argument('--out-dir', type=Path, default=Path('/tmp'),
                    help='Output dir for fig + csv (default: /tmp)')
    args = ap.parse_args()

    by_config = collect_all()
    n_total = sum(len(rs) for rs in by_config.values())
    print(f'parsed {n_total} eval rows across {sum(1 for v in by_config.values() if v)} configs:')
    for cfg in CONFIGS:
        rows = by_config[cfg]
        if not rows:
            print(f'  {cfg:<22} (no data)')
            continue
        steps = sorted({r['step'] for r in rows})
        print(f'  {cfg:<22} {len(rows)} rows, steps={steps[0]}..{steps[-1]} ({len(steps)} ticks)')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / 'fig7_v2_metrics.csv'
    write_csv(by_config, csv_path)
    print(f'\n→ {csv_path}')

    # 4 single-panel figs (paper layout): IID/OOD × IoU/ess on BC val.
    # iso/DC/Fu still in the CSV for follow-up analysis.
    panels = [
        ('BenchCAD val IID', 'iou', 'IID IoU (BenchCAD val IID, n=50)',
         'IoU (greedy, mean over 50)', args.out_dir / 'fig7_iid_iou.png'),
        ('BenchCAD val IID', 'ess', 'IID ess_ops (BenchCAD val IID, n=50)',
         'essential_pass rate', args.out_dir / 'fig7_iid_ess.png'),
        ('BenchCAD val OOD', 'iou', 'OOD IoU (BenchCAD val OOD, 10 held-out mech families)',
         'IoU (greedy, mean over 50)', args.out_dir / 'fig7_ood_iou.png'),
        ('BenchCAD val OOD', 'ess', 'OOD ess_ops (BenchCAD val OOD, 10 held-out mech families)',
         'essential_pass rate', args.out_dir / 'fig7_ood_ess.png'),
    ]
    fig_paths: list[Path] = []
    for bucket, metric, title, ylabel, path in panels:
        n = plot_single_panel(by_config, bucket, metric, title, ylabel, path)
        print(f'→ {path}  ({n} lines)')
        if n > 0:
            fig_paths.append(path)

    if args.post:
        post_to_discord(fig_paths, by_config)


if __name__ == '__main__':
    sys.exit(main())