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
    'ood_v2',
    'iid_enhanced_v2',
    'iid_v2',
    'baseline_v2',
]
COLORS = {
    'ood_enhanced_v2': '#d62728',  # red
    'ood_v2':          '#ff7f0e',  # orange
    'iid_enhanced_v2': '#1f77b4',  # blue
    'iid_v2':          '#2ca02c',  # green
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
    """Walk logs/, pick up all per-config rows from the CURRENT chain only.

    Filter logs by filename timestamp >= chain_start_ts (defaults to the
    last `=== chain start` line in logs/launch_chain_v2.log). Skips
    failed-start logs that contain a step=0 eval but never moved past it.

    Special case: ood_enhanced_v2 may resume from an earlier ckpt (its
    config sets resume_from_checkpoint), so we always include ALL its
    logs from the last 24h to keep step 0..current contiguous on the plot.

    When multiple logs exist for the same config, dedupe by (step, bucket)
    — later log wins (handles resume overlap cleanly).
    """
    if chain_start_ts is None:
        chain_start_ts = _current_chain_start()
    by_config: dict[str, dict[tuple[int, str], dict]] = {c: {} for c in CONFIGS}
    for cfg in CONFIGS:
        log_files = sorted(LOGS_DIR.glob(f'{cfg}_*.log'))
        for lf in log_files:
            m = _LOG_FILENAME_RX.search(lf.name)
            if not m:
                continue
            file_ts = m.group('ts')
            # Skip logs from earlier chain runs unless this is the resumed
            # config (currently ood_enhanced_v2 — see config yaml).
            if chain_start_ts and file_ts < chain_start_ts and cfg != 'ood_enhanced_v2':
                continue
            rows = parse_log(lf)
            # Skip failed-start logs (only step=0, never advanced)
            if {r['step'] for r in rows} == {0} and len(rows) <= 4:
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


def plot_metric(by_config: dict[str, list[dict]], metric: str, title: str,
                out_path: Path) -> int:
    """Plot one metric (iou or ess) as a 2×2 panel of bucket curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    n_lines = 0
    for ax, bucket in zip(axes, BUCKETS):
        for cfg, rows in by_config.items():
            if not rows:
                continue
            xs = [r['step'] for r in rows if r['bucket'] == bucket]
            ys = [r[metric] for r in rows if r['bucket'] == bucket]
            if not xs:
                continue
            ax.plot(xs, ys, '-o', color=COLORS[cfg], label=cfg, markersize=3,
                    linewidth=1.5)
            n_lines += 1
        ax.set_title(bucket, fontsize=10)
        ax.set_xlabel('step')
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=5,
                   bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(title, fontsize=12, y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return n_lines


def post_to_discord(iou_path: Path, ess_path: Path,
                    by_config: dict[str, list[dict]]) -> None:
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        print('DISCORD_WEBHOOK_URL not set — skipping post')
        return
    # Build the latest-step summary
    summary_lines = ['📊 **§7 v2 fig — auto refresh**', '', '```']
    for cfg in CONFIGS:
        rows = by_config.get(cfg, [])
        if not rows:
            summary_lines.append(f'{cfg:<22} no data')
            continue
        latest = max(r['step'] for r in rows)
        latest_rows = [r for r in rows if r['step'] == latest]
        bc_ood = next((r for r in latest_rows if r['bucket'] == 'BenchCAD val OOD'), None)
        iso_ood = next((r for r in latest_rows if r['bucket'] == 'iso val OOD'), None)
        if bc_ood and iso_ood:
            summary_lines.append(
                f'{cfg:<22} step={latest:>5}  BC_OOD IoU={bc_ood["iou"]:.3f}  '
                f'iso_OOD IoU={iso_ood["iou"]:.3f}  ess={bc_ood["ess"]:.3f}/{iso_ood["ess"]:.3f}'
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
    # JSON payload (content)
    _write(f'--{boundary}\r\n')
    _write('Content-Disposition: form-data; name="payload_json"\r\n')
    _write('Content-Type: application/json\r\n\r\n')
    _write(json.dumps({'content': content}))
    _write('\r\n')
    # Attachment 1: IoU panel
    for i, (path, label) in enumerate([(iou_path, 'fig7_iou.png'),
                                        (ess_path, 'fig7_ess.png')]):
        _write(f'--{boundary}\r\n')
        _write(f'Content-Disposition: form-data; name="files[{i}]"; filename="{label}"\r\n')
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
    iou_path = args.out_dir / 'fig7_v2_iou.png'
    ess_path = args.out_dir / 'fig7_v2_ess.png'
    csv_path = args.out_dir / 'fig7_v2_metrics.csv'

    n_iou_lines = plot_metric(by_config, 'iou',
                              'IoU vs step (greedy, n=50 per bucket)', iou_path)
    n_ess_lines = plot_metric(by_config, 'ess',
                              'ess_pass vs step (n=50 per bucket, OOD = held-out 10 mech fams)',
                              ess_path)
    write_csv(by_config, csv_path)
    print(f'\n→ {iou_path}  ({n_iou_lines} lines)')
    print(f'→ {ess_path}  ({n_ess_lines} lines)')
    print(f'→ {csv_path}')

    if args.post:
        post_to_discord(iou_path, ess_path, by_config)


if __name__ == '__main__':
    sys.exit(main())
