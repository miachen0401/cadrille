"""Plot IoU curves across runs in experiments_log/.

Reads each run's `eval_metrics.csv`, draws step-vs-IoU curves for
BenchCAD val / DeepCAD test / Fusion360 test, one panel per bucket.

Output: experiments_log/iou_curves.png

Usage:
    uv run python -m scripts.analysis.plot_experiments_log
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / 'experiments_log'

BUCKETS = ('BenchCAD val', 'DeepCAD test', 'Fusion360 test')
PALETTE = {
    'curriculum_qwen3vl_2b_20260425': ('#2E86AB', 'curriculum (20k)'),
    'big_bench_shell_50k_20260427':   ('#E63946', 'big-bench-shell (50k)'),
}


def load_run(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """{bucket: [(step, iou)]}"""
    out: dict[str, list[tuple[int, float]]] = {b: [] for b in BUCKETS}
    csv_path = run_dir / 'eval_metrics.csv'
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row['bucket'] not in out:
                continue
            iou = row.get('iou')
            step = row.get('step')
            if not iou or iou == '' or not step:
                continue
            try:
                out[row['bucket']].append((int(step), float(iou)))
            except ValueError:
                continue
    for b in out:
        out[b].sort()
    return out


def main():
    runs = []
    for d in sorted(EXP_DIR.iterdir()):
        if not d.is_dir(): continue
        if (d / 'eval_metrics.csv').exists():
            runs.append((d.name, load_run(d)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, bucket in zip(axes, BUCKETS):
        for run_name, data in runs:
            color, label = PALETTE.get(run_name, (None, run_name))
            pts = data.get(bucket, [])
            if not pts: continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker='o', markersize=4, color=color, label=label,
                    linewidth=1.6)
        ax.set_xlabel('step')
        ax.set_title(bucket)
        ax.grid(alpha=0.25)
        ax.set_ylim(0.0, 0.8)
    axes[0].set_ylabel('IoU')
    axes[0].legend(loc='lower right', framealpha=0.85)

    fig.suptitle('cadrille SFT runs — eval IoU vs training step', fontsize=14)
    fig.tight_layout()
    out_png = EXP_DIR / 'iou_curves.png'
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f'wrote {out_png}')


if __name__ == '__main__':
    main()
