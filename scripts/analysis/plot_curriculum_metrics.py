"""Parse curriculum_qwen3vl_2b training log + plot rare_recall + op_loss + IoU.

Pulls every per-step online_eval line out of the log file and produces 3 figures:
  - rare_recall vs step (per dataset)
  - op_loss_w vs step  (per dataset)
  - distinct_ops vs step (per dataset)
plus a combined 4-panel figure for the paper.

Usage:
  uv run python -m scripts.analysis.plot_curriculum_metrics \
      --log logs/curriculum_qwen3vl_2b_20260425_192907.log \
      --out eval_outputs/curriculum_metrics_paper
"""
import argparse
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Two regexes — log has 2 line shapes:
#  (A) sources with GT code (BenchCAD val, recode20k train, text2cad train):
#      [img/BenchCAD val] op_loss_w=0.350  recall=0.566  rare_recall=0.442  IoU=0.523  exec=85.0%  distinct_ops=20  ...
#  (B) sources WITHOUT GT code (DeepCAD test, Fusion360 test):
#      [img/DeepCAD test] IoU=0.477  exec=95.0%  distinct_ops=10  distinct_codes=1.00  (n=20)
LINE_FULL = re.compile(
    r'\[(?P<modality>img|text)/(?P<dataset>[A-Za-z0-9 ]+?)\]\s+'
    r'op_loss_w=(?P<op_loss>-?[\d.]+)\s+'
    r'recall=(?P<recall>[\d.]+)\s+'
    r'rare_recall=(?P<rare>[\d.]+)'
    r'(?:\s+IoU=(?P<iou>[\d.]+))?'
    r'.*?distinct_ops=(?P<distinct>\d+)'
)
LINE_IOU_ONLY = re.compile(
    r'\[(?P<modality>img|text)/(?P<dataset>[A-Za-z0-9 ]+?)\]\s+'
    r'IoU=(?P<iou>[\d.]+)\s+exec=[\d.]+%\s+distinct_ops=(?P<distinct>\d+)'
)
STEP_LINE = re.compile(r'\[online-eval\] step=(?P<step>\d+) running IoU eval')


def parse_log(log_path: Path) -> dict:
    """Returns: {dataset_label: list of (step, op_loss, recall, rare, iou, distinct)}"""
    rows_by_ds = {}
    cur_step = None
    for line in log_path.read_text().splitlines():
        s = STEP_LINE.search(line)
        if s:
            cur_step = int(s.group('step'))
            continue
        if cur_step is None:
            continue
        m = LINE_FULL.search(line)
        if m:
            ds = m.group('dataset').strip()
            rows_by_ds.setdefault(ds, []).append({
                'step': cur_step,
                'op_loss_w': float(m.group('op_loss')),
                'recall': float(m.group('recall')),
                'rare_recall': float(m.group('rare')),
                'iou': float(m.group('iou')) if m.group('iou') else None,
                'distinct_ops': int(m.group('distinct')),
            })
            continue
        m = LINE_IOU_ONLY.search(line)
        if m:
            ds = m.group('dataset').strip()
            rows_by_ds.setdefault(ds, []).append({
                'step': cur_step,
                'op_loss_w': None,
                'recall': None,
                'rare_recall': None,
                'iou': float(m.group('iou')),
                'distinct_ops': int(m.group('distinct')),
            })
    return rows_by_ds


# Datasets and color scheme
ORDER = ['BenchCAD val', 'recode20k train', 'text2cad train',
         'DeepCAD test', 'Fusion360 test']
COLORS = {
    'BenchCAD val':    '#ff7f0e',
    'recode20k train': '#9467bd',
    'text2cad train':  '#7f7f7f',
    'DeepCAD test':    '#1f77b4',
    'Fusion360 test':  '#2ca02c',
}


def _xs_ys(rows, key):
    xs = [r['step'] for r in rows]
    ys = [r[key] if r.get(key) is not None else np.nan for r in rows]
    return xs, ys


def _plot_with_phase_bands(ax):
    """Phase 1 (0-5k), Phase 2 (5k-10k), Phase 3 (10k-20k) shading."""
    ax.axvspan(0, 5000, alpha=0.05, color='blue')
    ax.axvspan(5000, 10000, alpha=0.05, color='green')
    ax.axvspan(10000, 20000, alpha=0.05, color='red')
    ax.text(2500, ax.get_ylim()[1] * 0.95, 'P1: 1:2:2', ha='center',
             fontsize=9, color='blue', alpha=0.7)
    ax.text(7500, ax.get_ylim()[1] * 0.95, 'P2: 2:1:1', ha='center',
             fontsize=9, color='green', alpha=0.7)
    ax.text(15000, ax.get_ylim()[1] * 0.95, 'P3: 8:1:1 (BenchCAD heavy)', ha='center',
             fontsize=9, color='red', alpha=0.7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = parse_log(Path(args.log))
    print('Datasets parsed:')
    for ds, rows in data.items():
        print(f'  {ds:>20s}: {len(rows)} steps')

    # === Figure 1: 2x2 paper-ready layout ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (a_iou, a_rare), (a_loss, a_distinct) = axes

    # IoU (only datasets that have IoU: BenchCAD val, DeepCAD test, Fusion360 test)
    for ds in ['BenchCAD val', 'DeepCAD test', 'Fusion360 test']:
        rows = data.get(ds, [])
        if not rows: continue
        xs, ys = _xs_ys(rows, 'iou')
        a_iou.plot(xs, ys, color=COLORS[ds], marker='o', markersize=3,
                    label=ds, linewidth=2)
    a_iou.set_xlabel('curriculum step')
    a_iou.set_ylabel('IoU (greedy, n=20)')
    a_iou.set_title('(a) Online IoU per eval bucket')
    a_iou.set_ylim(-0.02, 1.02)
    a_iou.grid(True, alpha=0.3)
    a_iou.legend(loc='lower right', fontsize=9)
    _plot_with_phase_bands(a_iou)

    # rare_recall (BenchCAD has rare ops; recode20k has zero rare; text2cad usually 1.0)
    for ds in ['BenchCAD val', 'recode20k train', 'text2cad train']:
        rows = data.get(ds, [])
        if not rows: continue
        xs, ys = _xs_ys(rows, 'rare_recall')
        a_rare.plot(xs, ys, color=COLORS[ds], marker='o', markersize=3,
                     label=ds, linewidth=2)
    a_rare.set_xlabel('curriculum step')
    a_rare.set_ylabel('rare-op macro recall')
    a_rare.set_title('(b) Rare-op (P≤0.20) recall per source')
    a_rare.set_ylim(-0.02, 1.05)
    a_rare.grid(True, alpha=0.3)
    a_rare.legend(loc='lower right', fontsize=9)
    _plot_with_phase_bands(a_rare)

    # op_loss_w
    for ds in ['BenchCAD val', 'recode20k train', 'text2cad train']:
        rows = data.get(ds, [])
        if not rows: continue
        xs, ys = _xs_ys(rows, 'op_loss_w')
        a_loss.plot(xs, ys, color=COLORS[ds], marker='o', markersize=3,
                     label=ds, linewidth=2)
    a_loss.set_xlabel('curriculum step')
    a_loss.set_ylabel('op loss (cosine-weighted)')
    a_loss.set_title('(c) Op-prediction loss (lower = better, weighted by -log P_global)')
    a_loss.set_ylim(-0.05, 1.05)
    a_loss.grid(True, alpha=0.3)
    a_loss.legend(loc='upper right', fontsize=9)
    _plot_with_phase_bands(a_loss)

    # distinct_ops
    for ds in ['BenchCAD val', 'recode20k train', 'text2cad train',
                'DeepCAD test', 'Fusion360 test']:
        rows = data.get(ds, [])
        if not rows: continue
        xs, ys = _xs_ys(rows, 'distinct_ops')
        a_distinct.plot(xs, ys, color=COLORS[ds], marker='o', markersize=3,
                         label=ds, linewidth=2)
    a_distinct.set_xlabel('curriculum step')
    a_distinct.set_ylabel('distinct ops produced (per 20-sample batch)')
    a_distinct.set_title('(d) Op vocabulary diversity')
    a_distinct.set_ylim(0, 30)
    a_distinct.grid(True, alpha=0.3)
    a_distinct.legend(loc='lower right', fontsize=9)
    _plot_with_phase_bands(a_distinct)

    fig.suptitle('Curriculum Qwen3-VL-2B SFT — online eval trajectories', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / 'paper_fig_curriculum_4panel.png', dpi=140)
    plt.close(fig)
    print(f'  → paper_fig_curriculum_4panel.png')

    # === Standalone single-panel exports for paper ===
    for tag, ykey, ylabel, ds_list, ylim, title, fname in [
        ('iou', 'iou', 'IoU (greedy)', ['BenchCAD val', 'DeepCAD test', 'Fusion360 test'],
          (-0.02, 1.02), 'IoU per eval bucket', 'iou_vs_step.png'),
        ('rare', 'rare_recall', 'rare-op macro recall',
          ['BenchCAD val', 'recode20k train'], (-0.02, 1.05),
          'Rare-op recall (BenchCAD val + recode20k train)', 'rare_recall_vs_step.png'),
        ('loss', 'op_loss_w', 'op loss (cosine-weighted)',
          ['BenchCAD val', 'recode20k train'], (-0.05, 1.05),
          'Op-prediction loss', 'op_loss_vs_step.png'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ds in ds_list:
            rows = data.get(ds, [])
            if not rows: continue
            xs, ys = _xs_ys(rows, ykey)
            ax.plot(xs, ys, color=COLORS[ds], marker='o', markersize=4,
                     label=ds, linewidth=2)
        ax.set_xlabel('curriculum step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        _plot_with_phase_bands(ax)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=140)
        plt.close(fig)
        print(f'  → {fname}')

    # CSV export for paper tables
    csv_path = out / 'curriculum_metrics.csv'
    with csv_path.open('w') as f:
        f.write('step,dataset,modality,op_loss_w,recall,rare_recall,iou,distinct_ops\n')
        for ds, rows in data.items():
            modality = 'text' if 'text2cad' in ds else 'img'
            for r in rows:
                f.write(f'{r["step"]},{ds},{modality},'
                         f'{r["op_loss_w"]},{r["recall"]},{r["rare_recall"]},'
                         f'{r["iou"] if r["iou"] is not None else ""},'
                         f'{r["distinct_ops"]}\n')
    print(f'  → {csv_path}')


if __name__ == '__main__':
    main()
