"""Training dynamics visualizer for CPPO/GRPO fine-tuning.

Reads log.txt produced by rl/train.py and generates a 4-panel training health
dashboard, similar in spirit to SAPO Fig. 4:

  Panel 1 (top-left):  Reward trajectory — mean + min/max band, reward_std,
                        collapse zones (reward_std < 0.5 → grey background)
  Panel 2 (top-right): OOB IS ratio decomposed by bound (lower/upper stacked bar)
                        + total clip_fraction line  [mirrors Fig. 4a/b]
  Panel 3 (bot-left):  Entropy + KL divergence on dual Y-axis
  Panel 4 (bot-right): Advantage decomposition — pos_frac area + adv_abs_mean line
                        + eval IoU overlay (if eval lines present in log)

Usage
-----
python viz/training_dynamics.py --log work_dirs/cadrille-rl-full/log.txt
python viz/training_dynamics.py --log work_dirs/cadrille-rl-full/log.txt \\
    --out viz/plots/training_dynamics/cadrille_rl_full.png

Format of log.txt (two line types)
-----------------------------------
Train line (every log_steps):
  step=N loss=F average_reward=F train/reward_std=F ... train/lr=F

Eval line (every eval_steps):
  step=N eval/pc/DeepCAD test/IoU mean=F eval/pc/DeepCAD test/CD mean=F ...
"""

import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def _parse_kv(line: str) -> dict:
    """Parse 'key=value key=value ...' with support for keys containing spaces."""
    result = {}
    # Split on whitespace before known numeric patterns: key=number or key=nan/inf
    # Use regex: match either 'word/word/word=value' or 'word=value'
    pattern = re.compile(r'([\w/. ]+?)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)')
    for m in pattern.finditer(line):
        key = m.group(1).strip()
        val = m.group(2).strip()
        try:
            result[key] = float(val)
        except ValueError:
            pass
    return result


def load_log(log_path: str):
    """Parse log.txt into train_rows and eval_rows (list of dicts)."""
    train_rows = []
    eval_rows  = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            kv = _parse_kv(line)
            if 'step' not in kv:
                continue
            if 'eval/pc/DeepCAD test/IoU mean' in kv or 'eval/pc' in line:
                eval_rows.append(kv)
            else:
                train_rows.append(kv)
    return train_rows, eval_rows


def _col(rows, key, default=np.nan, aliases=()):
    """Extract a column, trying key then any aliases."""
    all_keys = (key,) + tuple(aliases)
    result = []
    for r in rows:
        for k in all_keys:
            if k in r:
                result.append(r[k])
                break
        else:
            result.append(default)
    return np.array(result, dtype=float)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

COLLAPSE_THRESHOLD = 0.5   # reward_std below this → mode collapse zone
ENTROPY_THRESHOLD  = 0.5   # entropy below this → low-entropy zone
EPS_LOW   = 0.1            # CPPO lower clip bound  (ratio < 1 - EPS_LOW)
EPS_HIGH  = 0.1            # CPPO upper clip bound  (ratio > 1 + EPS_HIGH)


def _shade_collapse(ax, steps, reward_std, alpha=0.12):
    """Shade background grey where reward_std < threshold (mode collapse)."""
    in_collapse = False
    start = None
    for i, (s, std) in enumerate(zip(steps, reward_std)):
        if std < COLLAPSE_THRESHOLD and not in_collapse:
            in_collapse = True
            start = s
        elif std >= COLLAPSE_THRESHOLD and in_collapse:
            ax.axvspan(start, s, color='grey', alpha=alpha, lw=0)
            in_collapse = False
    if in_collapse:
        ax.axvspan(start, steps[-1], color='grey', alpha=alpha, lw=0)


def plot_reward(ax, steps, rows):
    mean_r  = _col(rows, 'average_reward', aliases=('train/mean_reward',))
    std_r   = _col(rows, 'train/reward_std')
    r_max   = _col(rows, 'train/reward_max')
    r_min   = _col(rows, 'train/reward_min')

    _shade_collapse(ax, steps, std_r)

    # min/max band (if available)
    if not np.all(np.isnan(r_max)):
        ax.fill_between(steps, r_min, r_max, alpha=0.15, color='steelblue', label='reward [min,max]')

    ax.plot(steps, mean_r, color='steelblue', lw=1.8, label='average_reward')

    ax2 = ax.twinx()
    ax2.plot(steps, std_r, color='coral', lw=1.2, ls='--', label='reward_std')
    ax2.axhline(COLLAPSE_THRESHOLD, color='coral', lw=0.8, ls=':', alpha=0.7)
    ax2.set_ylabel('reward_std', color='coral', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='coral', labelsize=7)

    ax.set_ylabel('Reward', fontsize=9)
    ax.set_title('(a) Reward Trajectory', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax2.legend(fontsize=7, loc='upper right')

    patch = mpatches.Patch(color='grey', alpha=0.3, label=f'collapse (std<{COLLAPSE_THRESHOLD})')
    ax.legend(handles=ax.get_legend().legend_handles + [patch], fontsize=7, loc='upper left')


def plot_adv_is(ax, steps, rows):
    """Panel (b): Advantage fraction (left) + IS ratio mean ± std (right).

    Left y-axis stackplot: fraction of top-N sequences with positive vs negative
    advantage (2-way split).  Collapse zones (reward_std≈0) are forward-filled
    and shaded grey.

    Right y-axis: IS ratio mean ± std band with clip-bound lines at
    [1-EPS_LOW, 1+EPS_HIGH].  Falls back to total OOB rate when ratio_mean
    is not present in the log.
    """
    std_r    = _col(rows, 'train/reward_std')
    pos_frac = _col(rows, 'train/adv_pos_frac')

    # Mask degenerate collapse steps and forward-fill for stackplot
    pos_frac  = np.where(std_r < 1e-3, np.nan, pos_frac)
    pos_filled = _ffill(pos_frac, fill_val=0.5)
    neg_filled = 1.0 - pos_filled

    _shade_collapse(ax, steps, std_r)
    ax.stackplot(steps,
                 [neg_filled, pos_filled],
                 labels=['Adv < 0', 'Adv > 0'],
                 colors=['#FF7F0E', '#1F77B4'],
                 alpha=0.55)
    ax.set_ylabel('Advantage Fraction', fontsize=9)
    ax.set_ylim(0, 1)

    # Right axis: IS ratio mean ± std
    ratio_mean = _col(rows, 'train/ratio_mean')
    ratio_std  = np.nan_to_num(_col(rows, 'train/ratio_std'), nan=0.0)
    has_ratio  = not np.all(np.isnan(ratio_mean))

    ax2 = ax.twinx()
    if has_ratio:
        rm = np.nan_to_num(ratio_mean, nan=1.0)
        ax2.plot(steps, rm, color='#D4A017', lw=1.5, label='IS ratio mean')
        ax2.fill_between(steps, rm - ratio_std, rm + ratio_std,
                         color='#D4A017', alpha=0.18, label='±ratio_std')
        ax2.axhline(1 - EPS_LOW,  color='#D4A017', lw=0.9, ls='--', alpha=0.7,
                    label=f'clip [{1 - EPS_LOW:.1f}, {1 + EPS_HIGH:.1f}]')
        ax2.axhline(1 + EPS_HIGH, color='#D4A017', lw=0.9, ls='--', alpha=0.7)
        ax2.axhline(1.0,          color='#D4A017', lw=0.6, ls=':', alpha=0.4)
        ax2.set_ylabel('IS Ratio (π_new / π_old)', color='#D4A017', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='#D4A017', labelsize=7)
        ax2.legend(fontsize=7, loc='upper right')
    else:
        clip_tot = np.nan_to_num(_col(rows, 'train/clip_fraction'))
        ax2.plot(steps, clip_tot * 100, color='#D4A017', lw=1.5, label='OOB rate (%)')
        ax2.set_ylabel('OOB Rate (%)', color='#D4A017', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='#D4A017', labelsize=7)
        ax2.legend(fontsize=7, loc='upper right')

    ax.set_title('(b) Advantage Fraction + IS Ratio', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')


def plot_entropy_kl(ax, steps, rows):
    entropy = _col(rows, 'train/entropy')
    kl      = _col(rows, 'train/kl_approx')

    ax.plot(steps, entropy, color='#2CA02C', lw=1.8, label='entropy')
    ax.axhline(ENTROPY_THRESHOLD, color='#2CA02C', lw=0.8, ls=':', alpha=0.6,
               label=f'τ={ENTROPY_THRESHOLD}')
    ax.set_ylabel('Per-token Entropy', color='#2CA02C', fontsize=9)
    ax.tick_params(axis='y', labelcolor='#2CA02C', labelsize=7)
    ax.set_title('(c) Entropy + KL Divergence', fontsize=9, fontweight='bold')

    if not np.all(np.isnan(kl)):
        ax2 = ax.twinx()
        ax2.plot(steps, kl, color='#9467BD', lw=1.2, ls='--', label='KL approx')
        ax2.set_ylabel('KL approx  E[r−1−log r]', color='#9467BD', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='#9467BD', labelsize=7)
        ax2.legend(fontsize=7, loc='upper right')

    ax.legend(fontsize=7, loc='upper left')


def _ffill(arr: np.ndarray, fill_val: float = 0.5) -> np.ndarray:
    """Forward-fill NaN values; use fill_val if no prior valid entry."""
    out = arr.copy()
    last = fill_val
    for i, v in enumerate(arr):
        if not np.isnan(v):
            last = v
        out[i] = last
    return out


def plot_4quadrant(ax, steps, rows, eval_rows):
    """Panel (d): 4-quadrant advantage × IS-ratio stackplot + eval IoU.

    The 4 quadrants partition every top-N sequence by:
      - advantage sign  (adv > 0  vs  adv ≤ 0)
      - IS ratio vs 1   (ratio > 1  vs  ratio ≤ 1)

    Healthy training concentrates in pp (green) and nn (blue).
    Mismatch quadrants np (red) and pn (orange) indicate the policy moved
    against the gradient signal.

    Falls back to 2-way advantage fraction when q_* metrics are absent (old logs).
    """
    std_r = _col(rows, 'train/reward_std')
    q_pp  = _col(rows, 'train/q_pp')   # adv>0, IS>1  (good)
    q_pn  = _col(rows, 'train/q_pn')   # adv>0, IS≤1  (mismatch)
    q_np  = _col(rows, 'train/q_np')   # adv≤0, IS>1  (mismatch)
    q_nn  = _col(rows, 'train/q_nn')   # adv≤0, IS≤1  (good)

    has_4q = not np.all(np.isnan(q_pp))

    _shade_collapse(ax, steps, std_r)

    if has_4q:
        # Mask degenerate steps and forward-fill
        def _prep(arr):
            return _ffill(np.where(std_r < 1e-3, np.nan, arr), fill_val=0.25)
        pp = _prep(q_pp);  pn = _prep(q_pn)
        np_ = _prep(q_np); nn = _prep(q_nn)

        # Normalise rows to sum to 1 (guard floating-point drift)
        total = pp + pn + np_ + nn
        total = np.where(total > 0, total, 1.0)
        pp /= total;  pn /= total;  np_ /= total;  nn /= total

        # Stacking order (bottom→top): nn, np, pn, pp
        ax.stackplot(steps,
                     [nn, np_, pn, pp],
                     labels=['Adv≤0 IS≤1 ✓', 'Adv≤0 IS>1 ✗', 'Adv>0 IS≤1 ✗', 'Adv>0 IS>1 ✓'],
                     colors=['#1F77B4', '#D62728', '#FF7F0E', '#2CA02C'],
                     alpha=0.65)
        ax.axhline(0.5, color='grey', lw=0.6, ls=':', alpha=0.5)
    else:
        # Fallback: 2-way advantage fraction (old log without q_* keys)
        pos_frac  = _col(rows, 'train/adv_pos_frac')
        pos_frac  = np.where(std_r < 1e-3, np.nan, pos_frac)
        pos_filled = _ffill(pos_frac, fill_val=0.5)
        ax.stackplot(steps,
                     [1.0 - pos_filled, pos_filled],
                     labels=['Adv < 0', 'Adv > 0'],
                     colors=['#FF7F0E', '#1F77B4'],
                     alpha=0.55)
        ax.text(0.5, 0.5, 'train/q_* not in log\n(run new training for 4-quadrant)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=7, color='grey', style='italic')

    ax.set_ylabel('Fraction of top-N sequences', fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('(d) Adv × IS-Ratio Quadrants + Eval IoU', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

    # Eval IoU overlay
    if eval_rows:
        e_steps = np.array([r['step'] for r in eval_rows])
        e_iou   = _col(eval_rows, 'eval/pc/DeepCAD test/IoU mean')
        ax2 = ax.twinx()
        ax2.plot(e_steps, e_iou, 'D-', color='black', ms=5, lw=1.2, label='eval IoU')
        ax2.set_ylabel('Eval IoU', fontsize=8)
        ax2.tick_params(axis='y', labelsize=7)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=7, loc='lower right')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='CPPO training dynamics visualizer — 4-panel Fig.4-style dashboard')
    parser.add_argument('--log', required=True,
                        help='Path to log.txt from rl/train.py')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: alongside log.txt)')
    parser.add_argument('--title', default=None,
                        help='Figure title (default: derived from log filename)')
    args = parser.parse_args()

    train_rows, eval_rows = load_log(args.log)
    if not train_rows:
        print(f'No train rows found in {args.log}')
        return

    steps = np.array([r['step'] for r in train_rows])

    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Training step', fontsize=8)
        ax.tick_params(labelsize=7)

    plot_reward(ax1, steps, train_rows)
    plot_adv_is(ax2, steps, train_rows)
    plot_entropy_kl(ax3, steps, train_rows)
    plot_4quadrant(ax4, steps, train_rows, eval_rows)

    title = args.title or os.path.splitext(os.path.basename(os.path.dirname(args.log)))[0]
    fig.suptitle(f'CPPO Training Dynamics — {title}', fontsize=11, fontweight='bold')

    out_path = args.out
    if out_path is None:
        out_dir = os.path.join(os.path.dirname(args.log), 'plots')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'training_dynamics.png')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')

    # Print summary stats
    reward_std = _col(train_rows, 'train/reward_std')
    n_collapse = int(np.sum(reward_std < COLLAPSE_THRESHOLD))
    print(f'\nSummary ({len(train_rows)} train steps, {len(eval_rows)} eval points)')
    mean_r = _col(train_rows, 'average_reward', aliases=('train/mean_reward',))
    print(f'  Reward mean: {np.nanmean(mean_r):.3f}')
    print(f'  Entropy mean: {np.nanmean(_col(train_rows, "train/entropy")):.3f}')
    print(f'  Collapse steps (reward_std<{COLLAPSE_THRESHOLD}): {n_collapse}/{len(train_rows)}')
    if eval_rows:
        iou = _col(eval_rows, 'eval/pc/DeepCAD test/IoU mean')
        print(f'  Eval IoU: {np.nanmin(iou):.3f} → {np.nanmax(iou):.3f}')
    gen_secs = _col(train_rows, 'train/gen_seconds')
    if not np.all(np.isnan(gen_secs)):
        mean_gen = np.nanmean(gen_secs)
        total_gen_h = mean_gen * len(train_rows) / 3600
        print(f'  Gen time: {mean_gen:.1f}s/step avg  '
              f'({total_gen_h:.1f}h total generation over {len(train_rows)} steps)')


if __name__ == '__main__':
    main()
