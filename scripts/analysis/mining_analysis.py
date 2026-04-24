"""mining_analysis.py — Visualise hard-example mining results.

Reads the per-example scores produced by rl/mine.py and outputs:
  1. IoU distribution histogram (DeepCAD + Fusion360, overlaid)
  2. Hard-example count vs R_th threshold curve
  3. Markdown summary table printed to stdout

Usage
-----
    python3 tools/mining_analysis.py
    python3 tools/mining_analysis.py --deepcad data/mined/deepcad_hard_scores.jsonl \
                                      --fusion360 data/mined/fusion360_hard_scores.jsonl \
                                      --out-dir work_dirs/mining_analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_scores(path: str) -> list[float]:
    scores = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            scores.append(float(r["mean_reward"]))
    return scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deepcad",   default="./data/mined/deepcad_hard_scores.jsonl")
    p.add_argument("--fusion360", default="./data/mined/fusion360_hard_scores.jsonl")
    p.add_argument("--out-dir",   default="./work_dirs/mining_analysis")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dc  = load_scores(args.deepcad)
    f3  = load_scores(args.fusion360)
    print(f"DeepCAD  : {len(dc):,} examples loaded")
    print(f"Fusion360: {len(f3):,} examples loaded")

    thresholds = np.arange(0.0, 1.01, 0.05)

    # ── Figure 1: IoU histogram ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(dc, bins=bins, alpha=0.6, color="steelblue", label=f"DeepCAD  (n={len(dc):,})")
    ax.hist(f3, bins=bins, alpha=0.6, color="tomato",    label=f"Fusion360 (n={len(f3):,})")
    ax.axvline(0.75, color="black", linestyle="--", linewidth=1.5, label="R_th = 0.75 (paper)")
    ax.set_xlabel("Mean IoU (SFT greedy, K=1)")
    ax.set_ylabel("Example count")
    ax.set_title("IoU Distribution — Mining Scan Results")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p1 = out / "iou_distribution.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Saved → {p1}")

    # ── Figure 2: Hard-count vs threshold ────────────────────────────────────
    dc_arr = np.array(dc)
    f3_arr = np.array(f3)
    dc_hard = [int((dc_arr < t).sum()) for t in thresholds]
    f3_hard = [int((f3_arr < t).sum()) for t in thresholds]
    both    = [d + f for d, f in zip(dc_hard, f3_hard)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, dc_hard, "o-", color="steelblue", label="DeepCAD hard")
    ax.plot(thresholds, f3_hard, "s-", color="tomato",    label="Fusion360 hard")
    ax.plot(thresholds, both,    "^-", color="purple",    label="Combined hard")
    ax.axvline(0.75, color="black", linestyle="--", linewidth=1.5, label="R_th = 0.75 (paper)")
    ax.set_xlabel("R_th threshold")
    ax.set_ylabel("Hard example count")
    ax.set_title("Hard Example Count vs Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = out / "hard_count_vs_threshold.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Saved → {p2}")

    # ── Markdown table ───────────────────────────────────────────────────────
    print("\n## Mining Results — Hard Count vs Threshold\n")
    print(f"| R_th | DeepCAD hard | DC rate | Fusion360 hard | F3 rate | Combined |")
    print(f"|------|-------------|---------|----------------|---------|----------|")
    for t, d, f in zip(thresholds, dc_hard, f3_hard):
        dr = 100 * d / len(dc) if dc else 0
        fr = 100 * f / len(f3) if f3 else 0
        marker = " ← paper" if abs(t - 0.75) < 0.01 else ""
        print(f"| {t:.2f} | {d:,} | {dr:.1f}% | {f:,} | {fr:.1f}% | {d+f:,} |{marker}")


if __name__ == "__main__":
    main()
