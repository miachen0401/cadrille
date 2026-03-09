"""Re-filter mining scores by a different R_th without re-running inference.

Reads one or more *_scores.jsonl files produced by mine.py and applies a
new IoU threshold to rebuild the hard-examples pkl.

Usage
-----
    # Single dataset
    python3 rl/filter_scores.py data/mined/deepcad_scores.jsonl --R-th 0.6

    # Combined (deepcad + fusion360)
    python3 rl/filter_scores.py \
        data/mined/deepcad_scores.jsonl \
        data/mined/fusion360_scores.jsonl \
        --R-th 0.70 \
        --output data/mined/combined_hard_0.70.pkl

    # Show IoU distribution (no output written)
    python3 rl/filter_scores.py data/mined/deepcad_scores.jsonl --stats-only
"""

import argparse
import json
import pickle
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("scores", nargs="+", help="*_scores.jsonl file(s)")
    p.add_argument("--R-th", type=float, default=0.75,
                   help="Keep examples where mean_reward < R_th (default 0.75)")
    p.add_argument("--output", default=None,
                   help="Output pkl path. Default: same dir as first scores file, "
                        "named combined_hard_{R_th}.pkl")
    p.add_argument("--stats-only", action="store_true",
                   help="Print distribution table, do not write pkl")
    return p.parse_args()


def main():
    args = parse_args()

    all_records = []
    for path in args.scores:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    if not all_records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    rewards = [r["mean_reward"] for r in all_records]
    total = len(rewards)

    # Distribution table
    thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    print(f"\n{'R_th':>6}  {'hard_count':>10}  {'hard_%':>7}  {'selected' if args.R_th in thresholds else '':>8}")
    print("-" * 40)
    for th in thresholds:
        n = sum(1 for r in rewards if r < th)
        marker = " <-- selected" if abs(th - args.R_th) < 1e-9 else ""
        print(f"{th:>6.2f}  {n:>10}  {100*n/total:>6.1f}%{marker}")

    # Failures (reward = -1)
    n_fail = sum(1 for r in rewards if r < 0)
    print(f"\n  Total scanned : {total}")
    print(f"  Failures (-1) : {n_fail}  ({100*n_fail/total:.1f}%)")
    print(f"  Mean IoU      : {sum(max(r,0) for r in rewards)/total:.4f}  (failures counted as 0)")

    if args.stats_only:
        return

    # Filter
    hard = [{"gt_mesh_path": r["gt_mesh_path"], "file_name": r["file_name"]}
            for r in all_records if r["mean_reward"] < args.R_th]

    # Output path
    if args.output is None:
        base = Path(args.scores[0]).parent
        tag = str(args.R_th).replace(".", "p")
        out_path = base / f"combined_hard_{tag}.pkl"
    else:
        out_path = Path(args.output)

    with open(out_path, "wb") as f:
        pickle.dump(hard, f)

    print(f"\n  R_th={args.R_th}: {len(hard)} hard examples written → {out_path}")


if __name__ == "__main__":
    main()
