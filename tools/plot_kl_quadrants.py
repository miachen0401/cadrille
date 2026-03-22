"""Plot KL quadrant percentage distribution over training steps from wandb.

For each step, the four KL quadrants (pp, np, pn, nn) already sum to 1
(each is kl_per_quadrant / kl_total) and are plotted as a stacked area chart.

Quadrant meanings (adv sign × ratio sign):
  pp (+adv, ratio>1): model increased prob on good samples  → ideal
  np (-adv, ratio>1): model increased prob on bad samples   → harmful
  pn (+adv, ratio≤1): model decreased prob on good samples  → bad
  nn (-adv, ratio≤1): model decreased prob on bad samples   → benign

Stack order (bottom→top): pp (red), np (yellow), pn (blue), nn (gray)

Usage:
  uv run python tools/plot_kl_quadrants.py <run_id_or_name> [<run2> ...]
  uv run python tools/plot_kl_quadrants.py --list              # list recent runs
  uv run python tools/plot_kl_quadrants.py <run1> <run2> --out kl_quad.png

Requires WANDB_API_KEY to be set. Either export it or load from .env first:
  source .env && uv run python tools/plot_kl_quadrants.py ...
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


WANDB_PROJECT = "cadrille-rl"
METRICS = ["train/kl_q_pp", "train/kl_q_np", "train/kl_q_pn", "train/kl_q_nn"]
COLORS = {
    "pp": "#e74c3c",
    "np": "#f1c40f",
    "pn": "#3498db",
    "nn": "#95a5a6",
}
LABELS = {
    "pp": "pp (+adv, ratio>1)",
    "np": "np (−adv, ratio>1)",
    "pn": "pn (+adv, ratio≤1)",
    "nn": "nn (−adv, ratio≤1)",
}


def get_wandb_api():
    try:
        import wandb
        api = wandb.Api()
        return api
    except ImportError:
        print("ERROR: wandb not installed. Run: uv pip install wandb")
        sys.exit(1)


def resolve_run(api, run_id_or_name, project=WANDB_PROJECT, entity=None):
    if run_id_or_name.count("/") == 2:
        try:
            return api.run(run_id_or_name)
        except Exception:
            pass
    for ent in ([entity] if entity else []) + [api.default_entity, "hula-the-cat"]:
        try:
            r = api.run(f"{ent}/{project}/{run_id_or_name}")
            return r
        except Exception:
            pass
    for ent in ([entity] if entity else []) + [api.default_entity, "hula-the-cat"]:
        try:
            runs = api.runs(f"{ent}/{project}")
            for r in runs:
                if r.id == run_id_or_name or r.name == run_id_or_name:
                    return r
        except Exception:
            pass
    print(f"ERROR: Run '{run_id_or_name}' not found in project '{project}'")
    sys.exit(1)


def fetch_kl_data(run):
    history = run.history(keys=METRICS, pandas=True)
    if history.empty or "_step" not in history.columns:
        return None
    history = history.dropna(subset=[m for m in METRICS if m in history.columns])
    steps = history["_step"].values
    pp  = history.get("train/kl_q_pp",  np.zeros(len(steps))).values if hasattr(history.get("train/kl_q_pp", None), "values") else history["train/kl_q_pp"].values if "train/kl_q_pp" in history.columns else np.zeros(len(steps))
    np_ = history["train/kl_q_np"].values if "train/kl_q_np" in history.columns else np.zeros(len(steps))
    pn  = history["train/kl_q_pn"].values if "train/kl_q_pn" in history.columns else np.zeros(len(steps))
    nn  = history["train/kl_q_nn"].values if "train/kl_q_nn" in history.columns else np.zeros(len(steps))
    pp  = history["train/kl_q_pp"].values if "train/kl_q_pp" in history.columns else np.zeros(len(steps))
    return steps, pp, np_, pn, nn


def smooth(arr, window):
    if window <= 1 or len(arr) <= 1:
        return arr
    w = min(window, len(arr))
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')


def plot_run(ax, steps, pp, np_, pn, nn, label, smooth_window=1):
    pp  = np.exp(smooth(pp,  smooth_window))
    np_ = np.exp(smooth(np_, smooth_window))
    pn  = np.exp(smooth(pn,  smooth_window))
    nn  = np.exp(smooth(nn,  smooth_window))
    total = np.sum(np.array([pp, np_, pn, nn]), axis=0)
    total = np.where(total == 0, 1, total)
    pp, np_, pn, nn = pp/total*100, np_/total*100, pn/total*100, nn/total*100
    ax.stackplot(steps, pp, np_, pn, nn,
                 colors=[COLORS["pp"], COLORS["np"], COLORS["pn"], COLORS["nn"]],
                 alpha=0.85)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL share (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


def list_runs(api, project=WANDB_PROJECT):
    entity = api.default_entity
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    print(f"{'Run ID':<12}  {'Name':<45}  {'State':<10}  Steps")
    print("-" * 80)
    for r in runs[:30]:
        steps = r.summary.get("_step", "?")
        print(f"{r.id:<12}  {r.name:<45}  {r.state:<10}  {steps}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="Run IDs or names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--project", default=WANDB_PROJECT)
    args = parser.parse_args()

    api = get_wandb_api()

    if args.list:
        list_runs(api, project=args.project)
        return

    if not args.runs:
        parser.print_help()
        sys.exit(1)

    n = len(args.runs)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    fig.suptitle("KL Quadrant Distribution over Steps", fontsize=13, y=1.01)

    for i, run_id in enumerate(args.runs):
        run = resolve_run(api, run_id, project=args.project)
        print(f"Fetching {run.name} ({run.id}) [{run.state}]...")
        result = fetch_kl_data(run)
        if result is None:
            axes[0][i].text(0.5, 0.5, f"No data\n({run.state})",
                            ha='center', va='center', transform=axes[0][i].transAxes, fontsize=12)
            axes[0][i].set_title(run.name, fontsize=10)
            continue
        steps, pp, np_, pn, nn = result
        plot_run(axes[0][i], steps, pp, np_, pn, nn,
                 label=run.name, smooth_window=args.smooth)

    patches = [
        mpatches.Patch(color=COLORS["pp"], label=LABELS["pp"]),
        mpatches.Patch(color=COLORS["np"], label=LABELS["np"]),
        mpatches.Patch(color=COLORS["pn"], label=LABELS["pn"]),
        mpatches.Patch(color=COLORS["nn"], label=LABELS["nn"]),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
