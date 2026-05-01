# Monitor & Analysis Scripts — single source of truth

Cleaned 2026-05-01. **11 scripts, all active, no duplicates.**

## Per-step monitoring

| Script | Purpose | IID/OOD aware? |
|---|---|---|
| `eval_report.py` | Per-step v3 vs v4 metric report (greedy IoU + max@8 + ops). Posts unified Discord table. Run via `--step N --post`. | ✓ splits BC val IID/OOD by family, ★ marker on OOD row |
| `eval_to_discord.py` | Trajectory collage poster (the watcher script). Renders mesh trajectories per bucket. | ✓ `pick_anchors()` forces half-IID + half-OOD for BC val; per-anchor `[IID]`/`[OOD]` tag drawn on collage |
| `render_ood_trajectory.py` | OOD-only render grid (n=9 OOD × n_steps), bypasses 8-anchor cap | ✓ explicitly OOD-only |
| `parse_cq.py` | helper library — cadquery code parsing | n/a (utility) |

## Paper figure generation

| Script | Output | Lines / panels |
|---|---|---|
| `main_figure_v2.py` | `paper/figures/fig_main_recall_vs_composition.png` | 4-panel: (a) OOD rare_recall, (b) OOD ess_pass, (c) gap bar, (d) IID control |
| `v4_failure_analysis.py` | `docs/v4_failure_*.png` suite | per-family OOD breakdown |
| `plot_4line_ess.py` | `paper/figures/fig_7_4line_ess_pass.png` | §7 4-line: IID/OOD/OOD+easy/no-bench |
| `plot_ood_iou_4line.py` | `paper/figures/fig_7_ood_iou_4line.png` | §7 OOD IoU 4-line |
| `plot_5_ops_candidates.py` | `paper/figures/fig_7_ops_metric_candidates.png` | 5 ops-metric options for §7 fig 1 |
| `retrospective_family_choice.py` | stdout report (no figure) | family-choice robustness check |

## Wrappers

- `scripts/watch_eval_post_discord.sh` — polls predictions/, fires `eval_to_discord.py` per new step, marks `.posted`
- `scripts/launch_chain_runs.sh` — sequential SFT runs after current finishes

## Conventions

1. **All per-case figures with BC val MUST tag IID/OOD** (red `[OOD]` / blue `[IID]`).
2. `_HOLDOUT_FAMILIES` constant lives in:
   - `scripts/analysis/eval_to_discord.py`
   - `train/sft/online_eval.py` (set via `set_holdout_families()` from train.py cfg)
   Keep them in sync.
3. Single source for spec:
   - `configs/eval/canonical_ops.yaml` — essential_pass per family
   - `configs/eval/op_taxonomy.yaml` — op patterns + rare set + feature set
4. Deleted (35 scripts) — pre-paper EDA, RL-only diagnostics, one-offs, superseded duplicates.
