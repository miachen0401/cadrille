# Monitor & Analysis Scripts — Reference

Single source of truth for all monitor/analysis scripts. **Active = used in current
training pipeline. Archive = one-off / superseded.**

## Active (in use right now)

| Script | Purpose | IID/OOD aware? |
|---|---|---|
| **eval_report.py** | Per-step v3 vs v4 metric report (greedy IoU + max@8 + ops). Posts unified Discord table. Triggered manually via `--step N --post`. | YES — splits BC val IID/OOD by family lookup, marks OOD with ★ in output. |
| **eval_to_discord.py** | Trajectory collage poster (the watcher script). Renders mesh trajectories per bucket. | YES — `pick_anchors()` forces half-IID + half-OOD for BenchCAD val bucket; per-anchor `[IID]`/`[OOD]` tag drawn in collage. |
| **render_ood_trajectory.py** | OOD-only render grid (n=9 OOD × n_steps), bypasses 8-anchor cap. | YES — explicitly OOD-only. |
| **op_trajectory_metrics.py** | Op-metric trajectory across runs (rare_recall, op_entropy, distinct_ops). | NO (per-bucket, mixes IID+OOD). |
| **retrospective_family_choice.py** | v3 family-choice robustness analysis. | YES (checks per-family ess_pass). |

## Plotting scripts (paper figures)

| Script | Output | Lines |
|---|---|---|
| **main_figure.py** | docs/paper_figures/fig_main_ops_recall.png (5-panel) | v3 vs v4 |
| **main_figure_v2.py** | docs/paper_figures/fig_main_recall_vs_composition.png (4-panel) | v3 vs v4 |
| **plot_4line_ess.py** | paper/figures/fig_7_4line_ess_pass.png | 4-line per recipe (IID + OOD + OOD+easy + no-bench) |
| **plot_ood_iou_4line.py** | paper/figures/fig_7_ood_iou_4line.png | 4-line OOD IoU |
| **plot_5_ops_candidates.py** | paper/figures/fig_7_ops_metric_candidates.png | 5 ops-metric options |
| **v4_failure_analysis.py** | docs/v4_failure_*.png suite | per-family OOD breakdown |

## Watcher (Discord automation)

| Script | What it does |
|---|---|
| `scripts/watch_eval_post_discord.sh` | Polls predictions/, fires `eval_to_discord.py` per new step, marks `.posted` |
| `scripts/launch_chain_runs.sh` | Sequential SFT runs (v4-hq-only → v4-holdout-noeasy → v4-baseline) |

## Archive (superseded / one-off)

| Script | Status |
|---|---|
| analyze_dim_errors.py / analyze_errors.py | Old failure-analysis tooling |
| analyze_sft_rl_delta.py | One-off SFT vs RL comparison |
| apply_data_filter.py | One-off corpus filter |
| bench_compare_vis.py | Old benchmark viz |
| comprehensive_data_audit.py / dataset_complexity_audit.py | One-off corpus audits |
| dataset_op_dist.py / dataset_samples_to_discord.py / dataset_stats.py | Pre-paper EDA |
| diag_resume_entropy.py | Resume-entropy diagnostic |
| diversity_analysis.py / diversity_benchcad_compare.py | Pre-paper EDA |
| failure_analysis.py | Old (different from v4_failure_analysis.py) |
| family_grid_audit.py | One-off |
| fillet_analysis.py | One-off |
| max_iou_trajectory_collage.py | Superseded by eval_to_discord.py |
| mining_analysis.py | RL mining analysis |
| op_distribution_plot.py / op_freq_pred_vs_gt.py | EDA |
| parse_cq.py | Helper library |
| plot_curriculum_metrics.py | Curriculum-only |
| plot_experiments_log.py / plot_kl_quadrants.py | RL diagnostics |

## Conventions

- **All per-case figures MUST tag IID/OOD** when the bucket is BenchCAD val.
- The `_HOLDOUT_FAMILIES` constant is centralized in `eval_to_discord.py` and
  `train/sft/online_eval.py`. Keep them in sync.
- Single source: `configs/eval/canonical_ops.yaml` for essential_pass spec.
- Single source: `configs/eval/op_taxonomy.yaml` for op patterns + rare set.
