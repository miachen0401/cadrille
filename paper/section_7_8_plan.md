# §7+§8 Experiment Queue

Storyline lives in `paper/storyline.md`. This file is the **experiment plan + status**.

## Tier 0 — required for credible NeurIPS main-track (~3 GPU-day)

| # | Run | Config | ETA | Status |
|---|---|---|---|---|
| T0-1 | v4-holdout to 50k | `configs/sft/big_bench_shell_50k_v4_holdout.yaml` | +24h | 🟡 23k/50k |
| T0-2 | v4-baseline 50k | `configs/sft/big_bench_shell_50k_v4_baseline.yaml` | +24h after T0-1 | ❌ pending (chain launcher armed) |
| T0-3 | v4-holdout-noeasy 50k | `configs/sft/big_bench_shell_50k_v4_holdout_noeasy.yaml` | +24h after T0-2 | ❌ pending (chain launcher armed) |
| T0-4 | v4-hq-only 50k | `configs/sft/big_bench_shell_50k_v4_hq_only.yaml` | +24h after T0-3 | ❌ pending (chain launcher armed) |
| T0-5 | Stratified offline eval @ ckpt-25k+50k for all 4 runs | `scripts/analysis/eval_report.py` | +30min × 8 = 4h after each run | partial (live report on v4-holdout) |

## Tier 1 — strong robustness (~3 GPU-day)

| # | Run | Resolves |
|---|---|---|
| T1-1 | 3 random holdout configs (10 fams each, 25k step) | Cherry-pick attack: gap stable across holdout choices |
| T1-2 | Cross-LLM IoU-24 eval (gpt-4o / Claude / Gemini on 50 OOD) | Show frontier LLMs also fail OOD composition |
| T1-3 | Retrospective v3 family-subset analysis | ✅ DONE 2026-04-30 (v3 0.87-0.99 on random subsets, our 10 = 1.000) |

## Tier 2 — extension (~2 GPU-day)

| # | Run | Resolves |
|---|---|---|
| T2-1 | benchcad-easy reverse holdout | Internal self-consistency |
| T2-2 | DC/Fu op-signature pseudo-families | Cross-dataset validation |
| T2-3 | Multi-metric verification (exact_op_match + AST sim) | Metric robustness |

## Tier 3 — RL (post-SFT, ~1 GPU-day)

| # | Run | Reward |
|---|---|---|
| T3-1 | RL-iou from v4-holdout ckpt-50k | pure IoU |
| T3-2 | RL-ess from v4-holdout ckpt-50k | IoU + 0.3 × essential_pass |

## Decision points (open)

1. ⏳ Confirm 10-family holdout is final (canonical: `configs/sft/holdout_families.yaml`)
2. ⏳ Pick §7 main figure ops-metric: A=rare_recall / B=ess_pass / C=gap / D=op_entropy / E=feat_F1
3. ⏳ Whether to launch Tier 1 (random holdouts) before paper draft

## Reviewer self-critique highlights

- **A. Comparability**: 4 lines confound holdout × supplement. Need 6-line plot adding v4-baseline (saw all + 60/40 mix).
- **B. RL reward**: Need 3+ variants (pure IoU / additive / multiplicative).
- **C. Mix ratio**: 60/40 chosen but not ablated.
- **D. Single seed**: 2nd seed needed for headline.
- **E. n=9 OOD too small**: Phase B refactor done — now n=50 stratified per eval.
- **F. Family choice**: Defended via T1-3 retrospective.

## Past outputs (already on Discord)

- v3 vs v4-holdout 23-step trajectory (per-step Discord posts via `eval_report.py`)
- 7 paper figures (`paper/figures/`)
- §7 storyline doc (`paper/storyline.md`)
- Run config mapping (`configs/sft/README.md`)
