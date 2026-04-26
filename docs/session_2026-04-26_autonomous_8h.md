# 8-hour autonomous session — 2026-04-26

User granted 8h of GPU time + autonomy. Session goal: validate path 2 (no PC,
Qwen3-VL-2B, scale data + steps + drop text2cad) by:
1. Letting current curriculum run finish to step 20000 → final phase 3 verdict
2. Final eval sweep on best ckpt (T8)
3. Launching new "clean recipe" run on bigger corpus
4. Prepping data for tomorrow's full Option B (1M corpus)

## TL;DR

**What we proved**:
- _(filled in at end of session)_

**Numbers** (placeholder until evals land):

| ckpt | BenchCAD val | DeepCAD test | Fusion360 test | notes |
|---|---:|---:|---:|---|
| **Curriculum step 11000** (BenchCAD peak) | _T8 pending_ | _T8 pending_ | _T8 pending_ | online_eval said 0.597 / 0.444 / 0.548 |
| **Curriculum step 15000** (DeepCAD peak) | _T8 pending_ | _T8 pending_ | _T8 pending_ | online_eval said 0.530 / 0.480 / 0.540 |
| **Curriculum step 20000** (final, Phase 3 over-trained) | _T8 pending_ | _T8 pending_ | _T8 pending_ | expected ≤ step 17000 |
| **Option A step 14000** (final, budget-fit target) | _pending_ | _pending_ | _pending_ | apples-to-apples vs curriculum step-14k |

## Decisions made autonomously

1. **Phase 3 verdict: HURTS** (online_eval log evidence)
   - BenchCAD val peaked at step 11000 = 0.597 → degraded to 0.509 by step 17000 (-0.09)
   - DeepCAD test peaked at step 15000 = 0.480 → dropped to 0.414 by step 17000 (-0.07)
   - Even sampling pool (max@8) degraded: DeepCAD max@8 0.616 (step 16k) → 0.596 (step 17k)
   - **Conclusion**: 8:1:1 mix is too aggressive as a stable recipe. Transition itself produced a +0.107 BenchCAD bump (P2→P3 step 10k→11k) but stable maintenance overfits BenchCAD and forgets cad-recode.

2. **Next-run recipe: drop curriculum, drop text2cad, larger recode corpus**
   - text2cad saturated at step 1000 (recall=1.0, op_loss=0) — useless after warmup
   - Static mix only (no phase shifts) — phase 3 evidence shows shifts harm more than help
   - benchcad:recode = 1:9 (paper-style)
   - 100k cad-recode-v1.5 subset (5× current 18k) reduces overfitting risk
   - **Config**: `configs/sft/qwen3vl_2b_recode_30k_clean.yaml` (max_steps=14k after
     budget recompute: A100 实测 1.59s/step → 14k steps + 7 evals = 6.85h; leaves
     1.15h for T8 + setup)

3. **Backbone unchanged** (per user constraint): Qwen3-VL-2B-Instruct
4. **No PC modality** (per user constraint): img-only

## Work artifacts produced

| artifact | purpose | status |
|---|---|---|
| `data_prep/fetch_cadrecode_full.py` | download + render + pkl-build for filapro/cad-recode-v1.5 | done |
| `eval/bench_sweep.py` patch | added `--backbone {qwen2_vl,qwen2_5_vl,qwen3_vl}` arg | done |
| `train/sft/train.py` patch | schema-detect cad-recode-v1.5 pkl → routes to img-only loader if no STLs | done |
| `configs/sft/qwen3vl_2b_recode_30k_clean.yaml` | "clean recipe" Option A config | done |
| `scripts/launch_t8_then_option_a.sh` | orchestrator: wait curriculum → T8 → Option A | running |
| `data/cad-recode-v1.5/` symlink → `/ephemeral/data/cad-recode-v1.5/` | new corpus root | done |
| `/ephemeral/checkpoints/curriculum_best_from_hf/{checkpoint-11000,checkpoint-15000}` | pulled from HF for T8 eval | done |
| `progress.md` | per-task progress log for this session | done |

## Open questions for user (when back)

1. **Option A trajectory** — did the clean recipe (no curriculum, no text2cad,
   bigger corpus) outperform curriculum at the same step count? _(answer
   pending T8 + Option A evals)_
2. **Best practical ckpt** — should we serve step 11000 (BenchCAD peak) or
   step 15000 (DeepCAD peak) of curriculum, knowing phase 3 over-shoots?
3. **Option B trigger** — full 1M corpus + 100k steps run takes ~30h on A100.
   Worth scheduling overnight if Option A shows promise but plateaus?

## Open work for next session

1. **T11/T12** — BenchCAD rare-op oversample + per-source weighted loss
   (highest expected impact for fillet/chamfer/shell recall=0)
2. **Option B (full)** — render remaining ~880k cad-recode-v1.5 (12-15h CPU on
   16 cores) → train 100k steps on 1M corpus (~25h on A100)
3. **T15** — PC modality on Qwen3-VL deepstack (engineering risk; biggest
   paper-validated lever for the IoU ceiling)
