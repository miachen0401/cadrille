# Repo Simplification Plan

Status as of 2026-04-24. Living doc — update as steps complete.

## Goal

Reorganise the repo into 3 clear buckets — **train / eval / tools** — by:
1. Extracting truly-shared utilities (`render_img`, `compute_metrics`) into a `common/` layer so eval doesn't import from `rl/`.
2. Folding SFT and RL training under a single `train/` package.
3. Moving paper-original `evaluate.py` / `test.py` into `eval/others/`; merging duplicate passk implementations.
4. Splitting `tools/` 33 scripts into 4 honest groups (analysis, data prep, benchmark, repair-LoRA experiment) instead of one big bag.

## Current state diagnosis (the WHY)

Findings from grep + dependency map (run 2026-04-24):

- **`rl.dataset.render_img` is shared infra, not RL-only** — imported by 9 files (rl/, eval/, tools/). Same for `rl.reward.compute_metrics` (8 importers). Naive "move `rl/` → `train/rl/`" would force eval to import from train/, wrong dependency direction.
- **passk has two implementations**: `eval/passk.py` claims to be a "thin wrapper" around `rl/eval_passk.py` but is actually a re-implementation. Both exist. Need to merge.
- **`tools/` has 5 different categories mashed together**: model behaviour analysis, data prep, throughput benchmarks, an entire repair-LoRA mini-experiment (4 files with their own shared helper), and 6 ad-hoc legacy eval scripts.
- **`viz/` is duplicated tooling** — viz/parse_cq.py is internally shared by 3 viz/ siblings, but viz/ vs tools/analyze_* serve the same purpose.
- **Top-level cruft**: `tmp_comunication.md`, `new_eval.md`, plus `viz/plots/*.png` (generated artefacts tracked in git).
- **`tools/_download_ckpts.py` is the only orphan** — no caller anywhere.

## Target structure

```
cadrille.py                      # model
dataset_sft.py                   # original dataset.py (SFT-only, renamed)

common/                          # ★ NEW — shared layer (eval+train+tools all import here)
  meshio.py                          # render_img, MeshDataset (was rl/dataset.py top half)
  metrics.py                         # compute_metrics, compute_reward, worker pool (was rl/reward.py)

train/                           # ── 1. TRAIN ──
  sft.py                             # was train.py
  rl/
    train.py
    algorithms/{cppo,dpo}.py
    dataset.py                       # only RLDataset/DPODataset/CurriculumRLDataset (no render_img)
    reward.py                        # only RL-specific shaping (CD reward etc.)
    eval_inloop.py                   # was rl/eval.py (training-time validation)
    config.py mine.py filter_scores.py
  configs/sft/*.yaml  configs/rl/*.yaml

eval/                            # ── 2. EVAL ──
  config.py pipeline.py runner.py render.py report.py
  passk.py                           # absorbs rl/eval_passk.py
  bench.py                           # was tools/eval_bench.py
  bench_visualize.py                 # was tools/bench_visualize.py
  configs/*.yaml
  others/                            # paper-original, untouched
    evaluate.py  test.py

tools/                           # ── 3. TOOLS (analysis only) ──
  analyze_errors.py  analyze_dim_errors.py  analyze_sft_rl_delta.py
  mining_analysis.py  plot_kl_quadrants.py  diag_resume_entropy.py
  render_comparison_grid.py  render_singleview_grid.py  bench_compare_vis.py
  # merged from viz/:
  parse_cq.py  compare_evals.py  failure_analysis.py
  fillet_analysis.py  training_dynamics.py  dataset_stats.py

data_prep/                       # ★ NEW — one-time data preparation
  prerender_dataset.py  deepcad2mesh.py  fusion360_train_mesh.py
  create_smoke_dataset.py  rewrite_recode_to_bench.py
  verify_recode_rewrite.py  push_bench_to_hf.py  upload_mined.py
  gen_repair_data.py

bench/                           # ★ NEW — training throughput benchmarks (not model eval)
  bench_config.py  bench_workers.py

experiments/
  cadevolve/                         # existing
  repair_lora/                       # ★ NEW — moved from tools/
    repair_feasibility.py  train_repair_lora.py
    eval_repair_lora.py  ablation_visual_conditioning.py
  legacy_eval/                       # ★ NEW — 6 ad-hoc scripts pending decision
    eval_img.py  eval_new_ckpts.py  eval_temperature.py
    smoke_eval.py  infer_cases.py  overfit_single.py

scripts/  tests/  docs/  configs (kept where they are)
```

## Test discipline

**One file: `tests/test_refactor_safety.py`** — see `.claude/projects/-workspace/memory/test_strategy.md` for why we cut the test plan to this.

Contents:
- imports smoke (every package imports without error, ~2 s)
- passk math (`_pass_at_k(n,c,k)` boundary cases — locks semantics before merging the two passk implementations)

Existing `test_iou.py` / `test_pipeline.py` / `test_cppo_step.py` cover deeper paths; they need GPU+ckpt+cadquery so run them after the refactor lands, not on every step.

## Execution order

| # | Step | Surface | Status |
|---|---|---|---|
| 0 | Write `tests/test_refactor_safety.py` + install pytest in uv env | new file | DONE |
| 1 | Extract `common/`: move `render_img`, `MeshDataset`, `compute_metrics`, `compute_reward` to `common/`. Leave re-export shims in `rl/dataset.py` and `rl/reward.py` so the 17 callers keep working unchanged | new dir + 2 shim files | DONE |
| 2 | Move `evaluate.py` + `test.py` → `eval/others/`. Update `scripts/run_eval.sh` paths | 2 file moves + 1 script | DONE |
| 3 | Merge `rl/eval_passk.py` into `eval/passk.py`. Update `scripts/run_passk.sh` to call `python -m eval.passk` | 1 deletion + 1 file + 1 script | DONE |
| 4 | Fold `train/`: rename `train.py` → `train/sft.py`, move `rl/` → `train/rl/`. Rewrite all `from rl...` imports across rl-internal files (~15 lines). Update `scripts/run_sft.sh` and `scripts/run_rl.sh` paths | ~15 internal imports + 2 scripts | DONE |
| 5 | tools/ split: merge `viz/` into `tools/`, split out `data_prep/`, `bench/`, `experiments/repair_lora/`, `experiments/legacy_eval/`. Rewrite `tools/README.md` | ~28 file moves | DONE |
| 6 | Cleanup: delete `tmp_comunication.md`, `viz/plots/*.png`, `new_eval.md` (or merge into docs). Drop `common/` re-export shims once nothing imports the old paths | deletes | DONE |
| 7 | Doc sync: update `CLAUDE.md` Canonical layout, `tools/README.md`, `plan.md` paths | 3 .md files | DONE |

After every step: `pytest tests/test_refactor_safety.py` must be green before moving on.

## Notes

- All steps live on the `revision` branch (see `.claude/projects/-workspace/memory/branch_layout.md`).
- `master` stays at `34415aa` = `origin/master` until refactor merges back via PR.
- Container files at repo root (`Dockerfile*`, `claude-container.sh`, `entrypoint.sh`) stay untouched — out of scope.
