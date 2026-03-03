# Progress

## Status Legend
- [ ] Pending
- [~] In progress
- [x] Done
- [!] Blocked

---

## Repo Restructuring + Official W&B Logging (2026-03-03)

- [x] Step 0: Commit pre-refactor state (commit 28667f9)
- [x] Step 1: `rl/__init__.py` — empty package marker
- [x] Step 1: `rl/reward.py` — moved from reward.py + `compute_metrics(→ iou_reward, cd)` using scipy cKDTree (8192 points, matches evaluate.py)
- [x] Step 1: `rl/mine.py` — moved from mine_hard_examples.py, updated imports
- [x] Step 2: `rl/train.py` — moved from rl_train.py + official W&B key names:
  - `loss`, `average_reward` at top level (matches official dashboard)
  - `eval/pc/DeepCAD test/IoU mean|median|CD mean|median|Failures fraction`
  - `run_validation()` now calls `compute_metrics()` → IoU + CD per sample
  - log.txt updated to match official key format
- [x] Step 3: `configs/sft/{default,full,h100,smoke}.yaml`
- [x] Step 3: `configs/rl/default.yaml` (+ val_split_dir, val_samples=200, eval_steps=500)
- [x] Step 3: `configs/rl/4080.yaml` (NEW — G=4, Adam8bit, for 16 GB RTX 4080)
- [x] Step 4: `scripts/run_sft.sh` — config paths → `configs/sft/`
- [x] Step 4: `scripts/run_rl.sh` — `rl_train.py` → `rl/train.py`, config paths → `configs/rl/`
- [x] Step 4: `scripts/run_eval.sh` — NEW: one-shot eval pipeline (test.py → evaluate.py)
- [x] Step 5: `requirements.txt` — pinned deps
- [x] Step 6: `README.md` — rewrite with FAIR-style sections + RL training docs

### Old flat files (PID 10126 killed — safe to delete)
- [x] Delete `rl_train.py` → superseded by `rl/train.py`
- [x] Delete `reward.py` → superseded by `rl/reward.py`
- [x] Delete `mine_hard_examples.py` → superseded by `rl/mine.py`
- [x] Delete flat `configs/*.yaml` → superseded by `configs/sft/` and `configs/rl/`

---

## Stage 3: Visualization & Analysis (2026-03-03)

- [x] `viz/__init__.py` — empty package marker
- [x] `viz/parse_cq.py` — regex feature extractor (segments, arcs, extrudes, unions, etc.)
- [x] `viz/dataset_stats.py` — 6 training data distribution plots
- [x] `viz/failure_analysis.py` — failure mode analysis (subprocess isolation, 7 plot types)
- [x] `evaluate.py` — added `--results-csv` flag for per-sample IoU/CD CSV output

### Key Findings (2026-03-03)

**Training dataset (2,810 train + 150 val):**
- Median code length: 355 chars (train), 352 chars (val) — well-matched distributions
- Median sketch ops: 7 per script; p95 = 21
- Top operations: extrude (91.5%), segment (75.7%), union (70.6%), arc (65.4%)
- Rare ops in training: revolve, loft, sweep, spline, fillet, polygon — each < 5%
- 100% of boolean ops are union; no cuts in training data (cut appears only via `mode='s'`)
- XY plane dominates (78%); ZX and YZ are each ~15%

**Baseline failure analysis (hf_baseline, 100 samples):**
- Only 1% failure rate (1 geometry_error, 99 success)
- Model most likely to fail on arcs (5.3% failure rate vs 0% for other ops)
- No syntax errors or timeouts in baseline eval

**Distribution shift (train vs. model-generated):**
- Model under-uses complex ops: segments (49% vs 76%), unions (14% vs 71%), arcs (19% vs 65%)
- Suggests model simplifies shapes — RL reward signal should incentivize structural fidelity

### Generated Plots
- `viz/plots/dataset_stats/` — 7 plots (op frequency×2, code length, sketch ops, plane types, body/bool, co-occurrence)
- `viz/plots/failure_analysis/` — legacy 5-plot run (no IoU)

---

## Full Eval + Root Cause Analysis (2026-03-03)

- [x] `evaluate.py` run on `eval_hf_baseline` → `results_hf_baseline.csv`
- [x] `evaluate.py` run on `eval_gbmgrb95_mini` → `results_gbmgrb95_mini.csv`
- [x] `test.py` + `evaluate.py` run on `cadrille-sft` checkpoint → `results_cadrille_sft.csv`
- [x] `viz/failure_analysis.py` updated: plots 8 (error analysis), 9-10 (CD), 11 (IoU/CD joint)
- [x] `viz/compare_evals.py` written: 5-plot side-by-side comparison
- [x] Full failure analysis on all 3 checkpoints (11 plots each)
- [x] Comparison plots: HF Baseline vs Cadrille-SFT (ours)

### Eval Results (deepcad_test_mini, 100 samples, pc mode)

| Checkpoint | IoU mean | CD median ×10³ | Failure rate | Notes |
|---|---|---|---|---|
| HF Baseline (`maksimko123/cadrille`) | 0.854 | 0.193 | 1% (1 geometry_error) | Official paper model |
| **Cadrille-SFT (ours)** | **0.880** | **0.192** | **0%** | Our 12k-step SFT repro |
| RL smoke test (10-step CPPO) | 0.133 | 129.0 | 11% | Expected: just 10 RL steps |

### Root Cause Analysis

**HF Baseline failures (1/100):**
- 1× `geometry_error`: `GC_MakeArcOfCircle::Value() - no result`
- Root cause: degenerate arc geometry (floating-point issue in OCC kernel)

**Cadrille-SFT failures (0/100):**
- No failures — 100% valid geometry

**RL smoke-test failures (11/100):**
- All 11× `geometry_error`: `BRep_API: command not done`
- Root cause: early RL training destabilises the model; generates geometrically invalid Boolean operations

### Distribution Shift (training vs model-generated)
- No fillet ops in training data (0%) → model never generates fillets
- Model under-uses arcs (19% vs 65%), segments (49% vs 76%), unions (14% vs 71%)
- Our SFT matches HF baseline on op usage — similar distribution shift pattern
- RL smoke-test generates even simpler code (fewer ops overall)

### Generated Plots
- `viz/plots/failure_analysis/hf_baseline/` — 11 plots (full suite with IoU/CD)
- `viz/plots/failure_analysis/cadrille_sft/` — 8 plots (no failures, so no error-analysis plot)
- `viz/plots/failure_analysis/gbmgrb95_mini/` — 11 plots (RL smoke-test, 11% failures)
- `viz/plots/compare/` — 5 comparison plots: HF Baseline vs Cadrille-SFT (ours)

---

## RL Fine-Tuning Reproduction

- [x] `rl/reward.py` — IoU-based reward via safe subprocess execution + compute_metrics() for IoU+CD
- [x] `rl/mine.py` — pre-filter training data for RL
- [x] `rl/train.py` — Dr. CPPO + DPO training loop + official W&B key names
- [x] `cadrille.py` — `compute_sequence_logprob()` static method
- [x] Smoke test: SFT (exit 0, eval_loss=2.28), RL (exit 0, W&B keys confirmed)
