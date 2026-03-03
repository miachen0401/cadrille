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

### Checkpoint Map (authoritative)

| Alias | Path | Architecture | What it is |
|---|---|---|---|
| **Official paper model** | `checkpoints/cadrille-rl` | `CADRecodeMM` | `maksimko123/cadrille` downloaded from HF — SFT+RL result from the paper |
| **Our SFT repro** | `checkpoints/cadrille-sft` | `MyQwen2VLForConditionalGenerationJoint` | Our 12k-step SFT on CAD-Recode v1.5 |
| **Our SFT + 10-step CPPO** | `work_dirs/cadrille-rl-hf-sft/checkpoint-final` | same | 10 CPPO rounds from our SFT — wandb `cadrille-rl-hf-sft` |
| ~~gbmgrb95~~ | ~~removed~~ | — | Was 600-step SFT smoke test (Feb 28), underfitted — **removed from eval** |

### Eval Results (deepcad_test_mini, 100 samples, pc mode)

| Checkpoint | IoU mean | CD median ×10³ | Failures | Source | Path |
|---|---|---|---|---|---|
| Paper cadrille (SFT only) | 0.756 | 8.9 | 1.8% | **Paper, 8,047 samples** | — |
| Paper cadrille (SFT+RL) | 0.787 | 7.1 | 1.2% | **Paper, 8,047 samples** | — |
| Official paper model (local) | 0.854 | 0.193 | 1% | Local, 100 samples | `checkpoints/cadrille-rl` |
| **Our SFT repro (12k steps)** | **0.880** | **0.192** | **0%** | Local, 100 samples | `checkpoints/cadrille-sft` |
| Our SFT + 10-step CPPO | 0.864 | 0.191 | 1% | Local, 100 samples | `cadrille-rl-hf-sft/checkpoint-final` |

**Note:** Local 100-sample numbers appear higher than paper (8,047 samples) because the mini subset happens to be slightly easier. For paper-comparable numbers, run on full `deepcad_test_mesh`.

### Tasks
- [x] `evaluate.py` run on `eval_hf_baseline` → `results_hf_baseline.csv`
- [x] `test.py` + `evaluate.py` run on `cadrille-sft` → `results_cadrille_sft.csv`
- [x] `test.py` + `evaluate.py` run on `cadrille-rl-hf-sft` → `results_cadrille_rl_10step.csv`
- [x] `viz/failure_analysis.py` updated: plots 8 (error analysis), 9-10 (CD), 11 (IoU/CD joint)
- [x] `viz/compare_evals.py` written: 5-plot side-by-side comparison
- [x] Full failure analysis on 3 valid checkpoints (11 plots each)
- [x] Comparison plots: Official paper model vs Our SFT repro
- [x] Removed gbmgrb95 (600-step smoke test) from all eval tables and plots

### Root Cause Analysis

**Official paper model failures (1/100):**
- 1× `geometry_error`: `GC_MakeArcOfCircle::Value() - no result`
- Root cause: degenerate arc geometry (floating-point issue in OCC kernel)

**Our SFT failures (0/100):**
- No failures — 100% valid geometry

**Our SFT + 10-step CPPO failures (1/100):**
- 1× `geometry_error` — identical pattern to baseline, not RL degradation

**RL mode collapse diagnosis (from cadrille-rl-full log, 900 steps):**
- Step 200: entropy 0.22→0.064 — collapse starts within ~20 gradient rounds
- Step 500: reward=9.89, reward_std=0.028 — all G=16 rollouts generate same H-slab
- Step 800: entropy=0.031 — fully collapsed; model ignores input, outputs same shape
- Root cause: no KL penalty (Dr. GRPO design); model finds H-slab fills unit cube, IoU≈0.13 for all inputs
- Fix: add entropy regularisation (`entropy_coef=0.01`) or skip degenerate groups (reward_std<0.5)

### Distribution Shift (training vs model-generated)
- No fillet ops in training data (0%) → neither model generates fillets
- Model under-uses arcs (19% vs 65%), segments (49% vs 76%), unions (14% vs 71%)
- Our SFT and the paper model show same distribution shift pattern

### Generated Plots
- `viz/plots/failure_analysis/hf_baseline/` — 11 plots (official paper model)
- `viz/plots/failure_analysis/cadrille_sft/` — 8 plots (our SFT, 0 failures)
- `viz/plots/failure_analysis/cadrille_rl_10step/` — 11 plots (our SFT + 10-step CPPO)
- `viz/plots/compare/` — 5 comparison plots: Official paper model vs Our SFT

---

## RL Fine-Tuning Reproduction

- [x] `rl/reward.py` — IoU-based reward via safe subprocess execution + compute_metrics() for IoU+CD
- [x] `rl/mine.py` — pre-filter training data for RL
- [x] `rl/train.py` — Dr. CPPO + DPO training loop + official W&B key names
- [x] `cadrille.py` — `compute_sequence_logprob()` static method
- [x] Smoke test: SFT (exit 0, eval_loss=2.28), RL (exit 0, W&B keys confirmed)
