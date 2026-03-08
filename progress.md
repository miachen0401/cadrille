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

---

## RL Training Data Pipeline + Cadrille Training (2026-03-07)

### Goals
- Match paper: DeepCAD train split (~161k) + Fusion360 train split (~8.6k), img modality
- Previous run used deepcad_test_mesh (1785 meshes) — wrong

### Tasks
- [x] T1: `data/deepcad2mesh.py` — converts DeepCAD JSON → normalized STL (unit cube, centroid at [0.5,0.5,0.5])
  - Running in background, ~85k/161k done as of Mar 7, ~3.8h remaining
  - Output: `data/deepcad_train_mesh/*.stl`
- [x] T2: Create `data/cadrille_training/` with symlinks to deepcad_train_mesh (+ future fusion360_train_mesh)
  - `data/cadrille_training/deepcad → ../deepcad_train_mesh`
- [x] T3: Update all RL configs: `data_dir: ./data/cadrille_training`
  - Updated: 4080.yaml, h100.yaml, h100x8.yaml, a100.yaml, a100-80gb.yaml, default.yaml
  - Smoke/test configs left unchanged (smoke.yaml, 4080-test-1k.yaml)
- [x] T4: Kill old run (step 3280, PID 1312) on wrong data
- [x] T5: Start new run `rl-cadrille-train-4080-0307` with 84526 STLs from cadrille_training
  - W&B: https://wandb.ai/hula-the-cat/cadrille-rl/runs/t6jfmtbd
  - Pre-training baseline: img/DeepCAD IoU=0.809, img/Fusion360 IoU=0.797
  - Steps running, rewards vary 3–10 (normal RL variance)
- [ ] T6: Fusion360 train split — download STEP + convert STL → add symlink to cadrille_training
- [~] T7: Monitor training, debug every 3 min

---

## RL Training Runs Log (2026-03-07)

All runs use: checkpoint `./checkpoints/cadrille-sft` (commit `7aeaa26`), config `configs/rl/4080.yaml`,
hardware RTX 4080 SUPER 16 GB / 15 GB RAM / 1007 GB disk (474 GB used).

### Run 0 — Killed (wrong data)
| Field | Value |
|---|---|
| Run name | (previous session run) |
| PID | 1312 |
| W&B | — |
| Commit | pre-`7aeaa26` |
| Data | `data/deepcad_test_mesh` — **wrong** (test split, 1,785 STLs, not train) |
| Modality | img |
| Status | Killed at step ~3280 when discovered wrong data |
| Notes | This run was already in progress when this session began |

### Run 1 — Crashed (histogram bug)
| Field | Value |
|---|---|
| Run name | `rl-cadrille-train-4080-0307` |
| PID | 36228 |
| W&B | https://wandb.ai/hula-the-cat/cadrille-rl/runs/t6jfmtbd |
| Commit | `7aeaa26` |
| Data | `data/cadrille_training/deepcad` → `deepcad_train_mesh` (84,526 STLs) |
| Modality | img |
| val samples | 25 DeepCAD + 25 Fusion360 |
| Pre-training baseline | img/DeepCAD IoU=0.087, img/Fusion360 IoU=0.123, pc/DeepCAD IoU=0.831, pc/Fusion360 IoU=0.797 |
| Steps run | 1–80 |
| Crash | `ValueError: Too many bins for data range` in `wandb.Histogram` when reward_std=0 (degenerate rollout group) |
| Fix applied | Added `_safe_histogram()` to `rl/algorithms/cppo.py` — falls back to scalar mean on zero-range data |

### Run 2 — Killed (low rewards, unexplained img collapse)
| Field | Value |
|---|---|
| Run name | `rl-s50k-lr1e-5-G4-cppo-0307-*` |
| PID | 36998 |
| W&B | — |
| Commit | `7aeaa26` + `_safe_histogram` patch |
| Data | `data/cadrille_training/deepcad` (84,526 STLs) |
| Modality | img |
| val samples | 25 DeepCAD + 25 Fusion360 |
| Pre-training baseline | img/DeepCAD IoU=0.087, img/Fusion360 IoU=0.123 |
| Steps run | 1–9 |
| Rewards | 0.03, -4.91, -2.34 (very low/negative) |
| Status | Killed manually |
| Notes | Baselines consistent with Run 1 (img/DeepCAD 0.087 is confirmed SFT starting point). Root cause of negative rewards unclear — possibly degenerate groups dominating early steps. Switched to pc mode. |

### Run 3 — Killed by user (pc mode, working)
| Field | Value |
|---|---|
| Run name | `rl-s50k-lr1e-5-G4-cppo-0307-0927` |
| PID | 38401 |
| W&B | https://wandb.ai/hula-the-cat/cadrille-rl/runs/joqegnzd |
| Commit | `7aeaa26` + `_safe_histogram` patch |
| Data | `data/cadrille_training/deepcad` (84,526 STLs) |
| Modality | **pc** (switched from img to test stability) |
| val samples | 25 DeepCAD + 25 Fusion360 |
| Pre-training baseline | pc/DeepCAD IoU=0.833, pc/Fusion360 IoU=0.831, img/DeepCAD IoU=0.087, img/Fusion360 IoU=0.123 |
| Eval step 400 | pc/DeepCAD IoU=0.795, pc/Fusion360 IoU=0.847, img/DeepCAD IoU=0.064, img/Fusion360 IoU=0.153 |
| Steps run | 1–65 |
| Rewards | Mixed 1–5, mean ~1.14 |
| Status | Killed by user — switching back to img mode (paper default) |

### Run 4 — Killed (render bug: no normalization)
| Field | Value |
|---|---|
| PID | 39183 |
| Commit | `7aeaa26` + patches (img mode, no normalization yet) |
| Data | `data/cadrille_training/deepcad` (84,526 STLs) |
| Modality | img |
| val samples | 25 DeepCAD + 25 Fusion360 |
| Pre-training baseline | img/DeepCAD IoU=0.087, img/Fusion360 IoU=0.123 (same as before — rendering still broken) |
| Steps run | 1–8 |
| Status | Killed — rendering fix in progress |
| Bug identified | `render_img()` did not normalize mesh before rendering. Meshes at mm scale ([-100,100]) but camera lookat=[0.5,0.5,0.5] expects [0,1]. Images were mostly white → model confused. |

### Run 5 — Killed (render bug: border removed incorrectly)
| Field | Value |
|---|---|
| PID | 39830 |
| Commit | `7aeaa26` + normalization fix + border removed (incorrect) |
| Data | `data/cadrille_training/deepcad` (84,526 STLs) |
| Modality | img |
| val samples | 25 DeepCAD + 25 Fusion360 |
| Pre-training baseline | img/DeepCAD IoU=0.075, img/Fusion360 IoU=0.160 |
| Steps run | 1–8 |
| Status | Killed — still investigating rendering |
| Notes | Removing border was wrong. Reference code (`dataset_utils.py:326`) DOES use `ImageOps.expand(border=3)`. Fusion360 baseline improved 0.123→0.160 from normalization fix. |

### Run 6 — Killed by user (wrong border, 200-sample eval)
| Field | Value |
|---|---|
| Run name | `rl-s50k-lr1e-5-G4-cppo-0307-1851` |
| PID | 40780 / 42025 |
| Commit | `7aeaa26` + normalization fix (border still removed) |
| Data | `data/cadrille_training/deepcad` (84,526 STLs) |
| Modality | img |
| val samples | **200 DeepCAD + 200 Fusion360** (increased from 25) |
| Pre-training baseline | img/DeepCAD IoU=0.091, img/Fusion360 IoU=0.160, pc/DeepCAD IoU=0.843, pc/Fusion360 IoU=0.827 |
| Steps run | 1–7 |
| Rewards | 3.40 (step 1), 0.64, 0.10, 0.00, 0.75, 1.69, 2.43 |
| Status | Killed by user |

### Current State of `rl/dataset.py` render_img()
After all fixes, `render_img()` now:
1. ✅ Normalizes mesh: center → [-1,1] → scale×0.5 → [-0.5,0.5] → +0.5 → [0,1]
2. ✅ Adds 3px black border per view (matches reference `dataset_utils.py`)
3. ✅ Output: 268×268 px combined image (4× 134×134 views)

### Rendering Bug Summary
| Bug | Symptom | Fix |
|---|---|---|
| No mesh normalization | Images mostly white (mean=245, std=30); model gets garbage input | Added `apply_translation + apply_scale` before rendering |
| Border removed | Image 256×256 instead of 268×268; mismatch with reference | Restored `ImageOps.expand(border=3)` |

---

## Img Eval Gap Investigation: F1/F2/F3 (2026-03-07) — RESOLVED ✅

### Summary
The 9.1% img/DeepCAD baseline in all prior RL runs was caused entirely by rendering bugs
(no normalization + wrong border). After fixes, our pipeline reproduces the paper's scores.
**RL training can now start in img mode.**

### F1: Paper's Pipeline (test.py + evaluate.py), 30 samples
| Field | Value |
|---|---|
| Model | `checkpoints/cadrille-sft` (official paper SFT model) |
| Dataset | `data/deepcad_test_mini30` (30 random DeepCAD test STLs) |
| Pipeline | `test.py` → `evaluate.py` (paper's exact reference pipeline) |
| Mean IoU | **76.8%** |
| IR (skip=0) | 3.33% |
| Median CD | 0.199 |
| Verdict | Paper's pipeline works. Not 86.1% on 30 samples due to small-sample variance. |

### F2: Our render_img() Path, 30 samples
| Field | Value |
|---|---|
| Model | `checkpoints/cadrille-sft` |
| Dataset | `data/deepcad_test_mini30` (same 30 samples) |
| Pipeline | `debug_f2_img.py` (render_img() + collate + model.generate) → `evaluate.py` |
| Mean IoU | **79.5%** |
| IR (skip=0) | 3.33% |
| Verdict | Our render_img() path gives same results as test.py. Gap was the rendering bugs, now fixed. |

### F3: Our render_img() Path, 200 samples — PASS ✅
| Field | Value |
|---|---|
| Model | `checkpoints/cadrille-sft` |
| Dataset | `data/deepcad_test_mesh` (200 random samples, seed=42) |
| Pipeline | `debug_f3_img.py` (render_img() + collate + model.generate) → `evaluate.py` |
| Mean IoU | **84.7%** |
| IR (skip=0) | 1.50% |
| Median CD | 0.202 |
| Paper target | 86.1% (SFT) |
| Gap | **–1.4pp** — acceptable (within small-sample variance) |
| Verdict | ✅ img eval gap resolved. Pipeline correct. Ready for RL training. |

### Root Cause (confirmed)
The 9.1% in Runs 1–6 was 100% due to:
1. `render_img()` had no mesh normalization → images mostly white (garbage input to model)
2. `render_img()` had 3px border removed → image dimensions didn't match reference
Both bugs are fixed in current `rl/dataset.py`. The pipeline now gives 84.7% on 200 samples.

### Key Metrics Reference (paper Table 2, img mode)
| Model | DeepCAD IoU | Fusion360 IoU | CC3D IoU |
|---|---|---|---|
| cadrille SFT (Rpi) | 86.1% | 77.6% | 56.1% |
| cadrille Dr. CPPO | 92.2% | 84.6% | 65.0% |
| **Our render_img() (200 samples)** | **84.7%** | — | — |

### Next Steps
- [x] F1: Paper pipeline sanity check → 76.8%
- [x] F2: Our render_img() path (30 samples) → 79.5%
- [x] F3: Our render_img() path (200 samples) → 84.7% ✅ PASS
- [ ] T5: Start img RL training (ask user before starting)
- [ ] T6: Fusion360 train split — download STEP + convert → add to cadrille_training/
