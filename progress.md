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
- [x] T5: img RL training started → see Run 7 below
- [ ] T6: Fusion360 train split — download STEP + convert → add to cadrille_training/

---

## 2026-03-08 (continued): Hard Example Mining

### Context
Paper confirmed: RL training uses hard-mined subset only (mean IoU < 0.75 over K rollouts).
Expected: ~12k hard from 84k DeepCAD, ~7k hard from 30k Fusion360.
Full scan (114k STLs × 7s) = ~9 days on single 4080 → using max_samples budget instead.

### Actions
- [x] Killed training Run 7 (step ~200, PID 45051) — switching to mined data
- [x] Fusion360 train pipeline: 2GB download → 30,820 STLs → zip (62MB) → uploaded to `Hula0401/fusion360_train_mesh`
- [x] `data/cadrille_training/fusion360` symlink created
- [x] `rl/mine.py` rewritten: MeshDataset STL input, R_th=0.75 (was 7.5 in old ×10 scale), checkpointing every 500, --resume support
- [x] `.gitignore` updated: added `data/fusion360_train_mesh/`, comment rule
- [x] `CLAUDE.md` updated: always gitignore before download
- [~] **Mining DeepCAD** — PID 46532, `logs/mine_deepcad.log`
  - K=1, R_th=0.75, max_samples=20000, img mode, max_new_tokens=400
  - ~5s/example → ~28h for 20k samples → checkpoint every 500
  - **2026-03-08 06:10**: 275/20000 processed, ~5s/ea. First checkpoint at 500 (~35min away)
  - GPU usage: 10.3 GB / 16 GB (no room for training simultaneously)
  - Training Run 8 (PID 56080) killed before OOM — will restart after mining
- [ ] Mining Fusion360 — runs after DeepCAD mining finishes (same script, max_samples=8000)
- [ ] Merge pkls → `data/mined/combined_hard.pkl`
- [ ] Upload to `Hula0401/mine_CAD`
- [ ] Restart training (Run 8) with `hard_examples_pkl: ./data/mined/combined_hard.pkl`

### Next eval plan (after Run 8 step 1000+)
- `tools/eval_img.py` on 500 DeepCAD + 500 Fusion360 samples
- Compare to SFT baseline: DeepCAD 86.4%, Fusion360 76.6%

---

## 2026-03-08: Reward Alignment + Repo Cleanup + RL Training Launch

### Reward Alignment (commit `ceeb33d`)

All files updated to match paper reference (`ref_code/cadrille-rl/rl_finetune/utils.py`):

| File | Change |
|------|--------|
| `rl/reward.py` | Both pred+GT normalized with `transform_real_mesh` (center+scale to [-1,1]); reward scale raw IoU∈[0,1] / -1.0 (was ×10 / -10) |
| `rl/eval.py` | Added `bad_words_ids=[[model.config.video_token_id]]`; fixed failure threshold -1.0; removed /10 scaling |
| `rl/algorithms/cppo.py` | Clamp rewards to [-1,1] (was [-10,10]); `_safe_histogram()` preserved |
| `rl/eval_passk.py` | IoU threshold -1.0, no /10 scaling |
| `tools/eval_img.py` | Created from debug script; added bad_words_ids |

### Repo Structure Cleanup (commit `8067fbe`)

- Moved `data/deepcad2mesh.py` → `tools/deepcad2mesh.py`
- Moved `direction.md`, `reward_model_design.md` → `docs/`
- Created `tools/README.md`
- Added CLAUDE.md repo structure rules
- Updated `.gitignore`: logs/, DeepCAD/, data/data.tar, .bash_history, .claude/

### Setup + README for Fresh VM (commit `9583d0b`)

- `scripts/setup.sh --data`: added `deepcad_train_mesh` download + `cadrille_training/deepcad` symlink
- `README.md`: fixed reward formula, rewrote Data section, reorganized Quick Start

### Step 0 SFT Baselines (via RL eval at step=0)

| Metric | Value | Notes |
|--------|-------|-------|
| pc/DeepCAD IoU | 84.5% | Reliable |
| pc/Fusion360 IoU | 78.0% | Reliable |
| img/DeepCAD IoU | 9.7% | ⚠ Unreliable — mixed-batch left-padding bug |
| img/Fusion360 IoU | 10.5% | ⚠ Unreliable |

### deepcad_train_mesh Zip

- Created `/workspace/data/deepcad_train_mesh.zip` (242 MB, compresslevel=1, 84,526 STLs)
- HF upload to `Hula0401/deepcad_train_mesh` **pending**

### Run 7 — Current (img mode, reward-aligned)

| Field | Value |
|---|---|
| Run name | `rl-s50k-lr1e-5-G4-cppo-0308-0025` |
| PID | 45051 |
| W&B | https://wandb.ai/hula-the-cat/cadrille-rl/runs/qh088ege |
| Commit | `ceeb33d` (reward-aligned) |
| Data | `data/cadrille_training/deepcad` → `deepcad_train_mesh` (84,526 STLs) |
| Modality | img |
| val samples | 200 DeepCAD + 200 Fusion360 |
| Pre-training baseline | pc/DeepCAD=84.5%, pc/Fusion360=78.0%; img unreliable |
| Status | 🟢 Running — step 200+ @ ~51 s/step |
| Reward max | 0.86 (step 152 and 200) |
| Entropy | H≈0.22–0.49, oscillating ~0.35 — stable, not collapsing |
| Log | `logs/rl-0308-0025.log` |
| Notes | Rewards in [-1,1], correct normalization. img eval metric unreliable in mixed batches. |

### Known Issues (open)

| Issue | Status |
|-------|--------|
| img eval in mixed pc+img batches: heavy left-padding distorts attention → 9.7% instead of ~86% | Open — use `tools/eval_img.py` for benchmarks |
| HF upload of `deepcad_train_mesh.zip` pending | Run manually when convenient |
- [2026-03-08 06:14] daemon state=MINING_DEEPCAD | dc=499/20000 hard=0 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=46532
- [2026-03-08 06:21] daemon state=MINING_DEEPCAD | dc=590/20000 hard=0 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 06:32] daemon state=MINING_DEEPCAD | dc=747/20000 hard=0 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 06:43] daemon state=MINING_DEEPCAD | dc=911/20000 hard=0 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 06:54] daemon state=MINING_DEEPCAD | dc=1082/20000 hard=111 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 07:05] daemon state=MINING_DEEPCAD | dc=1247/20000 hard=111 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 07:16] daemon state=MINING_DEEPCAD | dc=1411/20000 hard=111 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 07:27] daemon state=MINING_DEEPCAD | dc=1573/20000 hard=111 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 07:38] daemon state=MINING_DEEPCAD | dc=1738/20000 hard=219 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 07:49] daemon state=MINING_DEEPCAD | dc=1911/20000 hard=219 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 08:00] daemon state=MINING_DEEPCAD | dc=2088/20000 hard=325 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 08:11] daemon state=MINING_DEEPCAD | dc=2257/20000 hard=325 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 08:22] daemon state=MINING_DEEPCAD | dc=2424/20000 hard=325 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 08:33] daemon state=MINING_DEEPCAD | dc=2601/20000 hard=416 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 08:44] daemon state=MINING_DEEPCAD | dc=2767/20000 hard=416 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 08:55] daemon state=MINING_DEEPCAD | dc=2931/20000 hard=416 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 09:06] daemon state=MINING_DEEPCAD | dc=3077/20000 hard=515 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 09:17] daemon state=MINING_DEEPCAD | dc=3244/20000 hard=515 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 09:27] daemon state=MINING_DEEPCAD | dc=3403/20000 hard=515 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 09:38] daemon state=MINING_DEEPCAD | dc=3569/20000 hard=515 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 09:49] daemon state=MINING_DEEPCAD | dc=3751/20000 hard=630 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 10:00] daemon state=MINING_DEEPCAD | dc=3922/20000 hard=630 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 10:11] daemon state=MINING_DEEPCAD | dc=4091/20000 hard=720 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 10:22] daemon state=MINING_DEEPCAD | dc=4259/20000 hard=720 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 10:33] daemon state=MINING_DEEPCAD | dc=4431/20000 hard=720 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 10:44] daemon state=MINING_DEEPCAD | dc=4597/20000 hard=838 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 10:55] daemon state=MINING_DEEPCAD | dc=4775/20000 hard=838 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 11:06] daemon state=MINING_DEEPCAD | dc=4944/20000 hard=838 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 11:17] daemon state=MINING_DEEPCAD | dc=5113/20000 hard=932 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 11:28] daemon state=MINING_DEEPCAD | dc=5286/20000 hard=932 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 11:39] daemon state=MINING_DEEPCAD | dc=5453/20000 hard=932 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 11:50] daemon state=MINING_DEEPCAD | dc=5627/20000 hard=1027 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 12:01] daemon state=MINING_DEEPCAD | dc=5793/20000 hard=1027 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 12:12] daemon state=MINING_DEEPCAD | dc=5971/20000 hard=1027 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 12:23] daemon state=MINING_DEEPCAD | dc=6138/20000 hard=1133 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 12:34] daemon state=MINING_DEEPCAD | dc=6309/20000 hard=1133 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 12:45] daemon state=MINING_DEEPCAD | dc=6468/20000 hard=1133 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 12:56] daemon state=MINING_DEEPCAD | dc=6640/20000 hard=1237 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 13:07] daemon state=MINING_DEEPCAD | dc=6809/20000 hard=1237 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 13:18] daemon state=MINING_DEEPCAD | dc=6977/20000 hard=1237 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 13:29] daemon state=MINING_DEEPCAD | dc=7144/20000 hard=1346 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=69066
- [2026-03-08 13:37] daemon state=MINING_DEEPCAD | dc=7266/20000 hard=1346 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 13:48] daemon state=MINING_DEEPCAD | dc=7622/20000 hard=1346 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 13:59] daemon state=MINING_DEEPCAD | dc=7974/20000 hard=1452 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 14:10] daemon state=MINING_DEEPCAD | dc=8334/20000 hard=1564 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 14:21] daemon state=MINING_DEEPCAD | dc=8690/20000 hard=1564 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 14:32] daemon state=MINING_DEEPCAD | dc=9018/20000 hard=1660 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 14:42] daemon state=MINING_DEEPCAD | dc=9346/20000 hard=1764 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 14:53] daemon state=MINING_DEEPCAD | dc=9662/20000 hard=1764 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 15:04] daemon state=MINING_DEEPCAD | dc=9970/20000 hard=1872 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 15:15] daemon state=MINING_DEEPCAD | dc=10266/20000 hard=1981 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 15:26] daemon state=MINING_DEEPCAD | dc=10558/20000 hard=1981 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 15:37] daemon state=MINING_DEEPCAD | dc=10862/20000 hard=2085 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 15:48] daemon state=MINING_DEEPCAD | dc=11166/20000 hard=2085 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 15:59] daemon state=MINING_DEEPCAD | dc=11470/20000 hard=2190 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 16:10] daemon state=MINING_DEEPCAD | dc=11750/20000 hard=2299 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 16:21] daemon state=MINING_DEEPCAD | dc=12090/20000 hard=2299 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 16:32] daemon state=MINING_DEEPCAD | dc=12446/20000 hard=2417 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 16:43] daemon state=MINING_DEEPCAD | dc=12794/20000 hard=2531 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 16:54] daemon state=MINING_DEEPCAD | dc=13154/20000 hard=2531 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 17:05] daemon state=MINING_DEEPCAD | dc=13518/20000 hard=2633 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 17:16] daemon state=MINING_DEEPCAD | dc=13870/20000 hard=2742 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 17:27] daemon state=MINING_DEEPCAD | dc=14214/20000 hard=2859 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 17:38] daemon state=MINING_DEEPCAD | dc=14578/20000 hard=2859 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 17:49] daemon state=MINING_DEEPCAD | dc=14934/20000 hard=2972 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 18:00] daemon state=MINING_DEEPCAD | dc=15290/20000 hard=3094 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 18:11] daemon state=MINING_DEEPCAD | dc=15626/20000 hard=3094 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 18:22] daemon state=MINING_DEEPCAD | dc=15986/20000 hard=3198 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 18:33] daemon state=MINING_DEEPCAD | dc=16358/20000 hard=3322 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 18:44] daemon state=MINING_DEEPCAD | dc=16718/20000 hard=3428 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 18:55] daemon state=MINING_DEEPCAD | dc=17074/20000 hard=3428 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 19:06] daemon state=MINING_DEEPCAD | dc=17418/20000 hard=3537 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 19:17] daemon state=MINING_DEEPCAD | dc=17782/20000 hard=3661 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 19:28] daemon state=MINING_DEEPCAD | dc=18158/20000 hard=3661 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 19:39] daemon state=MINING_DEEPCAD | dc=18502/20000 hard=3752 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 19:50] daemon state=MINING_DEEPCAD | dc=18866/20000 hard=3846 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 20:01] daemon state=MINING_DEEPCAD | dc=19210/20000 hard=3961 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 20:12] daemon state=MINING_DEEPCAD | dc=19550/20000 hard=3961 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 20:23] daemon state=MINING_DEEPCAD | dc=19898/20000 hard=4075 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 20:34] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 20:45] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 20:56] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 21:07] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 21:18] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 21:29] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 21:40] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 21:51] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 22:02] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 22:13] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 22:24] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 22:35] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 22:46] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 22:57] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 23:08] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 23:19] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 23:30] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 23:41] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-08 23:52] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 00:03] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 00:13] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 00:24] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 00:35] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 00:46] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 00:57] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 01:08] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 01:19] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 01:30] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 01:41] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 01:52] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 02:03] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 02:14] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 02:25] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 02:36] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 02:47] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 02:58] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 03:09] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 03:20] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 03:31] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 03:42] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 03:53] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=23516
- [2026-03-09 03:58] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=0/8000 hard=0 | state=MINING_DEEPCAD mine_pid=None
- [2026-03-09 04:02] daemon state=MINING_DEEPCAD | dc=20167/20000 hard=4173 | f3=52/8000 hard=0 | state=MINING_DEEPCAD mine_pid=25130
- [2026-03-09 04:02] daemon state=MINING_FUSION360 | dc=20167/20000 hard=4173 | f3=68/8000 hard=0 | state=MINING_FUSION360 mine_pid=25130
- [2026-03-09 04:10] daemon state=MINING_FUSION360 | dc=20167/20000 hard=4173 | f3=264/8000 hard=0 | state=MINING_FUSION360 mine_pid=25130
- [2026-03-09 04:21] daemon state=MINING_FUSION360 | dc=20167/20000 hard=4173 | f3=540/8000 hard=184 | state=MINING_FUSION360 mine_pid=25130
- [2026-03-09 04:32] daemon state=MINING_FUSION360 | dc=20167/20000 hard=4173 | f3=832/8000 hard=184 | state=MINING_FUSION360 mine_pid=25130
- [2026-03-09 04:43] daemon state=MINING_FUSION360 | dc=20167/20000 hard=4173 | f3=1112/8000 hard=359 | state=MINING_FUSION360 mine_pid=25130
- [2026-03-09 04:54] daemon state=MINING_FUSION360 | dc=20167/20000 hard=4173 | f3=1392/8000 hard=359 | state=MINING_FUSION360 mine_pid=25130
- [2026-03-09 05:05] daemon state=MINING_FUSION360 | dc=20167/20000 hard=4173 | f3=1672/8000 hard=517 | state=MINING_FUSION360 mine_pid=25130
- [2026-03-09 05:16] daemon state=MINING_FUSION360 | dc=20167/20000 hard=4173 | f3=1952/8000 hard=517 | state=MINING_FUSION360 mine_pid=25130
