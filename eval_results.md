# Evaluation Results

Tracks paper targets vs our reproduction results.
Each entry records the exact commit, script, dataset, sample count, and metrics.

---

## Paper Targets

### Table 2 — CAD Reconstruction from Multi-view Images (img mode)

| Model | SFT Data | RL | DeepCAD IoU | Fusion360 IoU | CC3D IoU |
|-------|----------|----|-------------|---------------|----------|
| cadrille | Rpi | ✗ (SFT only) | 86.1% | 77.6% | 56.1% |
| cadrille | Rpi | DPO (D-i+F-i) | 86.9% | 78.5% | 56.0% |
| cadrille | Rpi | Dr. CPPO (D-i+F-i) | **92.2%** | **84.6%** | **65.0%** |

### Table 3 — CAD Reconstruction from Point Clouds (pc mode)

| Model | SFT Data | RL | DeepCAD IoU | Fusion360 IoU | CC3D IoU |
|-------|----------|----|-------------|---------------|----------|
| cadrille | Rpi | ✗ (SFT only) | 87.1% | 79.8% | 61.8% |
| cadrille | Rpi | Dr. CPPO (D-i+F-i) | **90.2%** | **85.0%** | **67.9%** |

---

## Our Results

### img mode — `checkpoints/cadrille-sft` (official paper SFT model)

Evaluated with `tools/eval_img.py` + `evaluate.py` (paper's reference pipeline).

| Date | Commit | Dataset | N | DeepCAD IoU | Fusion360 IoU | CC3D IoU | Notes |
|------|--------|---------|---|-------------|---------------|----------|-------|
| 2026-03-07 | `7aeaa26` | deepcad_test_mesh | 500 | **86.4%** | — | — | IR=1.6%, median CD=0.181 |
| 2026-03-07 | `7aeaa26` | fusion360_test_mesh | 500 | — | **76.6%** | — | IR=2.4%, median CD=0.205 |

### pc mode — `checkpoints/cadrille-sft` (official paper SFT model)

Evaluated with `rl/eval.py` (training-time eval; volumetric boolean IoU, both meshes normalized to [-1,1]³).

| Date | Commit | Dataset | N | DeepCAD IoU | Fusion360 IoU | CC3D IoU | Notes |
|------|--------|---------|---|-------------|---------------|----------|-------|
| 2026-03-08 | `ceeb33d` | deepcad_test_mesh | 200 | **84.5%** | — | — | RL run step=0 baseline |
| 2026-03-08 | `ceeb33d` | fusion360_test_mesh | 200 | — | **78.0%** | — | RL run step=0 baseline |

### CC3D

Not yet evaluated. **Dataset requires a license agreement** from CVI² lab (cvi2.uni.lu/cc3d-dataset/).
Contact shapify3d@uni.lu. Once obtained, place STLs at `data/cc3d_test_mesh/` and run:
```
python3 tools/eval_img.py --splits cc3d
```

---

---

## Phase 0 Full Test Set Baselines (2026-03-21, N=8046/1725)

Evaluated with `tools/analyze_errors.py` (batch=32, max_new_tokens=768, greedy).
Full DeepCAD test (n=8046) and Fusion360 test (n=1725). All failure cases (runtime_error, zero_iou, syntax_error) count as IoU=0.

| Checkpoint | Modality | DeepCAD IoU | Fusion360 IoU | DC fail% | F360 fail% | Notes |
|---|---|---|---|---|---|---|
| `cadrille-sft` | img | 87.94% | 79.65% | 3.24% | 5.80% | runtime_err: DC 132, F360 46 |
| `cadrille-rl` (official) | img | **92.70%** | **85.62%** | **0.56%** | **1.22%** | runtime_err: DC 7, F360 2 |
| `cadrille-sft` | pc | 90.14% | 83.76% | 5.15% | 9.22% | runtime_err: DC 178, F360 68 |
| `cadrille-rl` (official) | pc | **90.71%** | **86.01%** | **0.75%** | **1.91%** | runtime_err: DC 10, F360 5 |

### RL delta over SFT (full test set)

| Dataset | Δ img IoU | Δ pc IoU | Δ img fail% | Δ pc fail% |
|---|---|---|---|---|
| DeepCAD | **+4.76pp** | +0.57pp | **−2.68pp** | −4.40pp |
| Fusion360 | **+5.97pp** | +2.25pp | **−4.58pp** | −7.31pp |

### Error breakdown (full test set, all 8 combos)

| Combo | success | zero_iou | runtime_err | syntax_err | failure% |
|---|---|---|---|---|---|
| deepcad_sft_img | 7785 | 127 | 132 | 2 | 3.24% |
| deepcad_rl_img | **8001** | 38 | 7 | 0 | **0.56%** |
| deepcad_sft_pc | 7631 | 218 | 178 | 19 | 5.15% |
| deepcad_rl_pc | **7986** | 46 | 10 | 4 | **0.75%** |
| fusion360_sft_img | 1625 | 54 | 46 | 0 | 5.80% |
| fusion360_rl_img | **1705** | 18 | 2 | 0 | **1.22%** |
| fusion360_sft_pc | 1566 | 76 | 68 | 15 | 9.22% |
| fusion360_rl_pc | **1693** | 22 | 5 | 5 | **1.91%** |

Key finding: **`cadrille-rl` img/DeepCAD = 92.70% beats paper target 92.2%**. RL reduces runtime_error by 18× (img) and 18× (pc). Residual failures dominated by dim_error (72%) per error taxonomy — see `docs/analysis/`.

---

## Gap Summary (SFT baseline, img mode)

| Dataset | Paper SFT | Our Result | N | Gap | Notes |
|---------|-----------|------------|---|-----|-------|
| DeepCAD | 86.1% | **86.4%** | 500 | **+0.3pp** ✅ | Matches paper |
| Fusion360 | 77.6% | **76.6%** | 500 | –1.0pp ✅ | Within variance |
| CC3D | 56.1% | — | — | unknown | Requires license from CVI² lab |

---

## Comprehensive Baselines (2026-03-19, N=300/dataset, rl/eval.py pipeline)

Eval script: `/tmp/eval_ckpt.py` — uses `rl/eval.py:run_validation`, batch=16, workers=10, greedy decode, max_new_tokens=1024.
Both DeepCAD and Fusion360 test sets, 300 random samples each.

| Model | HF / Local | img/DeepCAD | img/Fusion360 | pc/DeepCAD | pc/Fusion360 | Notes |
|-------|-----------|-------------|---------------|------------|--------------|-------|
| `cadrille-sft` | `checkpoints/cadrille-sft` | **86.1%** | **77.4%** | 85.3% | 78.6% | Official paper SFT; true img baseline |
| A100 ckpt-900 | `Hula0401/cad_ckpt` rl-0318 | **86.5%** | 76.0% | 87.7% | 78.3% | lr=3.2e-6, G=16, step=900 |
| A100 ckpt-1200 | `Hula0401/cad_ckpt` rl-0318 | 86.0% | 77.2% | **87.9%** | 78.3% | lr=3.2e-6, G=16, step=1200 |
| rl-0311 ckpt-9000 | `Hula0401/cad_ckpt` rl-0311 | 82.2% | 72.0% | 87.2% | 79.3% | lr=1e-5 (too high), img regressed |
| rl-0318 ckpt-2100 | `Hula0401/cad_ckpt` rl-0318 | 85.3% | 77.0% | **87.3%** | 78.6% | lr=3.2e-6, G=16, bs=16, step=2100; N=300 |

**Paper targets (Table 2/3, Dr. CPPO):** img/DeepCAD=92.2%, img/Fusion360=84.6%, pc/DeepCAD=90.2%, pc/Fusion360=85.0%

**Takeaways:**
- A100 run (lr=3.2e-6, G=16) is on track: pc already +2.5pp over SFT at step 1200; img holding steady
- rl-0311 (lr=1e-5) overtrained: img/Fusion360 dropped to 72% vs SFT 77.4%
- Still ~6pp gap to paper target on img/DeepCAD; A100 run needs more steps

---

## Same-Config Comparison (2026-03-21, N=50, `eval/` framework)

All checkpoints evaluated on the **same 50 STLs per dataset** using `eval/runner.py` (configs/eval/quick.yaml variant: batch_size_img=4, batch_size_pc=16, greedy, max_new_tokens=768, flash_attn2).
Directly comparable numbers — no dataset/sampling differences.

| Checkpoint | DC/img | DC/pc | F360/img | F360/pc | Notes |
|---|---|---|---|---|---|
| `cadrille-sft` | 83.75% | 85.04% | 75.99% | 80.43% | Official SFT baseline |
| `cadrille-rl` (official) | **90.00%** | **89.82%** | **83.59%** | **84.88%** | Official RL (paper target) |
| `rl-0320-lr2e-5-s90` | 83.76% | 82.79% | 75.50% | 76.18% | ⚠ training temp=0.3 (others=1.0); only 90 steps |
| `rl-0320-lr1e-5-s360-a` | 85.04% | 87.76% | 77.09% | 79.08% | H100, lr=1e-5, step=360, seed A |
| `rl-0320-lr1e-5-s360-b` | 86.16% | 83.53% | 77.32% | 81.52% | H100, lr=1e-5, step=360, seed B |
| `rl-0321-lr3e-5-s60` | 84.11% | 83.05% | 78.22% | 82.65% | H100, lr=3e-5, step=60 (early) |

**Notes:**
- `rl-0320-lr2e-5-s90` degradation is **primarily caused by training temperature=0.3** (all other runs trained at temperature=1.0), not just the higher lr or low step count. Lower temperature during rollout generation reduces policy diversity and leads to faster entropy collapse.
- H100 runs at step 360 are ~1.5–2.5pp behind official-rl on DC/img; still ~4–5pp gap to close.
- `cadrille-rl` official at N=50 gives 90.00% DC/img — consistent with full-set 92.70% (N=50 variance expected, harder subset possible).

---

## RL Training Runs

| Run | Config | GPU | lr | G | Dataset | Steps | W&B | Status |
|-----|--------|-----|----|---|---------|-------|-----|--------|
| `rl-s50k-lr1e-5-G4-cppo-0311-0259` | `4080.yaml` | RTX 4080S | 1e-5 | 4 | s50k | 4500–9000 | — | ❌ Done (overtrained) |
| `rl-s50k-lr3.2e-6-G16-cppo-0318-0205` | A100 | A100 | 3.2e-6 | 16 | s50k | 900–1200 | — | ❌ Done |
| `rl-s3600-lr2e-5-G16-cppo-0320-0313` | H100 | H100 | 2e-5 | 16 | s3600 | 90 | — | ❌ Crashed at 90 steps; **training temp=0.3** (bug — others use 1.0) |
| `rl-s3600-lr1e-5-G16-cppo-0320-0524` | H100 | H100 | 1e-5 | 16 | s3600 | 360 | — | ❌ Done |
| `rl-s3600-lr1e-5-G16-cppo-0320-0531` | H100 | H100 | 1e-5 | 16 | s3600 | 360 | 25fgf79l | ❌ Done |
| `rl-s3600-lr3e-5-G16-cppo-0321-0209` | H100 | H100 | 3e-5 | 16 | s3600 | 60 | — | ❌ Early ckpt only |

---

## Eval Pipeline Notes

- **`eval/` framework** (primary, 2026-03-21+): unified one-command eval. `python3 -m eval.runner configs/eval/quick.yaml`. Config-driven: per-dataset n_samples, pass@k, separate GT/pred renders, resource tuning (batch_size_img/pc, prep_threads). Output in `eval_outputs/{tag}/`. Use this for all new evals.
- **`tools/eval_img.py` + `evaluate.py`**: legacy benchmark pipeline; uses `render_img()` from `rl/dataset.py`, calls paper's `evaluate.py` for volumetric boolean IoU + Chamfer Distance. GT meshes already in [0,1]³; pred normalized to [0,1]³ by `evaluate.py`.
- **`rl/eval.py`**: training-time eval called every `eval_steps` from `rl/train.py`; normalizes both pred and GT to [-1,1]³ before IoU computation. **pc metric is reliable; img metric is unreliable** (mixed pc+img batches cause attention issues — img items get heavy left-padding to match long img prompt lengths, distorting generation).
- All img evals use: `min_pixels=200704`, `max_pixels=1003520`, `max_new_tokens=768`, greedy decode, `bad_words_ids=[[video_token_id]]`.
- GT meshes in `deepcad_test_mesh` and `fusion360_test_mesh` are pre-normalized to [0,1]³.

---

## Known Issues / History

| Date | Issue | Fix | Commit |
|------|-------|-----|--------|
| 2026-03-07 | `render_img()` missing mesh normalization → images mostly white → img IoU = 9.1% | Added center+scale+translate to [0,1]³ in `rl/dataset.py` | `ceeb33d` |
| 2026-03-07 | `render_img()` border removed → image 256×256 instead of 268×268 | Restored `ImageOps.expand(border=3)` in `rl/dataset.py` | `ceeb33d` |
| 2026-03-07 | `wandb.Histogram` crash when reward_std=0 | Added `_safe_histogram()` in `rl/algorithms/cppo.py` | `ceeb33d` |
| 2026-03-08 | Reward scale mismatch vs paper (IoU×10 / -10 instead of raw IoU / -1) | Updated `rl/reward.py`, `rl/eval.py`, `rl/algorithms/cppo.py`, `rl/eval_passk.py` | `ceeb33d` |
| 2026-03-08 | Mesh normalization mismatch: only pred normalized; ref normalizes both to [-1,1] | `transform_real_mesh` applied to both pred and GT in all worker paths | `ceeb33d` |
| 2026-03-08 | `bad_words_ids` missing from eval `model.generate()` calls | Added to `rl/eval.py` and `tools/eval_img.py` | `ceeb33d` |
| 2026-03-08 | img eval unreliable in mixed pc+img batches (heavy left-padding distorts img attention) | Sort examples by modality before batching in `rl/eval.py:eval_one_pass` | `HEAD` |
