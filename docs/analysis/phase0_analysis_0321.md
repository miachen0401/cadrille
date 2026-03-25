# Phase 0 Analysis — Full Test Set Evaluation

**Date:** 2026-03-21
**Script:** `tools/analyze_errors.py` (batch=32, max_new_tokens=768)
**Data:** `data/analysis/{combo}/metadata.jsonl`

---

## 1. Overall IoU Results (n=8046 / 1725, full test sets)

| Combo | mean IoU (success) | failure rate | success n |
|---|---|---|---|
| deepcad_sft_img | 0.8794 | 3.2% | 7785 |
| **deepcad_rl_img** | **0.9270** | **0.6%** | **8001** |
| deepcad_sft_pc | 0.9014 | 5.2% | 7631 |
| deepcad_rl_pc | 0.9071 | 0.7% | 7986 |
| fusion360_sft_img | 0.7963 | 5.8% | 1625 |
| **fusion360_rl_img** | **0.8562** | **1.2%** | **1705** |
| fusion360_sft_pc | 0.8376 | 9.2% | 1566 |
| fusion360_rl_pc | 0.8599 | 1.9% | 1693 |

### RL delta over SFT

| Dataset | Δ img IoU | Δ pc IoU | Δ img fail rate | Δ pc fail rate |
|---|---|---|---|---|
| DeepCAD | **+4.76pp** | +0.57pp | **−2.6pp** | −4.5pp |
| Fusion360 | **+5.99pp** | +2.23pp | **−4.6pp** | −7.3pp |

---

## 2. IoU Distribution

| Bucket | SFT_img | RL_img | SFT_pc | RL_pc |
|---|---|---|---|---|
| > 0.9 (DeepCAD) | 5361 (68.7%) | **6462 (81.1%)** | 5544 (71.0%) | 5917 (75.7%) |
| 0.7–0.9 | 1463 (18.8%) | 1062 (13.3%) | 1428 (18.3%) | 1438 (18.4%) |
| 0.5–0.7 | 459 (5.9%) | 264 (3.3%) | 350 (4.5%) | 354 (4.5%) |
| 0.3–0.5 | 255 (3.3%) | 120 (1.5%) | 153 (2.0%) | 132 (1.7%) |
| < 0.3 (success) | 247 (3.2%) | 93 (1.2%) | 156 (2.0%) | 145 (1.9%) |
| failure (0 / null) | 261 (3.2%) | **45 (0.6%)** | 415 (5.2%) | **60 (0.7%)** |

RL shifts the entire distribution rightward. The > 0.9 bucket grows by **+12.4pp** for img mode.

---

## 3. Failure Breakdown

| Error type | SFT_img | RL_img | SFT_pc | RL_pc |
|---|---|---|---|---|
| runtime_error | 132 | **7** | 178 | **10** |
| zero_iou | 127 | **38** | 218 | **46** |
| syntax_error | 2 | 0 | 19 | **4** |
| timeout | 0 | 0 | 0 | 0 |

**RL reduces runtime_error by 19× on img, 18× on pc.** This is the dominant source of IoU gain: RL reward signal teaches the model to generate syntactically valid, geometrically non-degenerate code.

---

## 4. Error Taxonomy (low-IoU success cases, n=50 per combo)

From `docs/analysis/error_taxonomy_0321.md`. Sampled IoU ∈ (0, 0.70).

| Category | SFT_img | RL_img | SFT_pc | RL_pc | **Total** |
|---|---|---|---|---|---|
| **dim_error** | 34 (68%) | 40 (80%) | 35 (70%) | 34 (68%) | **143 (72%)** |
| **wrong_primitive** | 7 (14%) | 4 (8%) | 8 (16%) | 7 (14%) | **26 (13%)** |
| degenerate | 2 (4%) | 4 (8%) | 2 (4%) | 4 (8%) | 12 (6%) |
| wrong_plane | 4 (8%) | 1 (2%) | 3 (6%) | 2 (4%) | 10 (5%) |
| partial_geom | 3 (6%) | 0 (0%) | 2 (4%) | 2 (4%) | 7 (4%) |
| feature_count | 0 | 1 (2%) | 0 | 1 (2%) | 2 (1%) |

### Key observations from taxonomy

**dim_error dominates (72%)** — Model gets the topology right (correct plane, correct primitive type, correct boolean structure) but gets numeric dimensions wrong. This is the remaining bottleneck after RL eliminates most structural errors.

**RL reduces structural errors** — wrong_plane: 7→3, wrong_primitive: 15→11, partial_geom: 5→2. RL reward strongly penalises axis confusion and wrong primitive choices.

**img vs pc difference is small** — After controlling for failure rate, the residual failure modes are nearly identical between modalities. This suggests the model's visual grounding is not the bottleneck; numeric precision is.

**wrong_primitive = box() fallback (13%)** — Model outputs `.box()` for shapes that require sketch+extrude. Achieves IoU 0.5–0.7 on flat/plate-like shapes but cannot reproduce non-rectangular cross-sections. More common in pc mode (15%) than img mode (11%).

---

## 5. Key Conclusions

### C1: RL already beats the paper target
Official cadrille-rl checkpoint: img/DeepCAD = **92.7%** vs paper Table 2 target of 92.2%.

### C2: RL's main contribution is failure reduction, not precision
Failure rate drops from 3.2% → 0.6% (img) and 5.2% → 0.7% (pc). The >0.9 bucket grows by 12.4pp.

### C3: img improves MORE than pc under RL
SFT: img(87.9%) < pc(90.1%). RL: img(**92.7%**) > pc(90.7%). RL training (train_modality=img) specifically optimises img mode. The img/pc gap reverses.

### C4: Residual failures are dominated by numeric precision (dim_error, 72%)
After RL eliminates structural errors, the remaining low-IoU cases have correct code topology but wrong dimensions. This is what future work needs to address.

### C5: img vs pc residual failures are nearly identical in type
The visual modality does not introduce qualitatively different failure modes.

---

## 6. Implications for Phase 1

### 6.1 Precision reward signal (highest priority)
Add a **continuous geometric precision term** — e.g., Chamfer distance reward — that gives gradient signal even when topology is correct but dimensions are off.

`R = α·R_iou + (1−α)·exp(−β·CD)`

### 6.2 Scale-aware input encoding (medium priority)
The model doesn't know the absolute scale of the shape from the image alone. Adding an explicit scale token or bounding box conditioning could reduce dim_error.

### 6.3 View augmentation (quick win)
RL training uses a fixed 4-view render. Augmenting camera angles during training forces the visual encoder to learn view-invariant features. One-line change in `render_img()`.

### 6.4 Wrong_primitive fix
13% of residual failures are box() fallback. Identifiable at inference time (no `.sketch()` call in output). Rejection-sampling or constrained decoding could force sketch+extrude.

---

## 7. Next Steps

- [x] Step 0.1: Full inference run
- [x] Step 0.2: IoU distribution analysis
- [x] Step 0.3: Error taxonomy (automated, n=200)
- [x] Step 0.4: SFT vs RL per-case delta
- [ ] Phase 1: Implement precision reward (Chamfer distance term) in `rl/reward.py`
- [ ] Phase 1: View augmentation in `rl/dataset.py`
- [ ] Run new H100 training with improved reward
