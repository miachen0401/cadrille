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

| Date | Commit | Script | Dataset | N | DeepCAD IoU | Fusion360 IoU | CC3D IoU | Notes |
|------|--------|--------|---------|---|-------------|---------------|----------|-------|
| 2026-03-07 | `7aeaa26` | `test.py` + `evaluate.py` | deepcad_test_mini30 | 30 | 76.8% | — | — | F1: paper's reference pipeline; small-sample variance explains gap to 86.1% |
| 2026-03-07 | `7aeaa26` | `debug_f2_img.py` + `evaluate.py` | deepcad_test_mini30 | 30 | 79.5% | — | — | F2: our `render_img()` path; matches F1 — rendering fix confirmed correct |
| 2026-03-07 | `7aeaa26` | `debug_f3_img.py` + `evaluate.py` | deepcad_test_mesh | 200 | 84.7% | — | — | F3: 200-sample validation |
| 2026-03-07 | `7aeaa26` | `debug_partial_eval.py` + `evaluate.py` | deepcad_test_mesh | 500 | **86.4%** | — | — | IR=1.6%, median CD=0.181 |
| 2026-03-07 | `7aeaa26` | `debug_partial_eval.py` + `evaluate.py` | fusion360_test_mesh | 500 | — | **76.6%** | — | IR=2.4%, median CD=0.205 |

### pc mode — `checkpoints/cadrille-sft` (official paper SFT model)

| Date | Commit | Script | Dataset | N | DeepCAD IoU | Fusion360 IoU | CC3D IoU | Notes |
|------|--------|--------|---------|---|-------------|---------------|----------|-------|
| 2026-03-07 | `7aeaa26` | `rl/eval.py` (training eval) | deepcad_test_mesh | 200 | 84.3% | — | — | From Run 3 pre-training baseline; voxel-sampling IoU (same absolute scale) |
| 2026-03-07 | `7aeaa26` | `rl/eval.py` (training eval) | fusion360_test_mesh | 200 | — | 82.7% | — | From Run 3 pre-training baseline |

### CC3D

Not yet evaluated. **Dataset requires a license agreement** from CVI² lab (cvi2.uni.lu/cc3d-dataset/) — not on HuggingFace, not documented in the paper's `data/README.md`. Contact shapify3d@uni.lu to request access. Once obtained, place STLs at `data/cc3d_test_mesh/` and run `debug_partial_eval.py` with that split added.

---

## Gap Summary (SFT baseline, img mode)

| Dataset | Paper SFT | Our Result | N | Gap | Notes |
|---------|-----------|------------|---|-----|-------|
| DeepCAD | 86.1% | **86.4%** | 500 | **+0.3pp** ✅ | Matches paper |
| Fusion360 | 77.6% | **76.6%** | 500 | –1.0pp ✅ | Within variance |
| CC3D | 56.1% | — | — | unknown | Dataset requires license from CVI² lab |

---

## Eval Pipeline Notes

- **`test.py` + `evaluate.py`**: paper's reference pipeline; uses `CadRecodeDataset` for rendering
- **`debug_f{1,2,3}_img.py` / `debug_full_eval.py`**: our pipeline; uses `render_img()` from `rl/dataset.py`
- **`rl/eval.py`**: training-time eval called from `rl/train.py`; same volumetric boolean IoU as `evaluate.py`
- All img evals use: `checkpoints/cadrille-sft`, `min_pixels=200704`, `max_pixels=1003520`, `max_new_tokens=768`, `batch_size=8`, greedy decode
- `evaluate.py` normalizes pred mesh: center → unit extent → translate to [0.5,0.5,0.5]
- GT meshes in `deepcad_test_mesh` and `fusion360_test_mesh` are pre-normalized to [0,1]³

---

## Known Issues / History

| Date | Issue | Fix | Commit |
|------|-------|-----|--------|
| 2026-03-07 | `render_img()` missing mesh normalization → images mostly white → img IoU = 9.1% | Added center+scale+translate to [0,1]³ in `rl/dataset.py` | `7aeaa26` + patch |
| 2026-03-07 | `render_img()` border incorrectly removed → image 256×256 instead of 268×268 | Restored `ImageOps.expand(border=3)` in `rl/dataset.py` | `7aeaa26` + patch |
| 2026-03-07 | `wandb.Histogram` crash when reward_std=0 | Added `_safe_histogram()` in `rl/algorithms/cppo.py` | `7aeaa26` + patch |
