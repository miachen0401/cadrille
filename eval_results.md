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

## Gap Summary (SFT baseline, img mode)

| Dataset | Paper SFT | Our Result | N | Gap | Notes |
|---------|-----------|------------|---|-----|-------|
| DeepCAD | 86.1% | **86.4%** | 500 | **+0.3pp** ✅ | Matches paper |
| Fusion360 | 77.6% | **76.6%** | 500 | –1.0pp ✅ | Within variance |
| CC3D | 56.1% | — | — | unknown | Requires license from CVI² lab |

---

## RL Training Runs

| Run | Commit | Config | Start | W&B | Status |
|-----|--------|--------|-------|-----|--------|
| `rl-s50k-lr1e-5-G4-cppo-0308-0025` | `ceeb33d` | `configs/rl/4080.yaml` | 2026-03-08 | [qh088ege](https://wandb.ai/hula-the-cat/cadrille-rl/runs/qh088ege) | 🟢 Running |

**Config:** G=4, lr=1e-5, img mode, DeepCAD train (84k STLs), max_steps=50k, ~54 s/step on RTX 4080 SUPER.

---

## Eval Pipeline Notes

- **`tools/eval_img.py` + `evaluate.py`**: primary benchmark pipeline; uses `render_img()` from `rl/dataset.py`, calls paper's `evaluate.py` for volumetric boolean IoU + Chamfer Distance. GT meshes already in [0,1]³; pred normalized to [0,1]³ by `evaluate.py`.
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
