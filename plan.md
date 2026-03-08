# RL Training Plan — Cadrille Reproduction

## Goal
Reproduce Table 2 (img mode) and Table 3 (pc mode) from the Cadrille paper (ICLR 2026).
Train RL on DeepCAD train split (~84k STLs) + Fusion360 train split using `img` modality.

---

## Paper Target Metrics

### Table 2 — CAD Reconstruction from Multi-view Images (img mode)
| Model | SFT Data | RL | DeepCAD IoU | Fusion360 IoU | CC3D IoU |
|-------|----------|----|-------------|---------------|----------|
| cadrille | Rpi | ✗ (SFT only) | 86.1% | 77.6% | 56.1% |
| cadrille | Rpi | DPO (D-i+F-i) | 86.9% | 78.5% | 56.0% |
| cadrille | Rpi | **Dr. CPPO (D-i+F-i)** | **92.2%** | **84.6%** | **65.0%** |

### Table 3 — CAD Reconstruction from Point Clouds (pc mode)
| Model | SFT Data | RL | DeepCAD IoU | Fusion360 IoU | CC3D IoU |
|-------|----------|----|-------------|---------------|----------|
| cadrille | Rpi | ✗ (SFT only) | 87.1% | 79.8% | 61.8% |
| cadrille | Rpi | **Dr. CPPO (D-i+F-i)** | **90.2%** | **85.0%** | **67.9%** |

---

## Current Situation (2026-03-07): RESOLVED — img eval gap fixed ✅

**RL training can start. img eval gap resolved.**

### Final eval results (after rendering fix, 500 samples)
| Modality | Dataset | Our eval | Paper SFT | Gap |
|----------|---------|----------|-----------|-----|
| pc | DeepCAD | 84.3% | 87.1% | –3pp — acceptable |
| pc | Fusion360 | 82.7% | 79.8% | +3pp — acceptable |
| img | DeepCAD | **86.4%** | **86.1%** | **+0.3pp ✅** |
| img | Fusion360 | **76.6%** | **77.6%** | **–1.0pp ✅** |

Checkpoint used: `checkpoints/cadrille-sft` = **official paper SFT model** (downloaded from
`Hula0401/cadrille-sft`). The model is correct. The gap is in our evaluation pipeline.

### What we know about the gap

**pc mode works (~87%)** — point cloud input carries exact 3D geometry; no rendering involved.

**img mode broken (~9%)** — 4-view render pipeline has a bug somewhere in the chain:
  model input → rendering → tokenizer → forward pass → generated code → executed mesh → IoU

**Key facts:**
- Both `deepcad_test_mesh` and `deepcad_train_mesh` STLs are already in **[0,1]^3 scale**
  (confirmed by checking bounds: max extent ≈ 1.0, centered near [0.5, 0.5, 0.5])
- `test.py` correctly does NOT normalize test-split meshes (they're already [0,1])
- Our `render_img()` in `rl/dataset.py` applies adaptive normalization, which is a near-no-op
  for [0,1] meshes — so rendering is not the issue
- The 3px border (`ImageOps.expand(border=3)`) is present in reference code and ours
- Rendering parameters match reference: camera_distance=-0.9, fronts=[[1,1,1],[-1,-1,-1],...]
- `checkpoints/cadrille-sft` is the paper's own model — it SHOULD achieve 86.1%

**Remaining suspects (in priority order):**
1. **Image preprocessing by the tokenizer/processor** — Qwen2-VL processor may apply
   different pixel normalization or resizing depending on how the PIL image is passed.
   The collate function passes `{'type': 'video', 'video': [PIL_image], 'fps': 1.0}` —
   this is the video pathway of Qwen2-VL. The processor's `min_pixels`/`max_pixels`
   settings determine token count and resize behavior. Mismatch here would corrupt input.
2. **Evaluation IoU computation** — our training eval uses voxel-sampling IoU while the
   paper uses volumetric boolean IoU (`evaluate.py`). May give different absolute values.
3. **Generated code quality** — model may generate syntactically valid but geometrically
   wrong code when given our rendered images.

---

## Fix Plan (must complete before RL training)

### Phase 1 — Isolate the gap with `test.py` + `evaluate.py` (paper's exact pipeline)

**Step F1: Quick sanity check — 30 samples, img mode, cadrille-sft**
- Run `test.py --split deepcad_test_mesh --mode img --checkpoint-path ./checkpoints/cadrille-sft`
  on 30 samples (modify `n_samples` or use `head -30` subset)
- Run `evaluate.py` on output
- **Expected if pipeline correct**: IoU ≈ 86%
- **Expected if pipeline broken**: IoU ≈ 9%
- This tells us: is `test.py`+`evaluate.py` reproducing the paper, or is the bug also there?

**Step F2 (if F1 gives ~86%)**: The bug is only in our `rl/eval.py` rendering pipeline.
- Compare `test.py`'s image format vs what `rl/eval.py` sends to the model
- Check processor parameters (min_pixels, max_pixels) — reference uses different values?
- Fix `rl/eval.py` to match `test.py`'s exact preprocessing

**Step F2 (if F1 gives ~9%)**: The bug is in the fundamental pipeline — rendering or inference.
- Debug: print a rendered image, save it, visually inspect
- Check if the model even "sees" the image: inspect pixel_values_videos shape/stats
- Compare with pc mode: same model, same code path except input modality

**Step F3: Verify fix on 200 samples**
- Once F1/F2 identifies the bug and we fix it, validate on 200 samples
- Target: img/DeepCAD IoU > 80% (within 6pp of paper's 86.1%)

---

## Data Pipeline Status

| Dataset | Location | Count | Status |
|---------|----------|-------|--------|
| CAD-Recode v1.5 | `data/cad-recode-v1.5/` | pkl only, no STLs | ⚠ STLs missing |
| deepcad_test_mesh | `data/deepcad_test_mesh/` | 8,047 STLs in [0,1] | ✅ Ready |
| fusion360_test_mesh | `data/fusion360_test_mesh/` | 1,726 STLs | ✅ Ready |
| deepcad_train_mesh | `data/deepcad_train_mesh/` | 84,526 STLs in [0,1] | ✅ Ready |
| cadrille_training | `data/cadrille_training/deepcad` → symlink | 84,526 STLs | ✅ Ready |
| fusion360_train_mesh | not downloaded | — | ❌ Missing |

---

## Full Task List (ordered by dependency)

### [DONE] Phase 1: Fix img eval gap

- [x] **F1**: 30-sample img eval via `test.py` + `evaluate.py` → 76.8% IoU (paper pipeline works)
- [x] **F2**: Root cause: rendering bugs (no normalization + wrong border). Fixed in `rl/dataset.py`. Our `render_img()` gives 79.5% on same 30 samples.
- [x] **F3**: 200-sample validation → **84.7% IoU** (paper: 86.1%, gap = 1.4pp ✅)

### [DONE] Phase 2: Data

- [x] T1: DeepCAD train split → STL (84,526 STLs in `deepcad_train_mesh/`)
- [x] T2: Create `data/cadrille_training/deepcad` symlink
- [x] T3: Update all RL configs → `cadrille_training/`
- [ ] T4 (optional): Fusion360 train split — download + convert → add symlink

### Phase 3: RL Training (ready to start)

- [ ] T5: Start img RL training (50k steps, 4080 SUPER, ~23 days)
  - Config: `configs/rl/4080.yaml`
  - Only start after F3 confirms img eval gap is fixed
- [ ] T6: Monitor training (reward trend, eval every 200 steps)
- [ ] T7: At step 1000, evaluate on full deepcad_test_mesh (8047 samples) for Table 2 comparison

---

## Architecture Notes

- `MeshDataset` in `rl/dataset.py`: `glob(**/*.stl, recursive=True)` — finds STLs in subdirs
- `render_img()` normalization: center→[-1,1]→scale(0.5)→[-0.5,0.5]→+0.5→[0,1] — no-op for [0,1] meshes
- `test.py` + `evaluate.py`: paper's reference pipeline; uses volumetric boolean IoU
- `rl/eval.py`: our training-time eval; uses voxel-sampling IoU (faster, lower absolute values)
- CPPO reward: IoU × 10; degenerate groups (std=0) → skipped via `_safe_histogram()`
- lr=1e-5 (not 3e-5): G=4 has higher gradient variance than official G=16

## Known Bugs Fixed

| Bug | File | Fix |
|-----|------|-----|
| `wandb.Histogram` crash on zero-range rewards | `rl/algorithms/cppo.py` | Added `_safe_histogram()` |
| Mesh not normalized before rendering (was mm scale) | `rl/dataset.py` | Added `transform_real_mesh` equivalent |
| 3px border removed (wrong) | `rl/dataset.py` | Restored `ImageOps.expand(border=3)` |
| Reward scale mismatch: was IoU×10 / -10; ref uses raw IoU / -1 | `rl/reward.py`, `rl/eval.py`, `rl/algorithms/cppo.py`, `rl/eval_passk.py` | Changed to raw IoU ∈ [0,1] and -1 for failure |
| Mesh normalization mismatch: only pred normalized to [0,1]; ref normalizes both to [-1,1] | `rl/reward.py` | `transform_real_mesh` applied to both pred and GT (center + scale 2/max_extent) |
| `bad_words_ids` missing from `model.generate()` in eval | `rl/eval.py`, `tools/eval_img.py` | Added `bad_words_ids=[[model.config.video_token_id]]` |
