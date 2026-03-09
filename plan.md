# RL Training Plan — Cadrille Reproduction

## Goal
Reproduce Table 2 (img mode) and Table 3 (pc mode) from the Cadrille paper (ICLR 2026).
Target: DeepCAD IoU 92.2% (img), 90.2% (pc) via Dr. CPPO on DeepCAD + Fusion360 train.

---

## Paper Target Metrics

### Table 2 — CAD Reconstruction from Multi-view Images (img mode)
| Model | SFT Data | RL | DeepCAD IoU | Fusion360 IoU | CC3D IoU |
|-------|----------|----|-------------|---------------|----------|
| cadrille | Rpi | ✗ (SFT only) | 86.1% | 77.6% | 56.1% |
| cadrille | Rpi | **Dr. CPPO (D-i+F-i)** | **92.2%** | **84.6%** | **65.0%** |

### Table 3 — CAD Reconstruction from Point Clouds (pc mode)
| Model | SFT Data | RL | DeepCAD IoU | Fusion360 IoU | CC3D IoU |
|-------|----------|----|-------------|---------------|----------|
| cadrille | Rpi | **Dr. CPPO (D-i+F-i)** | **90.2%** | **85.0%** | **67.9%** |

---

## Data Pipeline (DONE ✅)

| Dataset | Location | Count | HuggingFace | Status |
|---------|----------|-------|-------------|--------|
| deepcad_train_mesh | `data/cadrille_training/deepcad` | 84,526 STLs | `Hula0401/deepcad_train_mesh` | ✅ |
| fusion360_train_mesh | `data/cadrille_training/fusion360` | 30,820 STLs | `Hula0401/fusion360_train_mesh` | ✅ |
| deepcad_test_mesh | `data/deepcad_test_mesh` | 8,047 STLs | `Hula0401/deepCAD_test` | ✅ |
| fusion360_test_mesh | `data/fusion360_test_mesh` | 1,726 STLs | `Hula0401/fusion360_test_mesh` | ✅ |
| cadrille-sft checkpoint | `checkpoints/cadrille-sft` | — | `maksimko123/cadrille` | ✅ |

---

## Paper Data Selection (CRITICAL — confirmed from paper)

> "Only use examples where mean reward over K=3 SFT rollouts < R_th=7.5 (= IoU 0.75 in raw scale)"

| Dataset | Raw | After mining (paper) | Expected ours |
|---------|-----|---------------------|---------------|
| DeepCAD train | ~160k | **~50k** hard examples | 84k → ~12k (img, IoU<0.75) |
| Fusion360 train | 6,900 designs | **~3k** hard examples | 30k STLs → ~7k |
| **Total** | | **~53k** | **~19k** |

Mining output → `data/mined/` → upload to `Hula0401/mine_CAD`

---

## Phase 3: Hard Example Mining (CURRENT)

### Status
- [x] Training Run 7 killed (step ~200, PID 45051)
- [x] Fusion360 train data ready (30,820 STLs)
- [x] `cadrille_training/fusion360` symlink created
- [x] `rl/mine.py` rewritten: MeshDataset, R_th=0.75, checkpointing, resume
- [~] **Mining DeepCAD** — PID TBD, log `logs/mine_deepcad.log`
  - Config: K=1, R_th=0.75, max_new_tokens=400, img mode
  - Input: 84,526 STLs → expected ~12k hard examples
  - Speed: ~7s/example → ~164 hours full scan OR use --max-samples 20000 (~39h)
- [ ] **Mining Fusion360** — run after DeepCAD mining completes (or in sequence)
  - Input: 30,820 STLs → expected ~7k hard examples
  - Speed: ~7s/example → ~60h full scan OR --max-samples 8000 (~16h)
- [ ] Merge DeepCAD + Fusion360 pkl → `data/mined/combined_hard.pkl`
- [ ] Upload to `Hula0401/mine_CAD`
- [ ] Update configs: `hard_examples_pkl: ./data/mined/combined_hard.pkl`

### Mining constraints on RTX 4080
Full scan of 115k STLs with K=1 ≈ 9 days. Strategy:
- Run overnight, collect as many as possible
- Checkpoint every 500 examples (resumable with --resume)
- Target: 20k DeepCAD + 8k Fusion360 scanned → ~4k hard examples
- Upload partial results, retrain, re-mine in background

---

## Phase 4: RL Training (Run 8 — after mining)

Config changes vs Run 7:
- `data_dir: null` (unused)
- `hard_examples_pkl: ./data/mined/combined_hard.pkl`
- Both DeepCAD + Fusion360 hard examples
- Same hyperparams: G=4, lr=1e-5, img mode

Monitoring checklist (check every ~2h):
- [ ] Reward trend rising? (target: mean > 0.5 by step 500)
- [ ] Entropy stable? (H > 0.1 — if drops to 0 → collapse, stop)
- [ ] Eval IoU improving? (check step 200, 400, 600 evals)
- [ ] Disk usage < 900 GB? (checkpoints ~4.5 GB each, save_total_limit=10)
- [ ] Training log fresh? (grep last timestamp)

---

## Phase 5: Evaluation (after Run 8 hits step 1000+)

```bash
python tools/eval_img.py \
    --checkpoint ./checkpoints/cadrille-rl-run8/checkpoint-1000 \
    --split deepcad_test_mesh --n-samples 500
```

Compare to:
- Paper SFT baseline: DeepCAD 86.1%, Fusion360 77.6%
- Paper RL target: DeepCAD 92.2%, Fusion360 84.6%
- Our SFT baseline: DeepCAD 86.4%, Fusion360 76.6%

---

## Architecture Notes

- Reward: raw IoU ∈ [0,1] / -1 for failure (both pred+GT normalized to [-1,1]³)
- CPPO reward clamp: [-1, 1]; degenerate groups (std=0) skipped
- `MeshDataset`: `glob(**/*.stl, recursive=True)` — picks up subdirs automatically
- `RLDataset`: loads from pkl `{gt_mesh_path, file_name}` — used when `hard_examples_pkl` set
- img eval in mixed pc+img batches unreliable → use `tools/eval_img.py` for benchmarks
- cadrille_training/ combines deepcad/ + fusion360/ via symlinks (MeshDataset recurses)

---

## Known Issues / Fixed Bugs

| Bug | Fix | Commit |
|-----|-----|--------|
| Reward scale ×10 vs raw IoU | Changed to raw IoU / -1 | ceeb33d |
| Mesh normalization pred-only | Both pred+GT → [-1,1]³ | ceeb33d |
| bad_words_ids missing | Added to all generate() calls | ceeb33d |
| img eval unreliable in mixed batches | Use tools/eval_img.py | open |
| mine.py uses CadRecodeDataset (pkl) | Rewrote to use MeshDataset (raw STL) | current |
| mine.py R_th=7.5 (old ×10 scale) | Fixed to R_th=0.75 (raw IoU scale) | current |
