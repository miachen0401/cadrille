# RL Training Plan — Cadrille Reproduction

## Goal
Reproduce Table 2 (img mode) from the Cadrille paper (ICLR 2026).
Target: DeepCAD IoU 92.2%, Fusion360 IoU 84.6% via Dr. CPPO on DeepCAD + Fusion360 train.

---

## Paper Target Metrics

| Model | RL | DeepCAD IoU | Fusion360 IoU |
|-------|----|-------------|---------------|
| cadrille SFT only | ✗ | 86.1% | 77.6% |
| cadrille Dr. CPPO (D-i+F-i) | ✅ | **92.2%** | **84.6%** |

Our SFT baseline: DeepCAD 86.4%, Fusion360 76.6%

---

## Data Pipeline

| Dataset | Location | Count | HuggingFace |
|---------|----------|-------|-------------|
| deepcad_train_mesh | `data/cadrille_training/deepcad` | 84,526 STLs | `Hula0401/deepcad_train_mesh` ✅ |
| fusion360_train_mesh | `data/cadrille_training/fusion360` | 30,820 STLs | `Hula0401/fusion360_train_mesh` ✅ |
| deepcad_test_mesh | `data/deepcad_test_mesh` | 8,047 STLs | `Hula0401/deepCAD_test` ✅ |
| fusion360_test_mesh | `data/fusion360_test_mesh` | 1,726 STLs | `Hula0401/fusion360_test_mesh` ✅ |
| cadrille-sft | `checkpoints/cadrille-sft` | — | `maksimko123/cadrille` ✅ |

---

## Current Pipeline Status

### Step 1 — IoU Distribution Analysis (TODO)
- [ ] Plot IoU histogram from `data/mined/deepcad_hard_scores.jsonl` + `fusion360_hard_scores.jsonl`
- [ ] Plot "hard example count vs R_th threshold" curve for both datasets
- [ ] Output: `work_dirs/mining_analysis/iou_distribution.png`

### Step 2 — Select Threshold & Build Training Dataset (TODO)
- [ ] Choose R_th based on Step 1 curve (paper uses 0.75; confirm or adjust)
- [ ] Re-filter via `rl/filter_scores.py` if threshold changes from current 0.75
- Current `combined_hard.pkl`: **6,861 examples** (4,173 DeepCAD + 2,688 Fusion360) at R_th=0.75
- Paper expected yield: ~12k DeepCAD + ~7k Fusion360 — ours is lower because we only scanned 20k+8k (not full 84k+30k)
- [ ] Upload final combined pkl + STL files to `Hula0401/cadrille_hard_examples`

### Step 3 — Temperature Sweep (TODO)
- [ ] Fix `tools/eval_temperature.py`: add `top_k=0, top_p=1.0` to override `generation_config`
- [ ] Run: n=100 from deepcad_test_mesh, K=4, temps [0.0, 0.15, 0.3, ..., 1.2], top_k=50
- [ ] Output markdown table: temp | best-IoU@4 | pass@4
- [ ] Select temperature with highest pass@4 → set `rollout_temperature` in `configs/rl/4080.yaml`

### Step 4 — Upload Dataset to HuggingFace (TODO)
- Pending decision: upload format (pkl only vs pkl + STL files)

### Step 5 — RL Training (TODO, after Steps 1-4)
- Config: `configs/rl/4080.yaml`
- Key settings: `hard_examples_pkl: ./data/mined/combined_hard.pkl`, `train_modality: img`
- `rollout_temperature`: TBD from Step 3
- Command: `python3 rl/train.py --config configs/rl/4080.yaml`
- Monitoring: reward trend, entropy (H > 0.1), eval IoU at step 200/400/600

---

## Mining Results (DONE ✅)

| Dataset | Scanned | Hard (R_th=0.75) | Hard rate | Scores file |
|---------|---------|-------------------|-----------|-------------|
| DeepCAD | 20,167 | 4,173 | 20.7% | `data/mined/deepcad_hard_scores.jsonl` |
| Fusion360 | 8,000 | 2,688 | 33.6% | `data/mined/fusion360_hard_scores.jsonl` |
| **Combined** | **28,167** | **6,861** | **24.4%** | `data/mined/combined_hard.pkl` |

Paper comparison: mined 20k/84k DeepCAD (24%) and 8k/30k Fusion360 (27%) — not full scan due to time.

---

## Key Architecture Notes

- `generation_config.json` in SFT checkpoint: `top_k` and `temperature` removed — cppo.py controls rollout sampling explicitly (`top_k=50, top_p=1.0, temperature=rollout_temperature`)
- `RLDataset` loads from pkl `{gt_mesh_path, file_name, is_pc}` — used when `hard_examples_pkl` set
- CPPO: degenerate groups (all rewards identical, std=0) → advantage=0, skip gradient → requires diverse rollouts (top_k≥10)
- img eval in mixed pc+img batches unreliable → use `tools/eval_img.py` for benchmarks
