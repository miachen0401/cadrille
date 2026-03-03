# Cadrille: SFT + RL Fine-Tuning Reproduction

## Code sources

| Repo | Branch | Content |
|------|--------|---------|
| `col14m/cadrille` | `master` | SFT: `train.py`, `cadrille.py`, `dataset.py`, `evaluate.py`, `test.py` |
| `col14m/cadrille` | `rl` | RL: `rl_finetune/train_cadrille_grpo.py`, `grpo_mm.py`, `cad_recode_model_mm.py` |
| `filapro/cad-recode` | `main` | Inference demo only — no training code released |

**The RL code IS publicly available** on the `rl` branch (contrary to the earlier paper draft).

---

## Stage 1: SFT

### Script
`train.py` — direct copy of official master branch with the following additions:

| Change | Official | Ours |
|--------|----------|------|
| `max_steps` | 120000 hardcoded | `--max-steps` CLI |
| `per_device_train_batch_size` | 15 hardcoded | **28** (accum=1, effective=28) |
| `gradient_accumulation_steps` | 2 hardcoded | **1** |
| `warmup_steps` | 1000 hardcoded | `min(1000, max_steps//10)` |
| `dataloader_num_workers` | 18 hardcoded | `--dataloader-workers 8` |
| `report_to` | None | `'wandb'` if `--wandb-project` |
| val split | required (crashes) | optional (skips eval if missing) |
| logging/save/eval steps | hardcoded | CLI flags |

### Official SFT hyperparameters (from code, not just paper)

```
optimizer:          AdamW (HF default)
learning_rate:      2e-4
lr_scheduler:       cosine → 0 at step 120k
warmup_steps:       1000
weight_decay:       0.01
max_steps:          120,000
batch (pc_img):     15 per-device × 2 accum = 30 effective   ← original
batch (pc_img):     28 per-device × 1 accum = 28 effective   ← our default
batch (use_text):   8 per-device  × 4 accum = 32 effective
precision:          bfloat16 + flash_attention_2
GPU:                single H100 (paper) / any single GPU (us)
```

Note: paper Appendix D says "batch=8, accum=4" — this refers to the `--use-text`
path only. The `pc_img` model (best result, Rpi) uses batch=15, accum=2.

### Dataset
cad-recode-v1.5: ~100k CadQuery `.py` files in `train/batch_00/` … `train/batch_10/`
No official val split; use `--val-batch batch_10` in `data/cadrecode2mesh.py`.

### Full training command
```bash
python train.py \
  --data-path ./data \
  --log-path ./work_dirs/cadrille-sft \
  --mode pc_img \
  --max-steps 120000 \
  --wandb-project cadrille-sft
```

### Expected time (120k steps)
| Hardware | Estimated |
|----------|-----------|
| Single H100 (paper) | ~17–33 hrs |
| Single A100 | ~30–50 hrs |
| Single RTX 4080 (bs=2) | ~10 days |

---

## Stage 2: RL Fine-Tuning (Dr. CPPO / GRPO)

### Script
`rl_train.py` — single-GPU adaptation of official `rl_finetune/train_cadrille_grpo.py`.

### Official hyperparameters (from rl branch)

| Param | Value | Source |
|-------|-------|--------|
| Algorithm | Dr. CPPO / GRPO | official code |
| GPUs | **8 H100s** via `torchrun --nproc-per-node=8` | official |
| Optimizer | **Adam** | official (not Adafactor) |
| Learning rate | **3e-5** | official |
| G (rollouts) | **16** | official `num_generations` |
| top_N | **4** | official `top_samples` |
| epsilon (high/low) | **0.1 / 0.1** | official `epsilon_high/low` |
| batch_updates | **3** | official `batch_updates` (PPO steps/rollout) |
| max_new_tokens | **400** | official `max_completion_length` |
| train_epochs | 20 | official |
| Training data | **real meshes** (deepcad_fusion_train) | official `RealDatasetMM` |
| Reward | -10 (invalid) or IoU×10 | official `combined_reward` |

### Algorithm per step (matches official grpo_mm.py)
```
1. Generate G=16 completions at temperature=1.0
2. Compute rewards R₁…G   →   r_invalid=-10 or r_IoU=IoU×10
3. Advantages Aᵍ = rᵍ − mean(r)
4. Select top_N=4 by |Aᵍ|
5. Compute π_old(τ|q) for selected samples (no grad)
6. Repeat batch_updates=3 times:
     new_logp = log π_θ(τ|q)   (with grad)
     ratio = exp(new_logp − old_logp)
     loss = −E[min(ratio·A, clip(ratio, 1-0.1, 1+0.1)·A)]
     Adam step (lr=3e-5)
7. Every K_update=10 steps: copy new → old policy
```

### Training data
Official trains on **real handcrafted meshes** from DeepCAD + Fusion360 train splits
(`data/deepcad_fusion_train`), not synthetic hard-mined examples.
The hard-mining R_th filtering is implicit: samples with uniform rewards (all high)
produce near-zero advantages and thus contribute ~0 gradient.

### Commands
```bash
# RL on real meshes (matches official)
python rl_train.py --mode cppo \
    --checkpoint-path maksimko123/cadrille \
    --data-dir ./data/deepcad_test_mesh \
    --output-dir ./work_dirs/cadrille-rl \
    --wandb-project cadrille-rl

# Legacy: from hard-mined pkl
python rl_train.py --mode cppo \
    --checkpoint-path ./checkpoints/cadrille \
    --hard-examples-pkl ./data/rl_hard_examples.pkl \
    --output-dir ./work_dirs/cadrille-rl
```

### Key differences: ours vs official
| | Official | Ours |
|-|----------|------|
| GPUs | 8 H100s (DDP) | 1 GPU (single-process) |
| Optimizer states | Adam shared across 8 GPUs | Adam on 1 GPU |
| Old policy | same device | CPU-offloaded |
| Batch per step | 16 samples × 8 GPUs = 128 | 1 sample × 1 GPU |

---

## Stage 1 smoke test results (200 steps, bs=2)

| Step | Train loss | Val loss |
|------|-----------|----------|
| 10 | 1.579 | — |
| 100 | 0.844 | 0.859 |
| 200 | 0.752 | 0.789 |

W&B: https://wandb.ai/hula-the-cat/cadrille-sft/runs/lfieikhy

---

## Reward signal

```
R(τ) = r_invalid + r_IoU
r_invalid = -10   (code fails to execute or produce valid geometry)
r_IoU     = IoU × 10   ∈ [0, 10]   (valid geometry)
```

CadQuery execution uses `subprocess.run()` (not `multiprocessing.Process`) to avoid
corrupting the CUDA context held by the training process.

---

## Stage 3: Visualization & Analysis (`viz/`)

Goal: understand training data structure, model failure modes, and complexity–quality tradeoffs.
These plots inform RL curriculum design, data augmentation decisions, and where to focus debugging.

### Directory layout

```
viz/
├── parse_cq.py          # Shared: regex feature extraction from CadQuery scripts
├── dataset_stats.py     # Training data distribution → plots/dataset_stats/
├── failure_analysis.py  # Eval failure modes → plots/failure_analysis/
└── plots/               # Output directory (gitignored)
    ├── dataset_stats/
    └── failure_analysis/
```

### `viz/parse_cq.py`

Regex-based feature extraction.  No heavy dependencies (only `re`, `pathlib`).

Per-script features extracted:
| Feature | Description |
|---------|-------------|
| `n_segments` | count of `.segment()` calls |
| `n_arcs` | count of `.arc()` calls |
| `n_splines` | count of `.spline()` calls |
| `n_circles`, `n_rects`, `n_polygons` | closed sketch shapes |
| `n_extrudes`, `n_revolves` | extrusion operations |
| `n_unions`, `n_cuts`, `n_intersects` | boolean operations |
| `n_cylinders`, `n_boxes`, `n_spheres` | primitives |
| `n_fillets`, `n_chamfers`, `n_shells` | detail modifiers |
| `n_push` | multi-region sketches (push/pop) |
| `n_workplanes` | number of `Workplane()` calls |
| `planes` | list of plane types used (XY/YZ/ZX) |
| `n_sketch_ops` | segments + arcs + splines + circles + rects + polygons |
| `n_bool_ops` | unions + cuts + intersects |
| `n_bodies` | n_bool_ops + 1 (approximate body count) |
| `code_length` | total character count |

### `viz/dataset_stats.py`

Reads: `./data/cad-recode-v1.5/train/batch_*/` (all training `.py` files)

Plots saved to `viz/plots/dataset_stats/`:
1. **Operation frequency** — horizontal bar, sorted; shows what the model has seen
2. **Code length distribution** — histogram; characterises typical script complexity
3. **Sketch ops per script** — histogram; distribution of geometric complexity
4. **Plane type distribution** — bar; which orientations dominate training
5. **Boolean ops per script** — histogram; number of bodies per shape
6. **Train vs val complexity** — overlaid histograms; checks for distribution shift

### `viz/failure_analysis.py`

Reads: `--eval-dir` (directory of generated `.py` files from `test.py`)
Optional: `--results-csv` (per-sample IoU/CD from `evaluate.py --results-csv`)

For each generated script:
1. Attempt subprocess execution with 10 s timeout (no CUDA risk)
2. Classify result into one of:
   - `success` — geometry produced
   - `syntax_error` — Python parse failed
   - `no_result` — code ran but `r` undefined
   - `attribute_error` — invalid CadQuery API call (hallucinated method)
   - `geometry_error` — OCC / trimesh failure (degenerate geometry)
   - `timeout` — execution exceeded timeout
   - `other_error` — any other exception

Plots saved to `viz/plots/failure_analysis/`:
1. **Failure type breakdown** — bar; overall error type distribution
2. **Code length: success vs failure** — overlaid KDE or histogram
3. **Sketch ops: success vs failure** — violin / box plots
4. **Failure rate by operation type** — horizontal bar; for each op, % of scripts
   containing that op that fail (reveals which operations the model struggles with)
5. **Train vs generated distribution** — side-by-side bars for top operations;
   shows distribution shift between training data and model outputs
6. **IoU vs code length** (if `--results-csv`) — scatter with regression line
7. **IoU by operation type** (if `--results-csv`) — box plot per op type

### `evaluate.py` extension

Add `--results-csv PATH` flag: after computing per-sample metrics, write a CSV with
columns `file_name, id, cd, iou` (one row per generated `.py`).  Used by
`failure_analysis.py` to correlate geometric quality with code features.

### Running the analyses

```bash
# 1. Training data distribution
python viz/dataset_stats.py --data-dir ./data/cad-recode-v1.5/train

# 2. Failure analysis on HF baseline eval
python viz/failure_analysis.py \
    --eval-dir ./work_dirs/eval_hf_baseline \
    --train-dir ./data/cad-recode-v1.5/train

# 3. With IoU scores (run evaluate.py first):
python evaluate.py \
    --pred-py-path ./work_dirs/eval_hf_baseline \
    --gt-mesh-path ./data/deepcad_test_mesh \
    --results-csv ./work_dirs/eval_hf_baseline/results.csv
python viz/failure_analysis.py \
    --eval-dir ./work_dirs/eval_hf_baseline \
    --results-csv ./work_dirs/eval_hf_baseline/results.csv \
    --train-dir ./data/cad-recode-v1.5/train
```

### Key questions these plots answer

| Question | Plot |
|----------|------|
| What operations does the model know? | Operation frequency (dataset_stats) |
| Does the model generate more/less complex code than training? | Train vs generated distribution |
| Which operation types cause the most failures? | Failure rate by operation type |
| Does longer code fail more? | Code length: success vs failure |
| Which error type dominates? | Failure type breakdown |
| Does IoU degrade with complexity? | IoU vs code length (with results-csv) |
