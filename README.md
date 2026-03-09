## CAD Reconstruction Research — NeurIPS 2026

Research codebase extending [cadrille (ICLR 2026)](https://arxiv.org/abs/2505.22914) with improved RL training infrastructure and richer reward signals for multi-modal CAD reconstruction.

---

### Baseline results (cadrille, for comparison)

| Method | DeepCAD IoU ↑ | Fusion360 IoU ↑ | CC3D IoU ↑ | Invalid ↓ |
|--------|:---:|:---:|:---:|:---:|
| CAD-Recode (ICCV 2025) | 0.721 | 0.663 | 0.357 | 2.3% |
| cadrille SFT | 0.756 | 0.674 | 0.368 | 1.8% |
| cadrille SFT + Dr. CPPO | **0.787** | **0.706** | **0.392** | **1.2%** |
| **Ours** | — | — | — | — |

---

### Overview

This repo extends the cadrille training pipeline with:

- **Modular RL package** (`rl/`) — clean separation of algorithm (`cppo.py`, `dpo.py`), dataset, evaluation, and reward; easy to swap reward signals and algorithms
- **Process-level rewards** (`rl/reward.py`) — IoU reward via isolated subprocess; Chamfer Distance computed alongside IoU for richer eval signal
- **Multi-GPU and Colab support** — configs for H100 80 GB, A100 40/80 GB, RTX 4080 16 GB; `colab.ipynb` for cloud training with Drive-backed checkpoints
- **Comprehensive validation** — per-step greedy eval on DeepCAD + Fusion360 test sets, both `pc` and `img` modalities, logged to W&B

The SFT backbone and model architecture are unchanged from cadrille (Qwen2-VL-2B + point cloud encoder). The RL training starts from the public cadrille SFT checkpoint.

---

### Quick Start — RL Training on a Remote VM

**Option A: Bare metal (recommended)**

```bash
git clone https://github.com/miachen0401/cadrille.git && cd cadrille
huggingface-cli login && wandb login

# Install deps + download checkpoint + all mesh data (~1.8 GB total)
# Downloads: cadrille-sft checkpoint, deepcad_train_mesh, deepcad_test_mesh, fusion360_test_mesh
bash scripts/setup.sh --data

# Train (pick config for your GPU — see table below)
python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1

# Resume after crash / session timeout
python rl/train.py --config configs/rl/h100.yaml \
    --run-name cadrille-rl-v1 \
    --checkpoint-path ./checkpoints/cadrille-rl-v1/checkpoint-5000
```

**Option B: Docker**

```bash
git clone https://github.com/miachen0401/cadrille.git && cd cadrille
docker build -t cadrille .

# Download data outside the container (mounted as volumes)
huggingface-cli login
bash scripts/setup.sh --data

# Train
docker run --gpus all --rm \
    -e WANDB_API_KEY=<your_wandb_key> \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v $(pwd):/workspace \
    cadrille \
    python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1
```

Config by GPU:

| VRAM | Config | G | Notes |
|------|--------|---|-------|
| ≥70 GB (H100 / A100 80G) | `configs/rl/h100.yaml` | 16 | Full paper hyperparameters |
| ~40 GB (A100 40G) | `configs/rl/a100.yaml` | 8 | Adam8bit |
| ~16 GB (RTX 4080 / 3090) | `configs/rl/4080.yaml` | 4 | Sequential generation |

For 8× GPU: replace `python` with `torchrun --nproc_per_node=8` and use `configs/rl/h100x8.yaml`.

---

### Installation

Dependencies are declared in `pyproject.toml` and managed with [uv](https://github.com/astral-sh/uv).
Three packages (`pytorch3d`, `cadquery`-from-git, `flash-attn`) need special build flags and are handled by `scripts/setup.sh` and the Dockerfiles.

**Docker** — see Quick Start above.

**Bare metal**
```bash
bash scripts/setup.sh           # deps only
bash scripts/setup.sh --data    # deps + download checkpoint + data
```

**Google Colab** — open `colab.ipynb`. Cells [1]–[7] set up the environment; cell [8] starts RL training. GPU is auto-detected (H100 / A100 40 GB / A100 80 GB).

---

### Data

All datasets are downloaded automatically by `bash scripts/setup.sh --data`.
Each is a single zip file on HuggingFace (avoids the 5000-file resolver rate limit).

| Dataset | HuggingFace | Size | Purpose |
|---------|-------------|------|---------|
| `deepcad_train_mesh` | `Hula0401/deepcad_train_mesh` | ~242 MB | RL training — DeepCAD (84k STLs) |
| `fusion360_train_mesh` | `Hula0401/fusion360_train_mesh` | ~TBD | RL training — Fusion360 (6.9k STLs) |
| `deepcad_test_mesh` | `Hula0401/deepCAD_test` | ~413 MB | Eval (8k STLs) |
| `fusion360_test_mesh` | `Hula0401/fusion360_test_mesh` | ~126 MB | Eval (1.7k STLs) |
| `cadrille-sft` checkpoint | `maksimko123/cadrille` | ~4.5 GB | RL starting point |

All STL meshes are pre-normalised to [0, 1]³. No preprocessing needed after download.

#### How `deepcad_train_mesh` was created

The DeepCAD dataset from Columbia University (~170k models in sketch-extrude JSON format)
was reconstructed to STL using `tools/deepcad2mesh.py` (OCC via cadquery-ocp):

```bash
wget http://www.cs.columbia.edu/cg/deepcad/data.tar -P data/
tar -xf data/data.tar -C data/          # → data/cad_json/{train,test,val}/
python tools/deepcad2mesh.py --split train --out data/deepcad_train_mesh --workers 16
```

**Challenges:** ~50% of models fail reconstruction (degenerate sketches, self-intersecting
extrusions, OCC Boolean failures). Conversion of 170k models takes ~6–12 hours on 16 cores.
The resulting 84k valid STLs are the ones we uploaded to HuggingFace — **no need to re-run
this on a fresh VM**.

#### SFT training dataset (optional — only for SFT from scratch)

```bash
# CAD-Recode v1.5 — ~100k CadQuery scripts + STL meshes
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/filapro/cad-recode-v1.5 data/cad-recode-v1.5
cd data/cad-recode-v1.5 && git lfs pull && cd ../..
python data/cadrecode2mesh.py   # convert .py → .stl (takes ~2 hours)
```

---

### SFT Training

Supervised fine-tuning on the CAD-Recode v1.5 dataset (~100k CadQuery scripts).
Starting from scratch or use the public [cadrille SFT checkpoint](https://huggingface.co/maksimko123/cadrille) directly.

```bash
# Single GPU (RTX 4080, 16 GB) — 12k steps, effective batch 28
bash scripts/run_sft.sh --config configs/sft/default.yaml

# Single H100 or multi-GPU — full 120k steps (matches cadrille paper)
bash scripts/run_sft.sh --config configs/sft/full.yaml

# Smoke test (600 steps, verifies setup in ~20 min)
bash scripts/run_sft.sh --config configs/sft/smoke.yaml
```

Key SFT hyperparameters:
```
optimizer:    AdamW  |  lr: 2e-4 (cosine)  |  warmup: 1000 steps
max_steps:    120,000  |  batch: 28  |  precision: bfloat16 + flash_attention_2
```

---

### RL Fine-tuning

Online RL (Dr. CPPO / GRPO) starting from the cadrille SFT checkpoint.
Reward: `r = -1` (invalid code) or `IoU ∈ [0, 1]` (valid geometry).
Both pred and GT meshes are normalised to [−1, 1]³ before IoU computation.

```bash
# RTX 4080 16 GB — G=4, Adam8bit, sequential generation
python rl/train.py --config configs/rl/4080.yaml

# A100 40 GB — G=8, Adam8bit
python rl/train.py --config configs/rl/a100.yaml

# H100 or A100 80 GB — G=16, full official hyperparameters
python rl/train.py --config configs/rl/h100.yaml

# Resume after crash / session timeout
python rl/train.py --config configs/rl/h100.yaml \
    --run-name cadrille-rl-v1 \
    --checkpoint-path ./checkpoints/cadrille-rl-v1/checkpoint-5000
```

Key RL hyperparameters (H100 config, matching cadrille paper):
```
algorithm:      Dr. CPPO / GRPO
optimizer:      Adam (Adam8bit on ≤ 40 GB GPU)
lr:             3e-5  |  G: 16 rollouts  |  top_N: 4  |  eps: 0.1
batch_updates:  3     |  max_new_tokens: 400
```

W&B metrics logged each step:
- `loss` — PPO clip loss
- `average_reward` — mean IoU reward across G rollouts
- `eval/pc/DeepCAD test/IoU mean` — greedy validation IoU
- `eval/pc/DeepCAD test/CD mean` — greedy validation Chamfer Distance
- `eval/pc/DeepCAD test/Failures fraction` — fraction of invalid completions

---

### Inference

```bash
python test.py \
    --checkpoint-path ./checkpoints/cadrille-rl-v1/checkpoint-final \
    --split deepcad_test_mesh \
    --mode pc \
    --py-path ./outputs/deepcad_pc
```

Supported modes: `pc` (point cloud), `img` (image), `pc_img` (both), `text`.

---

### Evaluation

```bash
# One-shot pipeline (generate + evaluate)
bash scripts/run_eval.sh \
    --checkpoint ./checkpoints/cadrille-rl-v1/checkpoint-final \
    --split deepcad_test_mesh \
    --mode pc_img

# Separately
python test.py --checkpoint-path $CKPT --split deepcad_test_mesh --mode pc --py-path ./outputs
python evaluate.py --py-path ./outputs
```

---

### Repository Structure

```
├── cadrille.py          Core model (Qwen2-VL-2B + point cloud encoder, unchanged from paper)
├── dataset.py           Dataset loaders (SFT)
├── train.py             SFT training (HuggingFace Trainer)
├── test.py              Inference / script generation
├── evaluate.py          Metrics: IoU, CD, invalidity rate
│
├── rl/
│   ├── train.py         Entry point — CLI + W&B init + model loading
│   ├── config.py        YAML loading and CLI/config merge
│   ├── dataset.py       MeshDataset, RLDataset, DPODataset
│   ├── eval.py          Greedy validation — IoU + CD per dataset/modality
│   ├── reward.py        IoU + CD reward via isolated subprocess
│   ├── mine.py          Hard example mining (optional)
│   └── algorithms/
│       ├── cppo.py      Dr. CPPO / GRPO implementation
│       └── dpo.py       DPO implementation
│
├── configs/
│   ├── sft/             SFT configs: default (12k), full (120k), h100, smoke
│   └── rl/              RL configs: 4080 (16 GB), a100 (40 GB), h100 (80 GB)
│
├── scripts/
│   ├── run_sft.sh       SFT launcher
│   ├── run_rl.sh        RL launcher
│   ├── run_eval.sh      Evaluation pipeline
│   └── setup.sh         Environment setup
│
├── tests/
│   └── test_cppo_step.py  Unit test for CPPO step (no mesh files needed)
│
└── colab.ipynb          Google Colab notebook (H100 / A100 40 GB / A100 80 GB)
```

---

### Pre-trained Models (cadrille baselines)

| Model | Description | HuggingFace |
|-------|-------------|-------------|
| `maksimko123/cadrille` | SFT on CAD-Recode v1.5 — starting point for RL | [🤗](https://huggingface.co/maksimko123/cadrille) |
| `maksimko123/cadrille-rl` | cadrille SFT + Dr. CPPO — baseline to beat | [🤗](https://huggingface.co/maksimko123/cadrille-rl) |

---

### Citation

If you use this codebase, please also cite the cadrille paper it builds on:

```bibtex
@inproceedings{kolodiazhnyi2026cadrille,
  title     = {cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning},
  author    = {Kolodiazhnyi, Maksim and Tarasov, Denis and Zhemchuzhnikov, Dmitrii and
               Nikulin, Alexander and Zisman, Ilya and Vorontsova, Anna and
               Konushin, Anton and Kurenkov, Vladislav and Rukhovich, Danila},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2505.22914}
}
```
