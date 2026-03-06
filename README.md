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

The fastest path to running RL on Lambda, RunPod, Vast.ai, or any bare GPU box:

```bash
# 1. Clone
git clone https://github.com/miachen0401/cadrille.git && cd cadrille

# 2. System libs (needed by CadQuery)
apt-get install -y libgl1 libglib2.0-0

# 3. Python deps
pip install uv
UV_SYSTEM_PYTHON=1 uv pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
UV_SYSTEM_PYTHON=1 uv pip install \
    transformers==4.50.3 accelerate==0.34.2 qwen-vl-utils==0.0.10 \
    trimesh==4.5.3 manifold3d open3d scipy==1.14.1 \
    wandb tqdm pyyaml cadquery-ocp==7.7.2 cadquery==2.4.0

# flash-attn prebuilt wheel (torch 2.5 + CUDA 12 + Python 3.12)
UV_SYSTEM_PYTHON=1 uv pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# pytorch3d — build once (~10 min), needed for point cloud encoder
MAX_JOBS=$(nproc) FORCE_CUDA=1 pip install \
    git+https://github.com/facebookresearch/pytorch3d.git

# 4. Download checkpoint + data (HF login avoids rate limits)
huggingface-cli login
huggingface-cli download maksimko123/cadrille \
    --repo-type model --local-dir ./checkpoints/cadrille-sft
huggingface-cli download maksimko123/deepcad_test_mesh \
    --repo-type dataset --local-dir ./data/deepcad_test_mesh
huggingface-cli download maksimko123/fusion360_test_mesh \
    --repo-type dataset --local-dir ./data/fusion360_test_mesh

# 5. Train
wandb login
CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1

# 6. Resume after crash
CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1 \
    --checkpoint-path ./checkpoints/cadrille-rl-v1/checkpoint-5000

# 7. 8× GPU (torchrun)
CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 rl/train.py \
    --config configs/rl/h100x8.yaml --run-name cadrille-rl-v1
```

Config by GPU:

| VRAM | Config |
|------|--------|
| ≥70 GB (H100 / A100 80G) | `configs/rl/h100.yaml` |
| ~40 GB (A100 40G) | `configs/rl/a100.yaml` |
| ~16 GB (RTX 4080 / 3090) | `configs/rl/4080.yaml` |

> `CUDA_LAUNCH_BLOCKING=1` synchronises CUDA ops to give accurate error tracebacks. Drop it once the run is confirmed stable.

---

### Installation

**Option 1: pip**
```bash
pip install -r requirements.txt
```

**Option 2: Docker** (recommended for reproducibility)
```bash
docker build -f Dockerfile -t cadrille-research .
docker run --gpus all -it cadrille-research bash
```

**Option 3: Google Colab**

Open `colab.ipynb`. Cells [1]–[7] set up the environment; cell [8] starts RL training.
GPU is auto-detected (H100 / A100 40 GB / A100 80 GB).

---

### Data

Download the mesh datasets from HuggingFace (each is a separate repo):

```bash
# DeepCAD test split — used as RL training prompts + evaluation
huggingface-cli download maksimko123/deepcad_test_mesh \
    --repo-type dataset --local-dir data/deepcad_test_mesh

# Fusion360 test split — cross-dataset validation
huggingface-cli download maksimko123/fusion360_test_mesh \
    --repo-type dataset --local-dir data/fusion360_test_mesh

# CAD-Recode v1.5 — for SFT training (~100k CadQuery scripts + STL meshes)
huggingface-cli download filapro/cad-recode-v1.5 \
    --repo-type dataset --local-dir data/cad-recode-v1.5
python data/cadrecode2mesh.py   # convert .py → .stl for the SFT dataset
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
Reward: `r = -10` (invalid) or `IoU × 10 ∈ [0, 10]` (valid geometry).

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
