## `cadrille`: Multi-modal CAD Reconstruction with Online Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2505.22914-b31b1b.svg)](https://arxiv.org/abs/2505.22914)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc)
[![HuggingFace SFT](https://img.shields.io/badge/🤗%20HuggingFace-SFT-yellow)](https://huggingface.co/maksimko123/cadrille)
[![HuggingFace RL](https://img.shields.io/badge/🤗%20HuggingFace-RL-orange)](https://huggingface.co/maksimko123/cadrille-rl)

**News**:
- 🔥 Jan, 2026. `cadrille` is accepted to ICLR 2026.
- 🔥 May, 2025. `cadrille` is state-of-the-art in three CAD reconstruction benchmarks: DeepCAD, Fusion360, CC3D.

This repository contains the implementation of `cadrille`, a multi-modal (point clouds / images / text) 3D CAD reconstruction method introduced in our paper:

> **cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning**<br>
> [Maksim Kolodiazhnyi](https://github.com/col14m),
> [Denis Tarasov](https://dt6a.github.io),
> [Dmitrii Zhemchuzhnikov](https://github.com/zhemdi),
> [Alexander Nikulin](https://howuhh.github.io),
> [Ilya Zisman](https://zis.mn),
> [Anna Vorontsova](https://highrut.github.io),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ),
> [Vladislav Kurenkov](https://dunnolab.ai),
> [Danila Rukhovich](https://github.com/filaPro) <br>
> https://arxiv.org/abs/2505.22914

---

### Overview

`cadrille` reconstructs 3D CAD models as executable CadQuery Python scripts from point clouds, images, or text descriptions. It combines supervised fine-tuning (SFT) of a 2B vision-language model with online reinforcement learning (Dr. CPPO / GRPO) guided by volumetric IoU rewards.

**Results on DeepCAD test split (point cloud input):**

| Method | IoU ↑ | CD ↓ | Invalid ↓ |
|--------|-------|------|----------|
| CAD-Recode | 0.721 | — | 2.3% |
| cadrille (SFT) | 0.756 | 0.0089 | 1.8% |
| **cadrille (SFT+RL)** | **0.787** | **0.0071** | **1.2%** |

---

### Installation

**Option 1: pip**
```bash
pip install -r requirements.txt
```

**Option 2: Docker** (recommended for reproducibility)
```bash
docker build -f Dockerfile -t cadrille .
docker run --gpus all -it cadrille bash
```

We support DeepCAD (test), Fusion360 (test), Text2CAD (train / val / test), and CAD-Recode (train, val) datasets. Follow the [data README](data/README.md) to download and preprocess data.

---

### SFT Training

Supervised fine-tuning on the CAD-Recode v1.5 dataset (~100k CadQuery scripts):

```bash
# Single GPU (RTX 4080, 16 GB) — 12k steps, effective batch=28
bash scripts/run_sft.sh --config configs/sft/default.yaml

# Single H100 or multi-GPU — full 120k steps
bash scripts/run_sft.sh --config configs/sft/full.yaml

# 8× H100 (matches paper exactly)
bash scripts/run_sft.sh --config configs/sft/h100.yaml

# Smoke test (600 steps, verifies setup)
bash scripts/run_sft.sh --config configs/sft/smoke.yaml
```

Training progress is logged to [Weights & Biases](https://wandb.ai). Set `wandb_project` in the config or pass `--wandb-project cadrille-sft`.

**Key SFT hyperparameters** (from `configs/sft/full.yaml`):
```
optimizer:    AdamW
lr:           2e-4  (cosine schedule)
warmup:       1000 steps
max_steps:    120,000
batch:        28 per-device × 1 accum = 28 effective
precision:    bfloat16 + flash_attention_2
```

---

### RL Fine-tuning

Online RL using Dr. CPPO (GRPO variant) with IoU rewards. Starts from your best SFT checkpoint:

```bash
# Single RTX 4080 (16 GB) — reduced G=4, Adam8bit optimizer
bash scripts/run_rl.sh --config configs/rl/4080.yaml \
    --checkpoint-path ./checkpoints/cadrille-sft/checkpoint-final

# 8× H100 — official G=16, full hyperparameters, independent shards
DISTRIBUTED=1 bash scripts/run_rl.sh --config configs/rl/default.yaml
```

**W&B dashboard** will show (matching official cadrille naming):
- `loss` — PPO clip loss
- `average_reward` — mean IoU reward across G rollouts
- `eval/pc/DeepCAD test/IoU mean` — validation IoU
- `eval/pc/DeepCAD test/CD mean` — validation Chamfer Distance
- `eval/pc/DeepCAD test/Failures fraction` — fraction of invalid completions

**Key RL hyperparameters** (from `configs/rl/default.yaml`):
```
algorithm:    Dr. CPPO / GRPO
optimizer:    Adam (Adam8bit on single GPU)
lr:           3e-5
G:            16 rollouts/step  (4 on RTX 4080)
top_N:        4 selected by |advantage|
eps:          0.1 (PPO clip)
batch_updates: 3
max_new_tokens: 400
reward:       r = -10 (invalid) or IoU × 10 ∈ [0, 10]
```

---

### Inference

Generate CadQuery scripts for a test split:

```bash
python test.py \
    --checkpoint-path maksimko123/cadrille-rl \
    --split deepcad_test_mesh \
    --mode pc \
    --py-path ./outputs/deepcad_pc
```

Supported modes: `pc` (point cloud), `img` (image), `pc_img` (both), `text`.

---

### Evaluation

Compute IoU, Chamfer Distance, and invalidity ratio:

```bash
# One-shot pipeline (generate + evaluate)
bash scripts/run_eval.sh \
    --checkpoint ./checkpoints/cadrille-rl/checkpoint-final \
    --split deepcad_test_mesh \
    --mode pc_img

# Or separately
python test.py --checkpoint-path $CKPT --split deepcad_test_mesh --mode pc --py-path ./outputs
python evaluate.py --py-path ./outputs
```

---

### Pre-trained Models

| Model | Description | HuggingFace |
|-------|-------------|-------------|
| `maksimko123/cadrille` | SFT on CAD-Recode v1.5 | [🤗 link](https://huggingface.co/maksimko123/cadrille) |
| `maksimko123/cadrille-rl` | SFT + RL fine-tuning | [🤗 link](https://huggingface.co/maksimko123/cadrille-rl) |

---

### Repository Structure

```
cadrille/
├── cadrille.py          Core model (Qwen2-VL + point cloud encoder)
├── dataset.py           Dataset loaders
├── train.py             SFT training (HuggingFace Trainer)
├── test.py              Inference / script generation
├── evaluate.py          Metrics: IoU, CD, invalidity
├── rl/
│   ├── train.py         RL training (Dr. CPPO / GRPO)
│   ├── reward.py        IoU + CD reward via subprocess
│   └── mine.py          Hard example mining (optional pre-filter)
├── configs/
│   ├── sft/             SFT configs (default, full, h100, smoke)
│   └── rl/              RL configs (default, 4080)
└── scripts/
    ├── run_sft.sh       SFT launcher
    ├── run_rl.sh        RL launcher
    ├── run_eval.sh      Evaluation pipeline
    └── setup.sh         Environment setup
```

---

### Citation

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

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b811b14-e646-48d6-9a0c-06a9655bdbaf" alt="cadrille scheme"/>
</p>
