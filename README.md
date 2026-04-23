## BenchCAD SFT — Multi-modal CAD Reconstruction

Supervised fine-tuning pipeline for Qwen2-VL-2B to emit [CadQuery](https://github.com/CadQuery/cadquery)
code from an image, a point cloud, or a natural-language description. Built on top of the
[cadrille (ICLR 2026)](https://arxiv.org/abs/2505.22914) backbone.

This branch (`benchcad-sft`) ships **only the SFT + evaluation pipeline** — data prep
tools, training entry point, evaluation runners, analysis scripts. Training is
deterministic given a fixed seed (`42`) so any run reproduces byte-for-byte.

---

### What's in this repo

```
cadrille.py              Model — Qwen2-VL-2B + point-cloud encoder (unchanged from cadrille paper)
dataset.py               CadRecodeDataset + Text2CADDataset, with len-filter cache
train.py                 SFT entry point (HuggingFace Trainer, torchrun-compatible)
test.py                  Inference — generate CadQuery scripts from a checkpoint
evaluate.py              Metrics — IoU, Chamfer Distance, invalidity rate

configs/
  sft/                   SFT configs: default (12k, 4080 16GB), full (120k),
                         a100, h100 (8-GPU), smoke (600-step sanity), mix_1_2_2 (3-source)
  eval/                  Eval configs: quick, full, compare, test

eval/                    Eval runner package: pipeline, pass@k, render, report
scripts/                 One-liner launchers: setup, run_sft, run_eval, pack_datasets
tools/                   Data prep + analysis (see tools/README.md)
tests/                   IoU + pipeline unit tests
```

---

### Installation

Dependencies are in `pyproject.toml` and managed with [uv](https://github.com/astral-sh/uv).
Three packages (`pytorch3d`, `cadquery` from git, `flash-attn`) need special build flags,
handled by `scripts/setup.sh`.

```bash
git clone https://github.com/miachen0401/cadrille.git && cd cadrille
git checkout benchcad-sft

bash scripts/setup.sh           # deps only
bash scripts/setup.sh --data    # deps + SFT checkpoint + eval meshes (~2 GB)

cp .env.example .env            # fill in HF_TOKEN and WANDB_API_KEY
source .env
```

**Docker** — `docker build -t cadrille .` (see `Dockerfile`).

---

### SFT training corpus

A 3-source code mixture:

| source | rows (len ≤ 1000) | input | notes |
|---|---|---|---|
| **recode** — [filapro/cad-recode-v1.5](https://huggingface.co/datasets/filapro/cad-recode-v1.5) | ~838k train + 822 val | rendered mesh → img/pc | AST-rewritten to bench-shell style (see `tools/rewrite_recode_to_bench.py`) |
| **text2cad** — cadquery subset | ~66k train + 5.9k val | description (text) | natural-language caption → code |
| **benchcad** — [BenchCAD/cad_bench](https://huggingface.co/datasets/BenchCAD/cad_bench) | ~15k | `composite_png` | synthetic benchmark with rich metadata (family / difficulty / ops) |

**Mixing ratio (metadata):** `text2cad : recode : benchcad = 1 : 2 : 2`
(recorded in `configs/sft/mix_1_2_2.yaml`; weighted-sampler enforcement is TODO —
current `ConcatDataset` samples uniformly across concatenated rows).

**HuggingFace archive:** [BenchCAD/cad_sft_training](https://huggingface.co/datasets/BenchCAD/cad_sft_training)
holds the rewritten code corpus in bench-shell parquet. The local training path reads
from `data/cad-recode-v1.5/` + `data/text2cad/` where images/meshes live; the HF
dataset is a text-only archive for reproducibility and downstream reuse.

---

### Running SFT

```bash
# Single RTX 4080 (16 GB) — 12k steps, effective batch 30
bash scripts/run_sft.sh --config configs/sft/default.yaml

# 3-source mix with 1:2:2 metadata + len ≤ 1000 filter
bash scripts/run_sft.sh --config configs/sft/mix_1_2_2.yaml

# Single A100 80 GB — paper hyperparameters, 120k steps
bash scripts/run_sft.sh --config configs/sft/a100.yaml

# 8× H100 — multi-GPU (torchrun)
bash scripts/run_sft.sh --config configs/sft/h100.yaml

# Full 120k-step production run (any GPU that fits)
bash scripts/run_sft.sh --config configs/sft/full.yaml

# Smoke test (600 steps, verifies setup in ~20 min on a 4080)
bash scripts/run_sft.sh --config configs/sft/smoke.yaml
```

Key hyperparameters (match the cadrille paper):

```
optimizer:    AdamW  |  lr: 2e-4 (cosine)  |  warmup: 1000 steps
max_steps:    120,000  |  eff. batch: 30  |  precision: bfloat16 + flash_attention_2
seed:         42 (dataloader + Trainer init)  |  max_code_len: 1000 chars
```

**Reproducibility.** Every SFT config pins `seed: 42` and `max_code_len: 1000`.
`Trainer` consumes both `seed` and `data_seed`, so dataloader shuffle order and
model init are deterministic. The `max_code_len` filter caches the kept-index
list under `{data_root}/.filter_cache/` so repeat runs skip the full scan.

Training metrics are logged to [Weights & Biases](https://wandb.ai/) under project
`cadrille-sft`. Set `WANDB_API_KEY` in `.env`.

---

### Inference

```bash
python test.py \
    --checkpoint-path ./checkpoints/cadrille-sft/checkpoint-final \
    --split deepcad_test_mesh \
    --mode pc \
    --py-path ./outputs/deepcad_pc
```

Supported modes: `pc` (point cloud), `img` (image), `pc_img` (both), `text`.

---

### Evaluation

```bash
# One-shot — generate + compute metrics
bash scripts/run_eval.sh \
    --checkpoint ./checkpoints/cadrille-sft/checkpoint-final \
    --split deepcad_test_mesh \
    --mode pc_img

# Two-step (same thing)
python test.py --checkpoint-path $CKPT --split deepcad_test_mesh --mode pc --py-path ./outputs
python evaluate.py --py-path ./outputs

# Full multi-checkpoint eval with pass@k
python -m eval.runner configs/eval/full.yaml
python -m eval.runner configs/eval/compare.yaml --out eval_outputs/my_compare
```

Metrics: IoU (volumetric intersection-over-union), CD (Chamfer Distance), invalid-code
fraction. Both pred and GT meshes are normalised to `[−1, 1]³` before computation.

---

### Evaluation data

Downloaded automatically by `bash scripts/setup.sh --data`. Each dataset is a single
zip on HuggingFace (avoids the 5000-file resolver rate limit).

| Dataset | HuggingFace | Size | Purpose |
|---------|-------------|------|---------|
| `deepcad_test_mesh` | `Hula0401/deepCAD_test` | ~413 MB | Eval (8k STLs) |
| `fusion360_test_mesh` | `Hula0401/fusion360_test_mesh` | ~126 MB | Eval (1.7k STLs) |
| `cadrille-sft` checkpoint | `maksimko123/cadrille` | ~4.5 GB | Paper baseline |

All STL meshes are pre-normalised to `[0, 1]³` — no preprocessing after download.

#### How `deepcad_train_mesh` was created

The DeepCAD dataset from Columbia University (~170k models in sketch-extrude JSON)
was reconstructed to STL using `tools/deepcad2mesh.py` (OCC via cadquery-ocp):

```bash
wget http://www.cs.columbia.edu/cg/deepcad/data.tar -P data/
tar -xf data/data.tar -C data/          # → data/cad_json/{train,test,val}/
python tools/deepcad2mesh.py --split train --out data/deepcad_train_mesh --workers 16
```

~50% of DeepCAD models fail reconstruction (degenerate sketches, self-intersecting
extrusions, OCC boolean failures). The 84k valid STLs are uploaded to HF; **no need
to re-run** on a fresh VM.

---

### Rebuilding / uploading the code corpus

The HF archive was built with three tools. Only rerun these if you regenerate the
corpus from a different source snapshot.

```bash
# 1. Rewrite filapro/cad-recode-v1.5 .py files into bench-shell style
uv run python tools/rewrite_recode_to_bench.py \
    --input data/cad-recode-v1.5/train \
    --output data/cad-recode-v1.5-bench/train --jobs 8

# 2. Verify the rewrite preserves geometry (mesh IoU)
uv run python tools/verify_recode_rewrite.py \
    --original data/cad-recode-v1.5/train \
    --rewritten data/cad-recode-v1.5-bench/train \
    --n 1000 --jobs 8

# 3. Package + push to HF (needs BenchCAD_HF_TOKEN in env)
uv run python tools/push_bench_to_hf.py --repo BenchCAD/cad_sft_training --private
```

Verification on the current snapshot: **10,000 / 10,000** recode pairs pass
(`vol_rel = 0`, IoU mean = 1.000000, min = 0.999999); **30 / 30** text2cad pairs
pass. The rewrite is pure AST reformatting — no numerical changes.

---

### Pre-trained Models

| Model | Description | HuggingFace |
|-------|-------------|-------------|
| `maksimko123/cadrille` | SFT on CAD-Recode v1.5 — paper baseline | [🤗](https://huggingface.co/maksimko123/cadrille) |

---

### Citation

If you use this codebase, please cite the cadrille paper it builds on:

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
