# BenchCAD SFT + RL

Two-stage training stack for **image / point-cloud → CadQuery code** generation,
built on Qwen2-VL-2B.

1. **SFT** — supervised fine-tune Qwen2-VL-2B on a 3-source CAD corpus
   (BenchCAD + cad-recode-20k + text2cad) to emit CadQuery code from
   pre-rendered images (or, optionally, point clouds).
2. **RL post-train** — CPPO / DPO with mesh-IoU reward on top of an SFT
   checkpoint. (Same stack as the cadrille paper, hardened for our data
   pipeline + eval.)

Online IoU + Failures eval runs every `eval_steps` against fixed subsets of
BenchCAD val / DeepCAD test / Fusion360 test, logged to W&B alongside
`eval_loss`. Checkpoints are pushed to a private HF model repo by a
non-blocking thread on every `save_steps`.

---

## Quick Start — From Zero on a Fresh A100/H100 VM

Five commands. Required tokens are documented in `.env.example`:
`HF_TOKEN` (read), `WANDB_API_KEY`.

```bash
# 1. Clone
git clone https://github.com/miachen0401/cadrille.git && cd cadrille

# 2. Credentials
cp .env.example .env && $EDITOR .env

# 3. Install (apt deps + uv venv + torch+cu124 + pytorch3d + cadquery
#    + flash-attn + Open3D source build with headless rendering).
#    See scripts/check_env/ for post-install smoke tests.
uv sync
uv run python data_prep/fetch_benchcad.py
uv run python data_prep/fetch_cad_sft.py
uv run python data_prep/prerender_dataset.py

# 4. Train
set -a; source .env; set +a
uv run python -m train.sft --config configs/sft/mix_bc4_r20k_t2c.yaml
```

For RL training, also fetch DeepCAD/Fusion360 test meshes (see
`data_prep/`) plus a reference SFT checkpoint, then run
`uv run python -m train.rl.train --config configs/rl/a100.yaml`.

**Resume / restart**: training configs default to
`resume_from_checkpoint: latest`, so re-running step 5 picks up the newest
`checkpoint-<N>/` in the run output dir.

---

## Repository layout

```
common/              # cross-cutting helpers (model, datasets, mesh I/O, metrics)
  model.py           # Cadrille (Qwen2-VL-2B + FourierPointEncoder) + collate
  datasets.py        # CadRecode / Text2CAD / BenchCad / CadRecode20k loaders
  meshio.py          # render_img + MeshDataset
  metrics.py         # compute_metrics / compute_reward / worker pools

train/
  sft/               # SFT training
    train.py           # entry  →  python -m train.sft
    online_eval.py     # IoU + Failures callback (mirrors RL eval wandb schema)
    hf_uploader.py     # background ckpt push to HF model repo
  rl/                # RL training (CPPO + DPO)
    train.py           # entry  →  python -m train.rl.train
    algorithms/{cppo,dpo}.py
    eval.py eval_passk.py mine.py filter_scores.py config.py dataset.py

eval/                # offline evaluation
  bench_sweep.py     # multi-dataset × multi-temp × max@N IoU sweep
  bench.py           # legacy single-temp BenchCAD eval
  passk.py           # pass@k estimator + CLI
  features.py        # BenchCAD feature_tags preservation
  pipeline.py runner.py render.py report.py
  others/            # cadrille paper-original evaluate.py + test.py

data_prep/           # one-time data materializers
  fetch_benchcad.py      # BenchCAD/cad_bench → STL + composite_png + pkl
  fetch_cad_sft.py       # Hula0401/cad-sft → cad-recode-20k + text2cad
  prerender_dataset.py   # mesh → 4-view _render.png
  rewrite_recode_to_bench.py + verify_recode_rewrite.py + push_bench_to_hf.py
  …

scripts/
  check_env/                   post-install env smoke (torch, open3d, cadquery, …)
  analysis/                    one-off diagnostics
    diversity_analysis.py        op-distribution GT vs pred
    diversity_benchcad_compare.py
    dataset_op_dist.py           cross-corpus op frequency
    plot_kl_quadrants.py mining_analysis.py training_dynamics.py …

bench/               # training-throughput bench (not model eval)
  bench_config.py bench_workers.py

experiments/         # off-main investigations
  cadevolve/  repair_lora/  data_prep_cadlib/

configs/sft/         # YAML — bench cad-only / mixes / from-warm-start
configs/rl/
configs/eval/

tests/test_refactor_safety.py   # 81 import + math tests, ~5 s
docs/                Dockerfile.official, sft_diagnostics_*.md, …
```

---

## Data sources

### Training

| corpus | source | local | size |
|---|---|---|---|
| BenchCAD | `BenchCAD/cad_bench` (HF, public) → `data_prep/fetch_benchcad.py` | `data/benchcad/{train,val}/` | 18,167 + 1,973 (`.py`+`.stl`+`_render.png`+metadata) |
| cad-recode-20k | `Hula0401/cad-sft/cad-recode-20k/*.parquet` → `data_prep/fetch_cad_sft.py` | `data/cad-recode-20k/{train,val}/` | 18,987 + 1,013 (`.py`+`_render.png`) |
| text2cad | `Hula0401/cad-sft/text2cad/*.{pkl,tar.gz}` → `data_prep/fetch_cad_sft.py` | `data/text2cad/{cadquery/,*.pkl}` | 76,238 train + 6,464 val + 8,035 test (`.py`+description, no images) |

### Evaluation

| corpus | source | local | size |
|---|---|---|---|
| BenchCAD val | local 90/10 hash split of `BenchCAD/cad_bench` | `data/benchcad/val/` | 1,973 |
| DeepCAD test | `Hula0401/deepCAD_test` | `data/deepcad_test_mesh/` | 8,046 (`.stl`+`_render.png`) |
| Fusion360 test | `Hula0401/fusion360_test_mesh` | `data/fusion360_test_mesh/` | 1,725 (`.stl`+`_render.png`) |

---

## Configs

| config | mix | bs × acc | max_steps | notes |
|---|---|---|---:|---|
| `configs/sft/mix_bc_r20k_t2c.yaml` | benchcad : recode20k : text2cad = 2 : 1 : 1 | 6 × 5 = 30 | 10,000 | Default SFT on the 3-source mix. |
| `configs/sft/mix_bc4_r20k_t2c.yaml` | 4 : 1 : 1 (benchcad-heavy) | 8 × 4 = 32 | 20,000 | Warm-starts from a prior checkpoint via `base_model:` to push op-level vocabulary (`hole/cut/chamfer/...`). |
| `configs/sft/mix_bc_r20k.yaml` | benchcad + recode20k 1 : 1 (no text2cad) | 4 × 2 | 4,000 | Faster, no long descriptions. |
| `configs/sft/benchcad_full.yaml` | benchcad only | 4 × 2 | 5,000 | Smoke on a single corpus. |
| `configs/rl/a100.yaml` | DeepCAD/Fusion360 hard-mined | per paper | 12k+ | RL post-train. |

---

## Eval

```bash
# Multi-temp, multi-sample sweep on a checkpoint:
uv run python -m eval.bench_sweep \
    --ckpt checkpoints/<run-name>/checkpoint-final \
    --datasets benchcad,deepcad,fusion360 \
    --temps 0,0.4,0.5,0.75,1.0,1.25 \
    --n-samples 16 --limit 50 --seed 42 \
    --modality img --batch-size 8 --score-workers 16 \
    --out eval_outputs/sweep_<tag>

# Pass@k:
uv run python -m eval.passk \
    --checkpoint checkpoints/<run-name>/checkpoint-final \
    --val-dir data/deepcad_test_mesh \
    --n-samples 5 --k-values 1,5

# Op-distribution diagnostic (no exec, regex-only):
uv run python -m scripts.analysis.diversity_analysis \
    --ckpt <ckpt> --n-items 30 --n-samples 8 --temps 0,0.5,1.0 \
    --out eval_outputs/diversity_<tag>
```

`scripts/analysis/diversity_benchcad_compare.py` and
`scripts/analysis/dataset_op_dist.py` widen op-distribution comparisons
to the full BenchCAD val (20k items) and across all three SFT corpora
respectively. See `docs/sft_diagnostics_*.md` for what they tend to show
on under-trained checkpoints.

---

## Tests

```
uv run pytest tests/test_refactor_safety.py
```

81 fast tests (~5 s): every package imports + passk math boundary cases.
`tests/test_iou.py / test_pipeline.py / test_cppo_step.py` cover
GPU+ckpt-bound paths and run after substantial training-loop changes.

---

## Acknowledgments

Built on [cadrille (ICLR 2026)](https://arxiv.org/abs/2505.22914) — Qwen2-VL-2B
backbone + point-cloud encoder + CPPO algorithm. Our additions are the data
pipeline (BenchCAD + cad-sft materializers, CadRecode20k loader, weighted +
length-grouped sampler), the eval tooling (multi-temp sweep, feature_recall,
op-distribution diagnostics), and the training infra (online IoU eval
callback, non-blocking HF ckpt uploader, auto-resume).
