# Cadrille — SFT + RL for image → CadQuery

Two-stage training stack for **image → CadQuery code** generation, built on
Qwen3-VL-2B with a swappable backbone mixin (Qwen2-VL / Qwen3-VL today;
Qwen2.5-VL adapter ready).

1. **SFT** — supervised fine-tune on a 5-source CAD corpus (~789k items
   after dedup + drop) to emit CadQuery from pre-rendered 4-view 268×268
   PNGs. Optional text-conditioning path for `text2cad-bench`.
2. **RL post-train** — CPPO / DPO with mesh-IoU reward on top of an SFT
   checkpoint (cadrille paper algorithm, hardened for our pipeline).

Online IoU + Failures eval runs every `eval_steps` against fixed subsets
of BenchCAD val / DeepCAD test / Fusion360 test. Greedy + max@8 (best of
8 candidates at t=1.0). Logged to W&B alongside `eval_loss`. Checkpoints
push to a public/private HF model repo by a non-blocking thread on every
`save_steps`.

> **Where the recipe lives**: the active SFT config is
> [`configs/sft/big_bench_shell_50k_v3.yaml`](configs/sft/big_bench_shell_50k_v3.yaml).
> Its header comments (lines 1-38) document mix design + per-source step
> share. Latest run learnings (data cleaning numbers, trajectory vs prior
> versions, op_loss interpretation, rendering pitfalls) are in
> [`docs/learnings_2026-04-29.md`](docs/learnings_2026-04-29.md).

---

## Quick Start — From Zero on a Fresh A100/H100 VM

Required tokens (in `.env`):
- `HF_TOKEN` — Hula0401 read+write (for cad-sft data + ckpt push)
- `BenchCAD_HF_TOKEN` — only if pushing back to `BenchCAD/*` upstream
- `WANDB_API_KEY`
- `DISCORD_WEBHOOK_URL` (optional) — for live eval-result posting

```bash
# 1. Clone
git clone https://github.com/miachen0401/cadrille.git
cd cadrille

# 2. Credentials
cp .env.example .env && $EDITOR .env

# 3. Install everything (apt deps, uv venv, torch+cu124, pytorch3d, cadquery,
#    flash-attn, Open3D source build w/ headless rendering, all training data)
bash scripts/setup.sh --data            # ~30-50 min wall-clock; idempotent

# 4. Reload PATH
source ~/.local/bin/env 2>/dev/null || source ~/.bashrc

# 5. Pre-flight check (catches bad pkl / missing PNGs / stale filter caches
#    BEFORE the GPU boots up — has saved several launches)
set -a && source .env && set +a
uv run python -m scripts.preflight_check \
    --config configs/sft/big_bench_shell_50k_v3.yaml

# 6. Train v3 (50 k steps, ~25 h on A100)
nohup uv run python -u -m train.sft \
    --config configs/sft/big_bench_shell_50k_v3.yaml \
    > logs/v3_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 7. (optional) Live Discord posting — fires on every eval landing
nohup bash scripts/watch_eval_post_discord.sh \
    "$(ls -t logs/v3_*.log | head -1)" \
    /ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-<timestamp> \
    > logs/watch.log 2>&1 &
```

For RL: `bash scripts/setup.sh --full` then `uv run python -m train.rl.train
--config configs/rl/a100.yaml`. RL is BLOCKED on SFT IoU ≥ 0.8 on
DeepCAD/Fusion360 (greedy) — current ckpt is at ~0.6 greedy / 0.68 max@8.

---

## Repository layout

```
common/              # cross-cutting helpers
  model.py           # Cadrille mixin (FourierPointEncoder + custom forward)
                     # works on any *VLForConditionalGeneration backbone
  datasets.py        # CadRecode / Text2CAD / BenchCad / CadRecode20k loaders
  meshio.py          # render_img + MeshDataset
  metrics.py         # compute_metrics / compute_reward / cq_reward_workers

train/
  sft/
    train.py         # entry  →  python -m train.sft --config <yaml>
    online_eval.py   # IoU + Failures + max@8 callback
    hf_uploader.py   # background ckpt push (non-blocking)
  rl/
    train.py algorithms/{cppo,dpo}.py  eval.py  mine.py  …

eval/
  bench_sweep.py     # multi-dataset × multi-temp × max@N IoU sweep
  passk.py           # pass@k estimator
  pipeline.py runner.py render.py report.py

data_prep/           # one-time data materializers (HF → local pkl + PNGs)
  fetch_cad_sft.py             # all 5 v3 SFT sources
  fetch_benchcad.py            # eval split
  prerender_dataset.py         # mesh → 4-view _render.png
  render_benchcad_easy.py      # cadquery → 4-view PNG, parallel + shard-aware
  merge_benchcad_easy_renders.py + upload_shards_to_hula0401.py

scripts/
  setup.sh                     # one-click installer (--data / --full)
  preflight_check.py           # MUST run before launching SFT
  watch_eval_post_discord.sh   # tail log → post IoU + collage to Discord
  analysis/                    # one-off diagnostics
    op_distribution_plot.py      # per-source op stats + plots
    source_grid_render.py        # 100-render grid per training source
    eval_to_discord.py           # full per-eval Discord post (history table,
                                 # comparison table, max@8, run lineage, etc)
    parse_cq.py                  # regex-based op extractor

bench/               # training-throughput benchmarks
experiments/         # off-main investigations (cadevolve, repair_lora, …)

configs/sft/         # YAML configs (one per recipe; v3 is current)
configs/rl/

tests/test_refactor_safety.py   # 81 import + math tests, ~5 s
docs/                # session reports, diagnostics, learnings_<date>.md
```

---

## Data sources

### v3 SFT training (~789k items after dedup + 80% drop on trivial families)

| source | HF | local | filtered count | mode |
|---|---|---|---:|---|
| benchcad | `Hula0401/cad-sft/benchcad/` | `data/benchcad/` | 11,443 | image |
| cad_iso_106 | `Hula0401/cad-sft/cad-iso-106-175k/` | `data/cad-iso-106/` | 122,483 | image |
| benchcad_simple | `Hula0401/cad-sft/benchcad-simple/` | `data/benchcad-simple/` | 76,671 | image |
| text2cad_bench (img) | `Hula0401/cad-sft/text2cad-bench/` | `data/text2cad-bench/` | 53,339 | image |
| text2cad_bench (text) | (same) | (same) | 53,339 | text |
| cad_recode_bench | `Hula0401/cad-sft/cad-recode-bench/` | `data/cad-recode-bench/` | 472,244 | image |

**Available but not yet wired into v3**:
`Hula0401/cad-sft/benchcad-easy/` (109k rows, 55 shards, all render_img filled).
Likely the next "v4" addition — see learnings doc §8.

### Evaluation (held-out)

| corpus | source | local | size |
|---|---|---|---:|
| BenchCAD val | local 90/10 hash split | `data/benchcad/val/` | 1,973 |
| DeepCAD test | `Hula0401/deepCAD_test` | `data/deepcad_test_mesh/` | 8,046 |
| Fusion360 test | `Hula0401/fusion360_test_mesh` | `data/fusion360_test_mesh/` | 1,725 |

---

## Configs

| config | mix / weights | total weight | bs × acc | max_steps | notes |
|---|---|---:|---|---:|---|
| **`configs/sft/big_bench_shell_50k_v3.yaml`** ← active | bench 11 / iso 122 / simple 77 / t2c-img 29 / t2c-text 29 / recode-bench 257 | 525 | 8 × 4 = 32 | 50,000 | **Current production recipe.** Qwen3-VL-2B from scratch, 60% HQ / 40% bench-stack, equal-per-item within group, n=50 eval, max@8 every 2nd eval. ETA ~25 h on A100. |
| `configs/sft/big_bench_shell_50k_phase2.yaml` | v2: bench 18 / iso 162 / simple 86 / t2c-bench 94 / recode-bench 175 | 535 | same | 50,000 | Predecessor (no dedup, no 80% drop). Best max@8 0.644/0.650/0.666 @ 27-29k. |
| `configs/sft/curriculum_qwen3vl_2b.yaml` | 3-phase curriculum | varies | 8 × 4 | 20,000 | Plateaued early; phase-3 (8:1:1) hurt DC. |
| `configs/sft/qwen3vl_2b_recode_30k_clean.yaml` | recode-only (single source) | – | 8 × 4 | 30,000 | Smoke / single-source baseline. |
| `configs/sft/smoke.yaml` | minimal | – | 1 × 1 | 100 | CI-style end-to-end check. |
| `configs/rl/a100.yaml` | DeepCAD/Fusion360 hard-mined | per paper | – | 12k+ | RL post-train (BLOCKED on SFT ≥ 0.8 greedy). |

> All historical configs are kept for reference. See
> `docs/learnings_2026-04-29.md` for the rationale of each version's mix
> change.

---

## Eval

```bash
# Multi-temp, multi-sample sweep on a checkpoint
uv run python -m eval.bench_sweep \
    --ckpt /ephemeral/checkpoints/<run>/checkpoint-final \
    --base-model Qwen/Qwen3-VL-2B-Instruct \
    --backbone qwen3_vl \
    --datasets benchcad,deepcad,fusion360 \
    --temps 0,0.4,0.75,1.0 \
    --n-samples 8 --limit 50 --seed 42 \
    --modality img --batch-size 8 --score-workers 16 \
    --out eval_outputs/sweep_<tag>

# Pass@k
uv run python -m eval.passk \
    --checkpoint <ckpt> \
    --val-dir data/deepcad_test_mesh \
    --n-samples 5 --k-values 1,5

# Per-source op-distribution diagnostic (no GPU)
uv run python -m scripts.analysis.op_distribution_plot
```

Online (during training) eval is automatic; results post to W&B and (if
configured) Discord. Each eval emits per-bucket greedy IoU + exec rate +
op_loss + rare_op_recall; every 2nd eval also runs max@8.

---

## Tests

```bash
uv run pytest tests/test_refactor_safety.py    # 81 fast tests, ~5 s
```

`tests/test_iou.py / test_pipeline.py / test_cppo_step.py` cover
GPU+ckpt-bound paths and should run after substantial training-loop changes.

---

## Acknowledgments

Built on [cadrille (ICLR 2026)](https://arxiv.org/abs/2505.22914) — base
VL backbone + FourierPointEncoder + CPPO algorithm. Our additions:

- 5-source SFT data pipeline with cleaning (dedup + family-drop)
- Backbone-agnostic Cadrille mixin (Qwen2 / Qwen3 / Qwen2.5)
- `online_eval.py` IoU + max@8 callback
- Non-blocking HF ckpt uploader
- Discord live-eval poster with comparison + history tables
- BenchCAD/benchcad-easy renderer (cadquery → 4-view 268×268)
- Multi-VM render orchestration (shard-aware, family-shuffle, looser
  tessellation tolerance)

License: see upstream cadrille.
