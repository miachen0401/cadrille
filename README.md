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

### v3 SFT training overview (~789k items after dedup + 80% drop)

| source | local | filtered | mode | step share | per-item epochs over 50k |
|---|---|---:|---|---:|---:|
| benchcad | `data/benchcad/` | 11,443 | image | 2.1% | 3.0 |
| cad_iso_106 | `data/cad-iso-106/` | 122,483 | image | 23.2% | 3.0 |
| benchcad_simple | `data/benchcad-simple/` | 76,671 | image | 14.7% | 3.0 |
| text2cad_bench (img) | `data/text2cad-bench/` | 53,339 | image | 5.5% | 1.65 |
| text2cad_bench (text) | (same) | 53,339 | text | 5.5% | 1.65 |
| cad_recode_bench | `data/cad-recode-bench/` | 472,244 | image | 49.0% | 1.65 |

**Mix design**: 60% HQ (text2cad_bench×2 + recode_bench), 40% bench-stack
(benchcad + iso + simple). Equal-per-item *within* group, so every item in
the bench-stack gets ~3× the exposure of every item in the HQ group.

### Per-source detail (with pros / cons / op stats)

Op stats from n=500 sample per source, seed=42 (regenerate with
`scripts/analysis/op_distribution_plot.py`).

#### `benchcad` — 11,443 items
- **HF**: `Hula0401/cad-sft/benchcad/`
- **Origin**: `BenchCAD/cad_bench` upstream → cleaned (37% dupes removed) +
  90/10 train/val hash split.
- **Family**: `simple_*` synthetic CAD families (block, bracket, channel,
  bar, plate, hole, ...). Every code uses 3-8 cadquery ops out of a small
  closed vocabulary (~30 distinct ops total).
- **Ops/case**: median 5, mean 4.89, p95 7.
- **✅ Pros**:
  - Clean cadquery surface, easy to reproduce mesh from code → high IoU
    achievable.
  - Anchors **BenchCAD val** (the eval bucket teammates report on most).
  - High per-item exposure in v3 (3.0 epochs over 50k) — small but
    weight-rich.
  - In v2 phase2b, bumping benchcad weight 18→50× lifted BC val greedy
    from 0.43 → 0.59. **Highest-leverage source for BC val.**
- **❌ Cons**:
  - Tiny vocabulary → ceiling on diversity. Cannot teach new ops.
  - Many families are visually similar (e.g. plate-with-hole variants);
    real generalization may be lower than IoU implies.
  - Can over-fit if weighted too high (curriculum's 8:1:1 phase regressed
    DC by training too benchcad-heavy).

#### `cad_iso_106` — 122,483 items
- **HF**: `Hula0401/cad-sft/cad-iso-106-175k/`
- **Origin**: BenchCAD `cad_iso_106` family — industrial-parts catalog
  (ISO standard parts, brackets, fasteners, gears, pulleys, ...).
- **Ops/case**: median 5, mean 4.81, p95 7.
- **✅ Pros**:
  - Only source with non-trivial **`fillet`** coverage (~19% of items have
    fillet; benchcad has <2%). Critical for the `fillet/chamfer/shell`
    rare-op recall target.
  - Real-world part shapes, not synthetic geometry — closer to DeepCAD/
    Fusion360 distribution than benchcad.
  - 122k items at 3 epochs gives wide rare-op coverage.
- **❌ Cons**:
  - Family-clustered codes — workers can stall if not shuffled (fixed in
    `data_prep/render_benchcad_easy.py`'s shuffle pre-step).
  - Some industrial parts have very fine tessellation tolerances → slow
    render (we use loose `tessellate(0.01, 0.5)` for thumbnails to
    compensate).
  - Op vocabulary leans toward common ops; not a long-tail source on its
    own.

#### `benchcad_simple` — 76,671 items
- **HF**: `Hula0401/cad-sft/benchcad-simple/`
- **Origin**: BenchCAD `benchcad_simple` — even simpler than benchcad,
  ~3-5 ops per code (extrude + workplane + 1-2 sketch primitives).
- **Ops/case**: median 4, mean 3.79, p95 5.
- **✅ Pros**:
  - Vocabulary breadth (lots of `workplane` placements, base_planes).
  - Fast to render, codes always exec cleanly (high quality signal).
  - Helps the model learn the "minimum viable" cadquery skeleton.
- **❌ Cons**:
  - **Lowest op-count source** — risks model collapsing to "always emit
    extrude+1 sketch" if weighted too high.
  - 12% dupes pre-clean (now removed).
  - No rare ops at all; do not rely on this for fillet/chamfer/shell signal.

#### `text2cad_bench` — 53,339 items × 2 modes
- **HF**: `Hula0401/cad-sft/text2cad-bench/`
- **Origin**: filapro/text2cad benchmark, re-rendered for our 4-view
  268×268 PNG format.
- **Ops/case**: median 4, mean 3.63, p95 6.
- **Two modalities, treated as separate v3 sources** with separate weights:
  - **`text2cad_bench_img`**: image (4-view PNG) → cadquery code
  - **`text2cad_bench_text`**: natural-language description → cadquery code
  - Per training step, exactly ONE modality is sampled per item
    (never img+text mixed on same sample → avoids encoder confusion).
- **✅ Pros**:
  - Only source with **natural-language descriptions** — gives the LLM a
    text-conditioned path that complements the visual path.
  - Diverse code styles (different humans wrote the originals).
- **❌ Cons**:
  - 29% dupes pre-clean (now removed).
  - 38 codes silently failed render in the upstream parquet (caught by
    pre-flight check, fixed). **Always run pre-flight before training.**
  - Text descriptions vary in quality (some are auto-generated, terse).
  - `text2cad_legacy` (different older corpus, 76k items) was deleted —
    measured 28% trivial codes, was dragging eval.

#### `cad_recode_bench` — 472,244 items (the workhorse)
- **HF**: `Hula0401/cad-sft/cad-recode-bench/`
- **Origin**: synthetically generated from filapro/cad-recode-v1.5 base,
  then re-rendered for our 268×268 4-view format. **49% of v3 step share.**
- **Ops/case**: median 5, mean 5.16, p95 7. (Same per-case op count as
  benchcad — surprisingly compact.)
- **✅ Pros**:
  - **Largest source by ~4×** — primary diversity driver.
  - Wide global op vocabulary (100+ distinct ops across the corpus). The
    long-tail breadth is what the rare-op recall metric actually measures.
  - Catches the model up on op-vocabulary that BC-family sources don't.
- **❌ Cons**:
  - At 49% step share with 1.65 epochs/item, individual rare items get
    seen <2× — model may mode-collapse on the long-tail at eval time
    (this drives the recode20k op_loss=0.77 vs benchcad=0.20 gap).
  - Some synth codes have unusual control-flow that doesn't render cleanly
    (~0.2% timeout / mesh failures on render — within tolerance).

### Available but not yet in v3 mix

#### `benchcad-easy` — 109,804 items (NEW, just filled this session)
- **HF**: `Hula0401/cad-sft/benchcad-easy/` (55 shards, all render_img filled)
  + `BenchCAD/benchcad-easy` upstream parquet (88,773 / 109,804 = 80.8% covered).
- **Family**: same `simple_*` taxonomy as benchcad/benchcad_simple.
- **Status**: data is ready, NOT yet wired into a config.
- **✅ Pros**:
  - 10× the size of `benchcad`. Same family signal at scale.
  - Likely +0.03-0.05 BC val from added image diversity (estimate based on
    v2's 50× benchcad ablation).
- **❌ Cons**:
  - Same family as benchcad/simple — does NOT add new op vocabulary.
  - Some codes have pathological tessellations (50w-face coil-spring
    family); already mitigated via loose `tessellate(0.01, 0.5)`.

#### `cad_recode_20k` — 18,987 items (legacy, NOT in v3)
- **Ops/case**: median **7**, mean 6.64, p95 10. (~40% more ops/case than
  any other source.)
- **Status**: was used in pre-v3 mixes. Replaced by `cad_recode_bench` at
  scale. Keep for ablations / single-source baselines.
- **✅ Pros**: per-case complexity; only source where each item has 7+ ops
  on average.
- **❌ Cons**: only 19k items; v3 prefers the 472k `cad_recode_bench`
  variant.

---

### Evaluation (held-out, never trained on)

| corpus | source | local | size | use |
|---|---|---|---:|---|
| BenchCAD val | local 90/10 hash split of `BenchCAD/cad_bench` | `data/benchcad/val/` | 1,973 | online IoU, n=50 sample |
| DeepCAD test | `Hula0401/deepCAD_test` | `data/deepcad_test_mesh/` | 8,046 | online IoU, n=50 sample. **The RL gate is greedy IoU ≥ 0.8 here.** |
| Fusion360 test | `Hula0401/fusion360_test_mesh` | `data/fusion360_test_mesh/` | 1,725 | online IoU, n=50 sample |
| recode20k train (probe) | uses `cad_recode_20k` from training corpus (n=50 sample) | – | 50/eval | rare-op recall sanity probe (NOT held-out) |

All eval buckets render meshes from `gt_code` then compare via
`compute_metrics`. See `train/sft/online_eval.py` for the exact eval loop;
greedy IoU + max@8 (8 candidates at t=1.0) emit per-bucket every
`eval_steps`.

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
