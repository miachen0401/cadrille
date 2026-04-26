# plan — task queue (pop from top)

> Repo sits on branch `revision`. SFT run `sft-s4k-lr2e-4-b15a2-img-0424-1805`
> is currently training (img-only, benchcad+recode20k 1:1, bs=15 acc=2, 4000
> steps). Everything below should be done **while it trains** or after.

## Pending queue

### T1 — Diversity analysis script vs GT (in progress)
File: `scripts/analysis/diversity_analysis.py`

**Step 1 (now):** aggregate-only — op distribution over a fixed bench subset.
  - Pull N items from `data/benchcad/val/` with seed=42
  - For each item, generate K samples at a small temperature sweep (e.g.
    t=0, 0.5, 1.0)
  - Build **two distributions**:
    - GT op frequency: how many items in the sample have op X (`.cylinder`,
      `.revolve`, `.fillet`, `.cut`, `.hole`, etc.)
    - Pred op frequency: same count on generated code (after exec succeeds
      OR on raw regex — start regex, no exec dependency)
  - Emit a markdown table: op | GT count | pred count | delta
  - Also: distinct-code-hash count per (item × temp) to quantify raw diversity
  - Output: `eval_outputs/diversity_<tag>/summary.md`

**Step 2 (later, deferred per user):** per-item side-by-side ops diff for human
  inspection — only after aggregate view is in place.

- Target ckpt: current run's `checkpoint-1000` (sft-s5k from earlier was
  deleted to free disk).

### T2 — Push SFT checkpoints to HF during training
- Write `scripts/analysis/push_ckpt_to_hf.py` (or wire a TrainerCallback):
  - On `on_save` event, detect newly-written `checkpoint-<N>/`
  - Push it to a HF model repo (e.g. `Hula0401/cadrille-sft-<tag>/<step>`) in
    background (non-blocking so training is unaffected)
  - Requires HF_TOKEN with write scope
- Benefits: (a) durable backup given our tiny 18 GB disk, (b) new collaborators
  can pull the latest ckpt from HF instead of re-running SFT.

### T8 — BenchCAD-style rewrite of recode/text2cad + 100k SFT data scale-up  ✅ DONE 2026-04-26

**Goal:** rewrite recode + text2cad code to BenchCAD shell style (v2: sketch
→ direct Workplane / `.workplane(offset)` → `.transformed` / `mode='s'` →
`.cut()`), then scale recode SFT data from 20k to 100k (reuse old 20k PNGs,
add 80k via `prepare_hf_cadrecode.py --offset 20000 --n 80000`).

**Result:** Hula0401/cad-sft on HF now has cad-recode-bench/ (49 parquet
shards, ~100k rows) and text2cad-bench/ (6 files, ~90k rows). All bench
style. See progress.md for full timeline. Next: smoke train.sft on
configs/sft/mix_bc_rb_t2cb.yaml.

**Status (2026-04-26):**
- ✅ Path X selected (Level-1 rewrite: format + 6 ops, no semantic guesswork)
- ✅ Pattern survey done — Rule A coverage: ~85% recode + ~77% text2cad
- ✅ IoU 6/6 = 1.0000 on hand-crafted rewrite patterns (T1-T6)

**Steps:**
1. Write `data_prep/rewrite_recode_to_benchcad_v2.py` (rules A+B+D, fallback
   to v1 AST pass on unsupported patterns)
2. 200-sample IoU validation (recode + text2cad, target ≥95% @ IoU≥0.99)
3. Re-run op-distribution table — confirm sketch/finalize/assemble drops
   from ~92% to ~0% on rewritable subset
4. **Phase A:** re-pack old 20k by downloading existing HF parquets, swap
   `code` field with v1.5-benchcad version (PNG bytes reused, no re-render),
   upload to `Hula0401/cad-sft/cad-recode-bench/`
5. **Phase B:** restore `prepare_hf_cadrecode.py` from commit 2f6396c, run
   `--seed 42 --offset 20000 --n 80000` (zero overlap with original 20k via
   1.3× over-sample slice [26000:130000]), incremental push every ~20k
6. **Phase C:** rewrite all text2cad cadquery/*.py via v2, IoU-sample 1k,
   pack + upload to `text2cad-bench/`
7. **Phase D:** wire fetcher (`--variant bench`) + new `CadRecodeBenchDataset`
   / `Text2CADBenchDataset` + flip configs/sft to bench

### T3 — Next SFT run: text2cad mix at 2:1:1
Config: `configs/sft/mix_bc_r20k_t2c.yaml` (to write)

- sft_mix_weights: **benchcad:2, recode20k:1, text2cad:1**
- use_text: true (wires Text2CADDataset)
- mode: img (Text2CADDataset provides no visual — collate auto-routes to
  pure-text branch for those items)
- Other hyperparameters: same as `mix_bc_r20k.yaml`
- Dataset scale: 18k + 19k + 76k = ~113k; at eff batch 30, 3 epochs ≈ 11k steps
- Start after current run completes.

### T4 — Extended eval sweep with feature preservation
Fire the moment T3 finishes (or earlier on checkpoint-<N> for progress checks):

```bash
set -a; source .env; set +a
CKPT=checkpoints/sft-s10k-lr2e-4-b6a5-img-0424-2001/checkpoint-final
OUT=eval_outputs/sweep_t3_$(date +%Y%m%d_%H%M)
nohup uv run python -u -m eval.bench_sweep \
    --ckpt "$CKPT" \
    --datasets benchcad,deepcad,fusion360 \
    --temps 0,0.4,0.5,0.75,1.0,1.25 \
    --n-samples 16 --limit 50 --seed 42 \
    --modality img --batch-size 8 --score-workers 16 \
    --out "$OUT" \
    --label "sft-t3-bc2-r20k1-t2c1-final" > logs/sweep_t3.log 2>&1 &
```

Output: `<out>/summary.md` + `summary.json` + `full.json`. Includes
feature_recall (benchcad): has_hole, has_fillet, has_chamfer, has_slot,
rotational. Cost: ~60 min generation + ~60 min scoring (tunable with
fewer samples or temps).

### T5 — From-zero one-click verification
- After the moves above are in, smoke-test:
    git clone … && cd cadrille
    bash scripts/setup.sh --data          # pulls benchcad + cad-sft + eval meshes
    uv run python -m train.sft --config configs/sft/mix_bc_r20k.yaml
- Fix any remaining gaps (apt deps prompt, Open3D source build auto-run).

### T6 — RL training (deferred until SFT stable)
- `python -m train.rl.train --config configs/rl/a100.yaml`
- (Not started; awaiting SFT converged weights.)

### T7 — Alternative dense VL backbones (Qwen2.5-VL, Qwen3-VL, …)
**Goal:** make Cadrille's backbone swappable so we can A/B Qwen2-VL-2B (current)
against newer / larger dense VLMs without forking the training pipeline.

**Current hardcoding** (audit before refactor):
- `common/model.py:5` — `from transformers import Qwen2VLForConditionalGeneration`
- `common/model.py:6` — `Qwen2VLCausalLMOutputWithPast` import
- `common/model.py:187` — `class Cadrille(Qwen2VLForConditionalGeneration)`
- `common/model.py:334` — `return Qwen2VLCausalLMOutputWithPast(…)` in forward
- Configs: `base_model: Qwen/Qwen2-VL-2B-Instruct` (and warm-start ckpt paths)
- `bad_words_ids=[[model.config.video_token_id]]` in eval/generation paths
- collate.py: `process_vision_info` from `qwen_vl_utils` — Qwen2/2.5 share, Qwen3 may differ

**Subtasks (do ONE backbone end-to-end before starting the next):**

T7.1 — **Refactor Cadrille to a backbone-agnostic mixin**
  - New `common/model.py::make_cadrille_class(BackboneCls, OutputCls) -> Cadrille`
  - Move FourierPointEncoder injection + custom forward into a mixin that
    `__init__`'s onto any `*VLForConditionalGeneration` parent.
  - Keep current `Cadrille` (= Qwen2-VL backed) as the default for back-compat.
  - Add `cfg['backbone']: qwen2_vl | qwen2_5_vl | qwen3_vl` switch in train/sft/train.py.
  - Smoke: `python -m train.sft --config configs/sft/smoke.yaml backbone=qwen2_vl`
    matches current behavior bit-for-bit.

T7.2 — **Qwen2.5-VL** (transformers 4.50.3 already supports it; same vision
  token layout as 2-VL, drop-in via the mixin)
  - Add `configs/sft/mix_bc4_r20k_t2c_qwen25vl3b.yaml`
    `base_model: Qwen/Qwen2.5-VL-3B-Instruct` (3 B is the closest size to current 2 B)
  - One-batch forward smoke (`python -c …` or a `tests/test_backbone_swap.py`)
  - 500-step toy SFT run, confirm IoU + ops eval emit cleanly, no shape mismatches
  - Full 20 k SFT run with same data mix → compare to Qwen2-VL-2B run on
    `op_loss_cos_weighted`, `rare_op_macro_recall`, IoU at matched steps

T7.3 — **Qwen3-VL** (blocked on transformers release containing
  `Qwen3VLForConditionalGeneration` — not in 4.50.3)
  - Pin transformers to the first version that ships Qwen3-VL
  - Re-run T7.1 smoke to ensure mixin still composes
  - Add `configs/sft/...qwen3vl.yaml`
  - 500-step toy → 20 k full

T7.4 — **A/B comparison report** (after ≥ 2 backbones have a full SFT run)
  - `docs/backbone_ab_2026-…md`: side-by-side wandb metrics (IoU,
    op_loss_cos_weighted, rare_op_macro_recall, exec_rate, distinct_codes_frac)
    at matched training step + matched compute, plus example generations.
  - Decision criteria: ≥ +0.05 IoU on BenchCAD val OR ≥ +0.10 rare_op_macro_recall
    to justify swapping the default backbone.

**Other dense VLM candidates** (not on the immediate critical path; consider
for T7.5+ once Qwen2.5/Qwen3 paths are proven):
  - InternVL2.5 / InternVL3 (different processor + vision tokenizer)
  - LLaVA-OneVision (transformers-native)
  - PaliGemma2 (Gemma2 LM, much smaller VL token budget)
  - MiniCPM-V 2.6 (good 8 B parameter-efficient option)
  - Llama-3.2-Vision (11 B / 90 B)
  - Each needs its own `process_vision_info` adapter; the mixin should be
    extended to take a `vision_info_fn` callable.

**Constraint:** all of T7 happens IN PARALLEL with the current SFT run on
master config (`mix_bc4_r20k_t2c.yaml` on Qwen2-VL-2B). Do not stop training.
Use the / branch for refactor + smoke; full backbone-comparison runs can wait
until current 20 k run finishes (~16 h ETA from 2026-04-25 06:00).

## Recent history (quick recall)

- SFT run 1 (benchcad-only, 5k steps, bs=4 acc=2): eval_loss 1.40→0.296@step1500,
  then overfit. 94.7% exec rate @ img modality on Hula0401/test_bench. Ckpt
  deleted to free disk.
- SFT run 2 (current, sft-s4k, bs=15 acc=2, 268×268 native, benchcad+recode20k
  1:1): in progress. Baseline eval_loss 1.253 → 0.275 @ step 500 → 0.276 @
  step 1000. Cleaner than run 1.
- eval sweep at sft-s5k: benchcad/val greedy exec=100% iou=0.147; any sampling
  temperature collapses exec to <10% — severe overfit symptom. Aborted for
  run 2 retest.
- Fixes during runs: accelerate 0.34→1.3 (data_seed requires 1.1+),
  show_object() stubbed in common.metrics exec contexts, img modality strict
  (no on-the-fly Open3D fallback).

## Canonical data sources (memorised)

| purpose | source | local |
|---|---|---|
| train BenchCAD | HF `BenchCAD/cad_bench` → `data_prep/fetch_benchcad.py` | `data/benchcad/` (18k+2k, .py+.stl+_render.png) |
| train cad-sft | HF `Hula0401/cad-sft` → `data_prep/fetch_cad_sft.py` | `data/cad-recode-20k/` (18k+1k, .py+_render.png); `data/text2cad/` (76k/6k/8k, .py+description only) |
| eval DeepCAD | HF `Hula0401/deepCAD_test` | `data/deepcad_test_mesh/` (8046, .stl+_render.png) |
| eval Fusion360 | HF `Hula0401/fusion360_test_mesh` | `data/fusion360_test_mesh/` (1725, .stl+_render.png) |
| eval BenchCAD | local 90/10 split of `BenchCAD/cad_bench` | `data/benchcad/val/` (1973) |

## Env invariants

- A100 80GB, uv venv at `.venv/`, torch 2.5.1+cu124, flash-attn 2.7.2.post1,
  pytorch3d 0.7.8, cadquery 2.5.0.dev0, open3d-cpu 0.18.0+8e43455 (source-built).
- `.env` has HF_TOKEN, WANDB_API_KEY, BenchCAD_HF_TOKEN, GITHUB_PAT_TOKEN.
- Disk 97 GB total, currently ~18 GB free. Monitor; keep SFT checkpoint dirs
  ≤ 2× ckpt size by preferring `save_only_model=true` + `save_total_limit=1`.
