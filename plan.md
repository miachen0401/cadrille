# plan — task queue (pop from top)

> Repo sits on branch `revision`. SFT run `sft-s4k-lr2e-4-b15a2-img-0424-1805`
> is currently training (img-only, benchcad+recode20k 1:1, bs=15 acc=2, 4000
> steps). Everything below should be done **while it trains** or after.

## Pending queue

### T1 — Diversity analysis script vs GT (in progress)
File: `scripts/analysis/diversity_analysis.py`

- Pull N items from `data/benchcad/val/` with fixed seed
- For each item, generate K samples at a small temperature sweep (e.g. t=0,0.5,1.0)
- Emit three views:
  1. **Op-level recall**: for every GT op/feature_tag, is it in pred code? Per-sample hit, aggregate recall per op.
  2. **Sample-to-sample diversity** (temp > 0 only): K samples per item → count distinct op sequences and distinct full-code hashes.
  3. **GT vs best pred diff**: side-by-side ops list for human inspection.
- Output: `eval_outputs/diversity_<tag>/summary.md` + `per_item.jsonl`.
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
Run: `python -m eval.bench_sweep --ckpt <new-run>/checkpoint-final`

- Metrics: IoU, exec_rate, max_iou@16 at temps {0, 0.4, 0.5, 0.75, 1.0, 1.25}
- Datasets: benchcad (our 90/10 val split), deepcad (test), fusion360 (test)
- BenchCAD-only: feature_recall per feature (has_hole / has_fillet /
  has_chamfer / has_slot / rotational) — reuses `eval/features.py`
- Target limit: ~50 per split with seed=42 (user signed off at 50-100)

### T5 — From-zero one-click verification
- After the moves above are in, smoke-test:
    git clone … && cd cadrille
    bash scripts/setup.sh --data          # pulls benchcad + cad-sft + eval meshes
    uv run python -m train.sft --config configs/sft/mix_bc_r20k.yaml
- Fix any remaining gaps (apt deps prompt, Open3D source build auto-run).

### T6 — RL training (deferred until SFT stable)
- `bash scripts/setup.sh --full` to pull hard-mined RL data + reference ckpt
- `python -m train.rl.train --config configs/rl/a100.yaml`
- (Not started; awaiting SFT converged weights.)

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
