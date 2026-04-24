# Tools

Model-behaviour analysis scripts. Data prep, benchmarks, paper-eval, and the
repair-LoRA experiment now live in their own top-level packages — see the
pointers at the bottom. Each script here has a full docstring; run with `--help`.

## Analysis

| Script | Purpose |
|--------|---------|
| `analyze_errors.py` | Full-dataset inference + rule-based error taxonomy. 8046 DeepCAD + 1725 Fusion360 cases × 2 models × 2 modalities → JSONL + pred `.py` + `.stl`. |
| `analyze_dim_errors.py` | Dim-error sub-classification (local_feat, aspect_ratio, vol_ratio) from analyze_errors.py output. |
| `analyze_sft_rl_delta.py` | Per-case SFT vs RL IoU delta: fixed / boosted / stable / regressed / broken breakdown. |
| `mining_analysis.py` | Hard-example mining health: reward/IoU distribution, exec-failure fraction, per-checkpoint shift. |
| `plot_kl_quadrants.py` | Pull `kl_quad_*` metrics from W&B runs and plot stacked-area KL quadrant composition over training. |
| `diag_resume_entropy.py` | Reproduce RL training entropy/KL at a single step to debug resume divergence. |
| `render_comparison_grid.py` | GT vs SFT-pred vs RL-pred composite grid PNG. |
| `render_singleview_grid.py` | Single-view contact sheet for DeepCAD/Fusion360 renders. |
| `bench_compare_vis.py` | Side-by-side visual diff of two bench eval runs (e.g. SFT vs RL). |

## Dataset + CadQuery parsing

| Script | Purpose |
|--------|---------|
| `parse_cq.py` | Regex-based feature extractor for CadQuery scripts. Imported by dataset_stats.py and failure_analysis.py. |
| `dataset_stats.py` | Operation frequency, code-length, plane-type distributions over a CAD corpus. |
| `compare_evals.py` | IoU/CD comparison plots across eval runs (two or more). |
| `failure_analysis.py` | Per-op failure rate + runtime-error distribution for a given eval run. |
| `fillet_analysis.py` | Fillet-specific analysis (arcs vs fillet, 3D comparisons). |
| `training_dynamics.py` | W&B training-curve plots (loss, reward, entropy) over checkpoints. |

## Env sanity (tools/check_env/)

Five fast scripts verifying the installed environment can run the full pipeline:
`check_torch.py` (+bf16), `check_open3d.py` (headless build), `check_cadquery.py`
(tessellate + STL), `check_dataset.py` (exec a recode sample), `check_model.py`
(load Qwen2-VL-2B forward). Plus `fetch_cad_recode.py` — throttled HF dataset downloader.

## Related packages

- **`data_prep/`** — one-time dataset preparation (prerender, deepcad2mesh,
  fusion360_train_mesh, rewrite_recode_to_bench, verify_recode_rewrite,
  push_bench_to_hf, fetch_benchcad, make_benchcad_partial_pkl, upload_mined,
  create_smoke_dataset, gen_repair_data).
- **`bench/`** — training throughput benchmarks (bench_config, bench_workers).
- **`eval/bench.py`, `eval/bench_visualize.py`** — model evaluation on the BenchCAD
  benchmark (formerly `tools/eval_bench.py`, `tools/bench_visualize.py`).
- **`experiments/repair_lora/`** — repair-LoRA mini-experiment (4 files).
- **`experiments/legacy_eval/`** — ad-hoc legacy eval scripts pending consolidation
  (eval_img, eval_new_ckpts, eval_temperature, smoke_eval, infer_cases,
  overfit_single).
