# Tools

Reusable scripts for the SFT pipeline: data preparation, rendering, evaluation,
and analysis. Each script has a full docstring — run with `--help`.

## Data preparation

| Script | Purpose |
|--------|---------|
| `rewrite_recode_to_bench.py` | AST-rewrite CAD-Recode v1.5 `.py` scripts into multi-line bench-shell CadQuery style. Parallel multiprocessing. |
| `verify_recode_rewrite.py` | Verify the rewrite preserves geometry — executes original + rewritten code and compares mesh volume / bbox / IoU. |
| `push_bench_to_hf.py` | Package the rewritten recode + text2cad + benchcad corpus into parquet shards and push to `BenchCAD/cad_sft_training` on HuggingFace. |
| `deepcad2mesh.py` | Convert DeepCAD JSON CAD sequences → STL meshes via cadlib + OCP. |
| `fusion360_train_mesh.py` | Assemble Fusion360 training meshes. |
| `create_smoke_dataset.py` | Build a tiny fixed-seed subset for smoke tests. |
| `prerender_dataset.py` | Pre-render STL meshes to 4-view PNG grids (`{stem}_render.png`). Run once to skip on-the-fly rendering during training. |
| `_download_ckpts.py` | Download pre-trained Cadrille checkpoints from HuggingFace. |

## Evaluation

| Script | Purpose |
|--------|---------|
| `eval_img.py` | Partial img-mode eval on DeepCAD / Fusion360 / CC3D. Renders meshes, runs inference, calls `evaluate.py`. |
| `eval_bench.py` | Evaluate Cadrille on HF bench track (`Hula0401/test_bench`). Runs model inference, executes GT/gen code, computes IoU+CD. Resume-safe. |
| `smoke_eval.py` | Standalone smoke test: run inference on N smallest meshes, report IoU per checkpoint. |
| `run_new_eval.sh` | Convenience launcher for the new eval pipeline. |

## Analysis / visualization

| Script | Purpose |
|--------|---------|
| `analyze_dim_errors.py` | Break down IoU failures by geometric dimension error. |
| `analyze_errors.py` | General error taxonomy across an eval run. |
| `bench_compare_vis.py` | Side-by-side mesh renders for bench samples (GT vs prediction). |
| `bench_visualize.py` | Render generated STEP files and compare against GT composite PNG per bench sample. |
| `render_singleview_grid.py` | Build single-view contact sheet from 4-view DeepCAD / Fusion360 renders. |
| `render_comparison_grid.py` | Side-by-side comparison grid across checkpoints / splits. |
