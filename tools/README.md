# Tools

Reusable scripts for evaluation and analysis. Each has a full docstring — run with `--help`.

| Script | Purpose |
|--------|---------|
| `eval_img.py` | Partial img-mode eval on DeepCAD / Fusion360 / CC3D. Renders meshes, runs inference, calls evaluate.py. |
| `deepcad2mesh.py` | Convert DeepCAD JSON CAD sequences → STL meshes via cadlib + OCP. |
| `prerender_dataset.py` | Pre-render STL meshes to 4-view PNG grids (`{stem}_render.png`). Run once before training to avoid on-the-fly rendering cost (~1 s/mesh). |
| `bench_config.py` | Benchmark RL training throughput across configs. Reports step time, gen/rew/grad breakdown, steps/hr, GPU peak memory. Same model + data used for all configs so results are directly comparable. |
| `bench_workers.py` | Sweep reward_workers concurrency to find optimal worker count. Generates fresh completions from the SFT model, then times compute_rewards_parallel() at 8/12/16/20/24 workers. Use --skip-generate for fast pool-overhead-only test. |
| `smoke_eval.py` | Standalone smoke test: run inference on N smallest meshes, report IoU per checkpoint. |
