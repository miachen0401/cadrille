# Tools

Reusable scripts for evaluation and analysis. Each has a full docstring — run with `--help`.

| Script | Purpose |
|--------|---------|
| `eval_img.py` | Partial img-mode eval on DeepCAD / Fusion360 / CC3D. Renders meshes, runs inference, calls evaluate.py. |
| `deepcad2mesh.py` | Convert DeepCAD JSON CAD sequences → STL meshes via cadlib + OCP. |
| `prerender_dataset.py` | Pre-render STL meshes to 4-view PNG grids (`{stem}_render.png`). Run once before training to avoid on-the-fly rendering cost (~1 s/mesh). |
