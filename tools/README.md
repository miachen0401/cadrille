# Tools

Reusable scripts for evaluation and analysis. Each has a full docstring — run with `--help`.

| Script | Purpose |
|--------|---------|
| `eval_img.py` | Partial img-mode eval on DeepCAD / Fusion360 / CC3D. Renders meshes, runs inference, calls evaluate.py. |
| `deepcad2mesh.py` | Convert DeepCAD JSON CAD sequences → STL meshes via cadlib + OCP. |
