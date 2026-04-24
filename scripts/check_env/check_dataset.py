"""Verify one cad-recode-v1.5 sample: exec its .py, tessellate, render to image.

Run: uv run python tools/check_env/check_dataset.py [--sample data/cad-recode-v1.5/train/batch_00/0.py]
"""
from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample",
        default="data/cad-recode-v1.5/train/batch_00/0.py",
        help="Path to a single cad-recode .py sample",
    )
    args = ap.parse_args()

    py_path = Path(args.sample)
    if not py_path.exists():
        print(f"FAIL: sample not found at {py_path}")
        return 1

    import cadquery as cq  # noqa: re-exported into globals for exec'd user code
    src = py_path.read_text()
    ns: dict = {"cq": cq}
    try:
        exec(src, ns)
    except Exception as e:
        print(f"FAIL: exec({py_path}) raised {type(e).__name__}: {e}")
        return 1
    if "r" not in ns:
        print("FAIL: sample did not define `r`")
        return 1

    import trimesh
    compound = ns["r"].val()
    vertices, faces = compound.tessellate(0.001, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
    print(f"  {py_path.name}: {len(vertices)} verts, {len(faces)} faces")

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as fp:
        stl_path = Path(fp.name)
    mesh.export(stl_path)

    # Render via Open3D headless
    import open3d as o3d
    o3d_mesh = o3d.io.read_triangle_mesh(str(stl_path))
    o3d_mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(visible=False, width=256, height=256):
        print("FAIL: Open3D create_window(visible=False) returned False")
        return 1
    vis.add_geometry(o3d_mesh)
    vis.poll_events()
    vis.update_renderer()
    import numpy as np
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    print(f"  rendered {img.shape} mean={img.mean():.3f}")
    if img.size == 0:
        print("FAIL: empty render")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
