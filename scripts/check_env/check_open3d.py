"""Verify Open3D is source-built with headless rendering enabled and can render.

Run: uv run python tools/check_env/check_open3d.py
Exits 0 on success, non-zero if the PyPI (GLFW) build is in place or render fails.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import open3d as o3d

    print(f"open3d {o3d.__version__}")
    cfg = dict(o3d._build_config)
    headless = cfg.get("ENABLE_HEADLESS_RENDERING")
    print(f"  ENABLE_HEADLESS_RENDERING = {headless}")
    if not headless:
        print("\nFAIL: PyPI Open3D detected. Need source build per Dockerfile.official.")
        return 1

    # Create a unit-box TriangleMesh, render offscreen to PNG, assert non-empty.
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(visible=False, width=256, height=256)
    if not ok:
        print("FAIL: create_window(visible=False) returned False")
        return 1
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    import numpy as np
    arr = np.asarray(img)
    if arr.size == 0 or arr.mean() == 0:
        print(f"FAIL: rendered image is empty (shape={arr.shape}, mean={arr.mean()})")
        return 1

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
        out = Path(fp.name)
    o3d.io.write_image(str(out), o3d.geometry.Image((arr * 255).astype("uint8")))
    print(f"  rendered {arr.shape} mean={arr.mean():.3f} -> {out}")

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
