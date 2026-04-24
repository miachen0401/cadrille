"""Verify cadquery (git version) + cadquery-ocp can build + tessellate + export STL.

Run: uv run python tools/check_env/check_cadquery.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import cadquery as cq
    print(f"cadquery {cq.__version__}")

    r = cq.Workplane("XY").box(10, 10, 10)
    compound = r.val()

    vertices, faces = compound.tessellate(0.001, 0.1)
    print(f"  tessellate: {len(vertices)} verts, {len(faces)} faces")
    if len(vertices) == 0 or len(faces) == 0:
        print("FAIL: empty tessellation")
        return 1

    import trimesh
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as fp:
        out = Path(fp.name)
    mesh.export(out)
    size = out.stat().st_size
    print(f"  exported STL: {out}  ({size} bytes)")
    if size < 200:
        print("FAIL: STL file suspiciously small")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
