"""Prep a tiny SFT smoke dataset from data/cad-recode-v1.5/val/*.py.

Generates STL meshes next to each .py file and writes train.pkl so the
regular CadRecodeDataset can load it. Takes ~5 min on 8 workers.

Usage:
    uv run python scripts/prep_smoke_data.py
"""
import os
import pickle
import sys
from glob import glob
from multiprocessing import Pool

import trimesh


ROOT = 'data/cad-recode-v1.5'
VAL_DIR = os.path.join(ROOT, 'val')


def _py_to_stl(py_path):
    stl_path = py_path[:-3] + '.stl'
    if os.path.exists(stl_path):
        return py_path, True
    try:
        with open(py_path) as f:
            code = f.read()
        g = {}
        exec(code, g)
        compound = g['r'].val()
        vertices, faces = compound.tessellate(0.001, 0.1)
        mesh = trimesh.Trimesh(
            [(v.x, v.y, v.z) for v in vertices], faces)
        mesh.export(stl_path)
        return py_path, True
    except Exception as e:
        return py_path, False


def main():
    py_files = sorted(glob(os.path.join(VAL_DIR, '*.py')))
    if not py_files:
        print(f'No .py files in {VAL_DIR}. Download cad-recode-v1.5 first.')
        sys.exit(1)

    print(f'Converting {len(py_files)} .py → .stl (8 workers) ...')
    with Pool(8) as pool:
        results = pool.map(_py_to_stl, py_files)

    ok = [p for p, success in results if success]
    print(f'OK: {len(ok)} / {len(py_files)}')

    annotations = []
    for py_path in ok:
        rel_py = os.path.relpath(py_path, ROOT)
        rel_stl = rel_py[:-3] + '.stl'
        annotations.append({'py_path': rel_py, 'mesh_path': rel_stl})

    out = os.path.join(ROOT, 'train.pkl')
    with open(out, 'wb') as f:
        pickle.dump(annotations, f)
    print(f'Wrote {out}  ({len(annotations)} rows)')

    # Copy to val.pkl as well — using the same set is fine for a smoke run;
    # Trainer will still log eval/loss against it.
    val_out = os.path.join(ROOT, 'val.pkl')
    with open(val_out, 'wb') as f:
        pickle.dump(annotations, f)
    print(f'Wrote {val_out} ({len(annotations)} rows, same as train for smoke)')


if __name__ == '__main__':
    main()
