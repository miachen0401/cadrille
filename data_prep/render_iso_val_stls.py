"""Render the 100 iso val GT meshes (50 IID + 50 OOD) needed for online IoU eval.

Without GT STLs, the iso val buckets have no IoU during SFT online eval —
only ess_ops. This script exec's the GT .py and tessellates to STL for the
exact 100 uids that show up in the §7 v2 eval (frozen via seed=42).

Output: data/cad-iso-106/val_meshes/{uid}.stl

Usage:
    uv run python data_prep/render_iso_val_stls.py
"""
from __future__ import annotations
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
ISO_ROOT = REPO / 'data/cad-iso-106'
OUT_DIR = ISO_ROOT / 'val_meshes'


def render_one(py_path: Path, out_stl: Path) -> bool:
    captured: dict = {}
    g = {'show_object': lambda obj, *a, **kw: captured.setdefault('r', obj)}
    try:
        exec(py_path.read_text(), g)
    except Exception as e:
        print(f'  exec failed for {py_path.name}: {e!r}')
        return False
    result = captured.get('r') or g.get('result') or g.get('r')
    if result is None:
        print(f'  no result captured for {py_path.name}')
        return False
    try:
        verts, faces = result.val().tessellate(0.1)
        verts_np = np.array([(v.x, v.y, v.z) for v in verts])
        faces_np = np.array(faces)
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
        mesh.export(out_stl)
    except Exception as e:
        print(f'  tessellate/export failed for {py_path.name}: {e!r}')
        return False
    return True


def main() -> None:
    uids_path = REPO / 'data/_eval_uids/v2_eval_uids.json'
    if not uids_path.exists():
        print('eval uid dump missing — run scripts/dump_eval_uids.py first')
        sys.exit(1)
    eval_uids = json.loads(uids_path.read_text())
    needed = list(set(eval_uids['iso val IID']) | set(eval_uids['iso val OOD']))
    print(f'Rendering {len(needed)} iso val GT meshes → {OUT_DIR}')

    rows = pickle.load(open(ISO_ROOT / 'val.pkl', 'rb'))
    by_uid = {r['uid']: r for r in rows}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    skip = 0
    t0 = time.time()
    for i, uid in enumerate(sorted(needed), 1):
        if uid not in by_uid:
            print(f'  uid {uid!r} not in val.pkl — skip')
            fail += 1
            continue
        py_path = ISO_ROOT / by_uid[uid]['py_path']
        out_stl = OUT_DIR / f'{uid}.stl'
        if out_stl.exists():
            skip += 1
            continue
        if render_one(py_path, out_stl):
            ok += 1
        else:
            fail += 1
        if i % 20 == 0:
            print(f'  {i}/{len(needed)} ok={ok} fail={fail} skip={skip}  '
                  f'({time.time()-t0:.1f}s)')

    print(f'\nDone. ok={ok} fail={fail} skip={skip}  total={time.time()-t0:.1f}s')
    print(f'STLs in {OUT_DIR}: {len(list(OUT_DIR.glob("*.stl")))}')


if __name__ == '__main__':
    main()
