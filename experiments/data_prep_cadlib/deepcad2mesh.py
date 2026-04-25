#!/usr/bin/env python3
"""data/deepcad2mesh.py — Convert DeepCAD JSON CAD sequences → STL meshes.

Uses DeepCAD's cadlib (JSON parsing) + cadquery-ocp (OCP) for 3D reconstruction.
The OCP API mirrors pythonocc-core (OCC.Core.*) but uses OCP.* import paths.

Usage:
    python data/deepcad2mesh.py --split train --out data/deepcad_train_mesh
    python data/deepcad2mesh.py --split train --out data/deepcad_train_mesh --workers 16
    python data/deepcad2mesh.py --split train --out data/deepcad_train_mesh --limit 100  # smoke test
"""

import os
import sys
import json
import argparse
from copy import copy
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from tqdm import tqdm
import numpy as np

# ── DeepCAD cadlib — JSON parsing only ────────────────────────────────────────
_DEEPCAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'DeepCAD')
sys.path.insert(0, os.path.abspath(_DEEPCAD_DIR))

from cadlib.extrude import CADSequence, EXTRUDE_OPERATIONS, EXTENT_TYPE  # noqa: E402
from cadlib.curves import Line, Arc, Circle  # noqa: E402

# ── OCP (cadquery-ocp) — 3D geometry reconstruction ───────────────────────────
from OCP.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax2, gp_Ax3, gp_Pln, gp_Circ
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCP.GC import GC_MakeArcOfCircle
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS


# ── Geometry helpers (ported from DeepCAD/cadlib/visualize.py) ────────────────

def _pt(point, sketch_plane):
    """Convert 2D sketch-plane point → 3D gp_Pnt."""
    g = point[0] * sketch_plane.x_axis + point[1] * sketch_plane.y_axis + sketch_plane.origin
    return gp_Pnt(float(g[0]), float(g[1]), float(g[2]))


def _edge(curve, sketch_plane):
    """Build one OCP edge from a DeepCAD curve + sketch plane. Returns None to skip."""
    if isinstance(curve, Line):
        if np.allclose(curve.start_point, curve.end_point):
            return None
        return BRepBuilderAPI_MakeEdge(_pt(curve.start_point, sketch_plane),
                                       _pt(curve.end_point,   sketch_plane)).Edge()

    elif isinstance(curve, Circle):
        center = _pt(curve.center, sketch_plane)
        axis   = gp_Dir(*[float(x) for x in sketch_plane.normal])
        circ   = gp_Circ(gp_Ax2(center, axis), float(abs(curve.radius)))
        return BRepBuilderAPI_MakeEdge(circ).Edge()

    elif isinstance(curve, Arc):
        arc = GC_MakeArcOfCircle(
            _pt(curve.start_point, sketch_plane),
            _pt(curve.mid_point,   sketch_plane),
            _pt(curve.end_point,   sketch_plane),
        ).Value()
        return BRepBuilderAPI_MakeEdge(arc).Edge()

    raise NotImplementedError(type(curve))


def _wire(loop, sketch_plane):
    wire = BRepBuilderAPI_MakeWire()
    for curve in loop.children:
        e = _edge(curve, sketch_plane)
        if e is not None:
            wire.Add(e)
    return wire.Wire()


def _face(profile, sketch_plane):
    origin = gp_Pnt(*[float(x) for x in sketch_plane.origin])
    normal = gp_Dir(*[float(x) for x in sketch_plane.normal])
    x_axis = gp_Dir(*[float(x) for x in sketch_plane.x_axis])
    gp_face = gp_Pln(gp_Ax3(origin, normal, x_axis))
    loops   = [_wire(loop, sketch_plane) for loop in profile.children]
    face    = BRepBuilderAPI_MakeFace(gp_face, loops[0])
    for loop in loops[1:]:          # inner loops → holes
        face.Add(TopoDS.Wire_s(loop.Reversed()))
    return face.Face()


def _extrude(extrude_op):
    profile      = copy(extrude_op.profile)
    profile.denormalize(extrude_op.sketch_size)
    sketch_plane = copy(extrude_op.sketch_plane)
    sketch_plane.origin = extrude_op.sketch_pos

    face   = _face(profile, sketch_plane)
    normal = gp_Dir(*[float(x) for x in extrude_op.sketch_plane.normal])
    vec1   = gp_Vec(normal).Multiplied(float(extrude_op.extent_one))
    body   = BRepPrimAPI_MakePrism(face, vec1).Shape()

    sym_idx = EXTENT_TYPE.index("SymmetricFeatureExtentType")
    two_idx = EXTENT_TYPE.index("TwoSidesFeatureExtentType")
    if extrude_op.extent_type == sym_idx:
        body = BRepAlgoAPI_Fuse(body, BRepPrimAPI_MakePrism(face, vec1.Reversed()).Shape()).Shape()
    elif extrude_op.extent_type == two_idx:
        vec2 = gp_Vec(normal.Reversed()).Multiplied(float(extrude_op.extent_two))
        body = BRepAlgoAPI_Fuse(body, BRepPrimAPI_MakePrism(face, vec2).Shape()).Shape()
    return body


def _create_cad(cad_seq):
    new_idx  = EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation")
    join_idx = EXTRUDE_OPERATIONS.index("JoinFeatureOperation")
    cut_idx  = EXTRUDE_OPERATIONS.index("CutFeatureOperation")
    int_idx  = EXTRUDE_OPERATIONS.index("IntersectFeatureOperation")

    body = _extrude(cad_seq.seq[0])
    for op in cad_seq.seq[1:]:
        new_body = _extrude(op)
        if op.operation in (new_idx, join_idx):
            body = BRepAlgoAPI_Fuse(body, new_body).Shape()
        elif op.operation == cut_idx:
            body = BRepAlgoAPI_Cut(body, new_body).Shape()
        elif op.operation == int_idx:
            body = BRepAlgoAPI_Common(body, new_body).Shape()
    return body


def _write_stl(shape, path, deflection=0.001):
    """Tessellate OCC shape, normalize to [0,1]³ centered at [0.5,0.5,0.5], write STL.

    mesh_to_image() has lookat=[0.5,0.5,0.5] hardcoded — meshes must be in [0,1]³
    to appear centred in rendered views, matching the existing deepcad_test_mesh STLs.
    DeepCAD's cad_seq.normalize() puts geometry in ±0.75 (centered at origin), so we
    rescale: max_extent→1, translate center→[0.5,0.5,0.5].
    """
    import trimesh
    BRepMesh_IncrementalMesh(shape, deflection).Perform()
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        tmp_path = tmp.name
    StlAPI_Writer().Write(shape, tmp_path)
    mesh = trimesh.load(tmp_path)
    os.unlink(tmp_path)
    # Normalize: scale max extent → 1, center at [0.5, 0.5, 0.5]
    extents = mesh.bounding_box.extents          # [dx, dy, dz]
    scale   = 1.0 / extents.max()
    mesh.apply_scale(scale)
    center  = mesh.bounding_box.centroid
    mesh.apply_translation([0.5 - center[0], 0.5 - center[1], 0.5 - center[2]])
    mesh.export(path)


# ── Per-file worker (runs in subprocess) ──────────────────────────────────────

def _convert_one(args):
    """Convert one JSON file to STL. Returns status string."""
    json_path, out_path = args
    if os.path.exists(out_path):
        return 'skip'
    try:
        with open(json_path) as f:
            data = json.load(f)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = _create_cad(cad_seq)
        if not BRepCheck_Analyzer(shape).IsValid():
            return 'invalid'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        _write_stl(shape, out_path)
        return 'ok'
    except Exception as e:
        return f'err:{e}'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Convert DeepCAD JSON → STL')
    parser.add_argument('--json-dir',   default='data/cad_json/cad_json',
                        help='Root of extracted cad_json/ directory')
    parser.add_argument('--split-file', default='data/cad_json/data/train_val_test_split.json',
                        help='Path to train_val_test_split.json')
    parser.add_argument('--split',   default='train',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--out',     default='data/deepcad_train_mesh',
                        help='Output directory for STL files')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=30,
                        help='Per-file timeout in seconds (OCC can hang on degenerate geometry)')
    parser.add_argument('--limit',   type=int, default=None,
                        help='Process only first N files (smoke test)')
    args = parser.parse_args()

    with open(args.split_file) as f:
        split_ids = json.load(f)[args.split]
    if args.limit:
        split_ids = split_ids[:args.limit]

    # Build task list: (json_path, out_path)
    tasks, missing = [], 0
    for uid in split_ids:
        stem      = uid.split('/')[-1]
        folder    = uid.split('/')[0]
        json_path = os.path.join(args.json_dir, folder, f'{stem}.json')
        out_path  = os.path.join(args.out, f'{stem}.stl')
        if not os.path.exists(json_path):
            missing += 1
            continue
        tasks.append((json_path, out_path))

    os.makedirs(args.out, exist_ok=True)
    print(f'Split : {args.split}  |  {len(tasks)} files  '
          f'({missing} JSON not found)  |  workers={args.workers}  timeout={args.timeout}s')
    print(f'Output: {args.out}')

    counts = {'ok': 0, 'skip': 0, 'invalid': 0, 'err': 0, 'timeout': 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_convert_one, t): t for t in tasks}
        pbar = tqdm(total=len(tasks), desc='json→stl')
        for fut in futures:
            try:
                result = fut.result(timeout=args.timeout)
            except FuturesTimeout:
                fut.cancel()
                result = 'timeout'
            except Exception as e:
                result = f'err:{e}'
            if result == 'ok':          counts['ok']      += 1
            elif result == 'skip':      counts['skip']     += 1
            elif result == 'invalid':   counts['invalid']  += 1
            elif result == 'timeout':   counts['timeout']  += 1
            else:                       counts['err']      += 1
            pbar.update(1)
            pbar.set_postfix(counts)
        pbar.close()

    total = sum(counts.values())
    print(f'\nDone  : {counts["ok"]} ok  |  {counts["skip"]} skipped  |  '
          f'{counts["invalid"]} invalid  |  {counts["timeout"]} timeout  |  '
          f'{counts["err"]} errors  (total {total})')
    print(f'STLs  : {args.out}')


if __name__ == '__main__':
    main()
