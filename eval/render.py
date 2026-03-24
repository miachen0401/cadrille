"""Render GT and predicted STLs to PNG for visual inspection.

Output layout:
    renders/
        gt/{dataset}/{case_id}.png        ← copy/link from prerendered
        pred/{ckpt_label}/{dataset}_{modality}/{case_id}.png  ← rendered from pred STL

GT renders are copied from {dataset_path}/{stem}_render.png (must be prerendered).
Pred renders are generated using rl.dataset.render_img() from the saved .stl file.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np


def select_cases(metadata_path: Path, strategy: str, n: int) -> list[str]:
    """Return up to n case_ids to render, based on strategy.

    strategies:
        failures   — cases where error_type != 'success'
        low_iou    — success cases with lowest IoU
        random     — random sample of all cases
        all        — all cases (up to n)
    """
    import json
    import random

    records = []
    if not metadata_path.exists():
        return []
    with open(metadata_path) as f:
        for line in f:
            records.append(json.loads(line))

    if strategy == 'failures':
        chosen = [r['case_id'] for r in records if r['error_type'] != 'success']
    elif strategy == 'low_iou':
        success = [r for r in records if r['error_type'] == 'success' and r.get('iou')]
        success.sort(key=lambda r: r['iou'])
        chosen = [r['case_id'] for r in success]
    elif strategy == 'random':
        chosen = [r['case_id'] for r in records]
        random.Random(42).shuffle(chosen)
    else:
        # 'all'
        chosen = [r['case_id'] for r in records]

    return chosen[:n]


def copy_gt_renders(case_ids: list[str], dataset_path: Path, gt_render_dir: Path) -> int:
    gt_render_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for cid in case_ids:
        src = dataset_path / f'{cid}_render.png'
        dst = gt_render_dir / f'{cid}.png'
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            n_copied += 1
    return n_copied


def render_pred_stls(case_ids: list[str], combo_dir: Path, pred_render_dir: Path) -> int:
    pred_render_dir.mkdir(parents=True, exist_ok=True)
    n_rendered = 0
    for cid in case_ids:
        stl_path = combo_dir / f'{cid}.stl'
        dst = pred_render_dir / f'{cid}.png'
        if dst.exists() or not stl_path.exists():
            continue
        try:
            _render_stl_to_png(stl_path, dst)
            n_rendered += 1
        except Exception as e:
            continue
    return n_rendered


def _render_stl_to_png(stl_path: Path, out_png: Path) -> None:
    import tempfile
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    import trimesh
    import numpy as np

    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f'Not a valid mesh: {stl_path}')

    bounds = mesh.bounds
    scale = (bounds[1] - bounds[0]).max()
    if scale < 1e-9:
        raise ValueError('Degenerate mesh')

    mesh.apply_translation(-bounds[0])
    mesh.apply_scale(1.0 / scale)

    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        mesh.export(f.name)
        tmp_stl = f.name

    try:
        _render_stl_direct(tmp_stl, str(out_png))
    finally:
        import os
        os.unlink(tmp_stl)


def _render_stl_direct(stl_path: str, out_png: str) -> None:
    import open3d as o3d
    import numpy as np
    from PIL import Image

    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    mn, mx = verts.min(0), verts.max(0)
    scale = (mx - mn).max()
    if scale < 1e-9:
        raise ValueError('Degenerate mesh')

    mesh.translate(-mn)
    mesh.scale(1.0 / scale, center=[0, 0, 0])

    views = [
        ([2, 2, 2], [0, 0, 0]),
        ([3, 0, 0], [0, 0, 0]),
        ([0, 3, 0], [0, 0, 0]),
        ([0, 0, 3], [0, 0, 0]),
    ]

    frames = []
    for eye, center in views:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=256, height=256)
        vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front(np.array(eye) - np.array(center))
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        vis.destroy_window()
        frames.append((img * 255).astype(np.uint8))

    top = np.concatenate([frames[0], frames[1]], axis=1)
    bot = np.concatenate([frames[2], frames[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    Image.fromarray(grid).save(out_png)
