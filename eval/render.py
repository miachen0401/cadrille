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
    from rl.dataset import render_img
    pred_render_dir.mkdir(parents=True, exist_ok=True)
    n_rendered = 0
    for cid in case_ids:
        stl_path = combo_dir / f'{cid}.stl'
        dst = pred_render_dir / f'{cid}.png'
        if dst.exists() or not stl_path.exists():
            continue
        try:
            result = render_img(str(stl_path))
            result['video'][0].save(str(dst))
            n_rendered += 1
        except Exception:
            continue
    return n_rendered
