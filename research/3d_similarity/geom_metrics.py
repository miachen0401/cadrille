"""Surface-geometry similarity that's more tolerant than IoU.

The headline metric is **F-score @ τ** — the harmonic mean of the precision
and recall at a fixed surface-distance threshold τ. It's the standard
robustness companion to Chamfer Distance in modern 3D reconstruction
benchmarks (Tatarchenko et al. CVPR 2019).

Both meshes should already be normalised to a common unit cube (in our
pipeline, [-1, 1]^3 via `transform_real_mesh` — `compute_iou`'s convention).
τ is then in those normalised units. τ = 0.05 ≈ 5% of the bounding cube
side, a sensible default for "do these surfaces broadly agree".
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fscore_at_tau(
    gt_mesh,
    pred_mesh,
    tau: float = 0.05,
    n_points: int = 8192,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """F-score @ τ between two trimesh meshes (already normalised to a common cube).

    Sample n_points uniformly on each mesh's surface; for each sampled pred
    point look up the nearest GT-surface point (and vice versa). Threshold
    each distance at τ → fraction-within counts as precision (pred→gt) and
    recall (gt→pred). F1 = 2·P·R/(P+R).

    Returns (fscore, precision, recall). Any can be None on degenerate input.
    """
    try:
        from scipy.spatial import cKDTree
        pred_pts = pred_mesh.sample(n_points).astype(np.float32)
        gt_pts   = gt_mesh.sample(n_points).astype(np.float32)
        tree_gt   = cKDTree(gt_pts)
        tree_pred = cKDTree(pred_pts)
        d_pg, _ = tree_gt.query(pred_pts, k=1)   # pred → gt distances
        d_gp, _ = tree_pred.query(gt_pts,  k=1)  # gt → pred distances
        precision = float((d_pg < tau).mean())   # pred surface ≤ τ from GT
        recall    = float((d_gp < tau).mean())   # gt   surface ≤ τ from pred
        if precision + recall == 0:
            return 0.0, precision, recall
        f = 2 * precision * recall / (precision + recall)
        return float(f), precision, recall
    except Exception:
        return None, None, None
