"""Reward module for RL fine-tuning of Cadrille.

Uses subprocess.run() to execute CadQuery code so that:
  1. CadQuery memory leaks stay in the subprocess (not the training process)
  2. The CUDA context in the main training process is not corrupted by fork()

Reward formula (from paper):
    R(τ) = r_IoU + r_invalid
    r_IoU    = IoU × 10    (range [0, 10])
    r_invalid = -10 (invalid code) or 0 (valid code)
"""

import os
import sys
import json
import textwrap
import tempfile
import subprocess
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

# Allow standalone execution from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Worker script embedded as a string; written to a temp file on first use.
# Runs in a fresh Python interpreter — no CUDA context, no memory leaks.
_WORKER_SCRIPT = textwrap.dedent('''\
    """CadQuery mesh executor — spawned by rl/reward.py as a subprocess."""
    import sys
    import json
    import numpy as np


    def compute_iou(gt_mesh, pred_mesh):
        try:
            intersection_volume = 0
            for gt_mesh_i in gt_mesh.split():
                for pred_mesh_i in pred_mesh.split():
                    intersection = gt_mesh_i.intersection(pred_mesh_i)
                    volume = intersection.volume if intersection is not None else 0
                    intersection_volume += volume
            gt_volume = sum(m.volume for m in gt_mesh.split())
            pred_volume = sum(m.volume for m in pred_mesh.split())
            union_volume = gt_volume + pred_volume - intersection_volume
            assert union_volume > 0
            return float(intersection_volume / union_volume)
        except Exception:
            return None


    def compute_cd(gt_mesh, pred_mesh, n_points=8192):
        """Chamfer Distance via surface point sampling.

        Matches evaluate.py:compute_chamfer_distance() exactly:
          - 8192 surface samples on each mesh
          - Bidirectional: mean(d²(pred→gt)) + mean(d²(gt→pred))  ← L2 Chamfer
        """
        try:
            from scipy.spatial import cKDTree
            pred_pts = pred_mesh.sample(n_points).astype(np.float32)
            gt_pts   = gt_mesh.sample(n_points).astype(np.float32)
            tree_gt   = cKDTree(gt_pts)
            tree_pred = cKDTree(pred_pts)
            d_pg, _ = tree_gt.query(pred_pts, k=1)
            d_gp, _ = tree_pred.query(gt_pts, k=1)
            return float(np.mean(np.square(d_pg)) + np.mean(np.square(d_gp)))
        except Exception:
            return None


    def run_worker(code_str, gt_mesh_path, compute_chamfer=False):
        import trimesh
        import cadquery as cq  # noqa: F401 (used implicitly via exec)

        g = {}
        exec(code_str, g)
        compound = g['r'].val()

        # Tessellate to mesh
        vertices, faces = compound.tessellate(0.001, 0.1)
        pred_mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
        assert len(pred_mesh.faces) > 2

        # Normalize: center → scale to unit extent → translate to [0.5, 0.5, 0.5]
        # Matches evaluate.py:run_cd_single() exactly.
        center = (pred_mesh.bounds[0] + pred_mesh.bounds[1]) / 2.0
        pred_mesh.apply_translation(-center)
        extent = np.max(pred_mesh.extents)
        if extent > 1e-7:
            pred_mesh.apply_scale(1.0 / extent)
        pred_mesh.apply_transform(
            trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        gt_mesh = trimesh.load_mesh(gt_mesh_path)
        iou = compute_iou(gt_mesh, pred_mesh)
        cd  = compute_cd(gt_mesh, pred_mesh) if compute_chamfer else None
        return iou, cd


    if __name__ == '__main__':
        payload = json.loads(sys.stdin.read())
        try:
            iou, cd = run_worker(
                payload['code_str'],
                payload['gt_mesh_path'],
                compute_chamfer=payload.get('compute_chamfer', False))
            print(json.dumps({'iou': iou, 'cd': cd, 'error': None}))
        except Exception as e:
            print(json.dumps({'iou': None, 'cd': None, 'error': str(e)}))
        sys.stdout.flush()
''')

# Module-level cache: written once per process lifetime
_worker_path: Optional[str] = None


def _get_worker_path() -> str:
    """Write the embedded worker script to a temp file on first call, reuse afterwards."""
    global _worker_path
    if _worker_path is not None and os.path.exists(_worker_path):
        return _worker_path
    fd, path = tempfile.mkstemp(suffix='.py', prefix='cq_reward_worker_')
    with os.fdopen(fd, 'w') as f:
        f.write(_WORKER_SCRIPT)
    _worker_path = path
    return path


def _execute_code_in_subprocess(
    code_str: str,
    gt_mesh_path: str,
    timeout: float = 10.0,
    compute_chamfer: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Execute CadQuery code in a fresh Python interpreter and return (iou, cd).

    Returns (IoU in [0,1], CD) or (None, None) on any failure.

    Rationale for subprocess over multiprocessing.Process:
    The main training process holds a CUDA context. Forking it via
    multiprocessing corrupts CUDA state. subprocess.run() spawns a clean
    interpreter without CUDA, so CadQuery + OCP run safely.
    """
    payload = json.dumps({
        'code_str': code_str,
        'gt_mesh_path': gt_mesh_path,
        'compute_chamfer': compute_chamfer,
    })
    try:
        proc = subprocess.run(
            [sys.executable, _get_worker_path()],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if not proc.stdout.strip():
            return None, None
        data = json.loads(proc.stdout.strip())
        return data.get('iou'), data.get('cd')
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_iou(gt_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh) -> Optional[float]:
    """Volumetric IoU between two trimesh objects.

    Copied from evaluate.py for use when both meshes are already loaded in
    the main process (e.g. during evaluation without CUDA).
    """
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return float(intersection_volume / union_volume)
    except Exception:
        return None


def compute_reward(code_str: str, gt_mesh_path: str, timeout: float = 10.0) -> float:
    """Compute IoU-based reward for a single generated code string.

    Formula:
        R(τ) = r_IoU + r_invalid
        r_IoU    = IoU × 10   (range [0, 10])
        r_invalid = -10 if code is invalid, 0 otherwise

    Returns a scalar in [-10, 10].
    """
    iou, _ = _execute_code_in_subprocess(code_str, gt_mesh_path, timeout=timeout)
    if iou is None:
        return -10.0          # r_invalid penalty
    return float(iou) * 10.0  # r_IoU in [0, 10]


def compute_metrics(
    code_str: str,
    gt_mesh_path: str,
    timeout: float = 30.0,
) -> Tuple[float, Optional[float]]:
    """Compute both IoU reward and Chamfer Distance for a single code string.

    Uses 8192 surface point samples for CD, matching evaluate.py exactly.

    Returns:
        (iou_reward, cd)
        iou_reward: float in [-10, 10]  (same scale as compute_reward)
        cd:         float or None       (None if code is invalid)
    """
    iou, cd = _execute_code_in_subprocess(
        code_str, gt_mesh_path, timeout=timeout, compute_chamfer=True)
    if iou is None:
        return -10.0, None
    return float(iou) * 10.0, cd


def compute_rewards_parallel(
    codes: List[str],
    gt_paths: List[str],
    workers: int = 4,
    timeout: float = 10.0,
) -> List[float]:
    """Compute rewards for multiple codes in parallel.

    Uses ThreadPoolExecutor: each thread spawns a subprocess, achieving
    true parallelism even under the GIL. Thread-level concurrency is
    sufficient because the actual computation runs in subprocesses.
    """
    assert len(codes) == len(gt_paths), "codes and gt_paths must have the same length"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(compute_reward, code, path, timeout)
            for code, path in zip(codes, gt_paths)
        ]
        return [f.result() for f in futures]
