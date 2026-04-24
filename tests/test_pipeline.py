"""Regression tests for the full rendering + IoU pipeline.

Covers:
  RENDER  — render_img() produces correct-size, non-trivial image
  REWARD  — GT code on GT mesh → IoU ≥ 0.95 via compute_reward (subprocess)
  METRICS — compute_metrics IoU matches compute_reward IoU (within 0.01)
  EVAL_PY — evaluate.py IoU ≈ compute_reward IoU on same (code, mesh) pair
  COLLATE — render_img output has correct tensor shape after processor collate

Golden case: a unit cube.
  GT mesh  : trimesh-generated unit cube in [0,1]³ (written to a tmp STL)
  Pred code: `r = cq.Workplane("XY").box(1,1,1)` — should reproduce the cube exactly.

Both compute_reward and evaluate.py normalise meshes before IoU, so their
results should agree within floating-point + tessellation variance (≤ 0.02).

Run:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v -k "render"   # just render test
"""

import io
import os
import sys
import csv
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

_HAS_CADQUERY = False
try:
    import cadquery  # noqa: F401
    _HAS_CADQUERY = True
except Exception:
    pass

_HAS_OPEN3D_VIS = False
try:
    import open3d as _o3d
    _vis = _o3d.visualization.Visualizer()
    _vis.create_window(width=10, height=10, visible=False)
    _vis.destroy_window()
    _HAS_OPEN3D_VIS = True
except Exception:
    pass

requires_cadquery = pytest.mark.skipif(
    not _HAS_CADQUERY,
    reason='cadquery not importable (libGL / OCP missing)')

requires_render = pytest.mark.skipif(
    not _HAS_OPEN3D_VIS,
    reason='open3d Visualizer not working in this environment')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEEPCAD_TEST = os.path.join(_REPO_ROOT, 'data', 'deepcad_test_mesh')
_CAD_RECODE   = os.path.join(_REPO_ROOT, 'data', 'cad-recode-v1.5', 'train', 'batch_00')

requires_deepcad = pytest.mark.skipif(
    not os.path.isdir(_DEEPCAD_TEST),
    reason='deepcad_test_mesh not present')

requires_cadrecode = pytest.mark.skipif(
    not os.path.isdir(_CAD_RECODE),
    reason='cad-recode-v1.5 not present')


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def cube_stl(tmp_path_factory):
    """Write a unit-cube STL normalised to [0,1]³ — golden GT mesh."""
    tmp = tmp_path_factory.mktemp('golden')
    mesh = trimesh.creation.box([1.0, 1.0, 1.0])
    # center at (0.5, 0.5, 0.5) so it sits in [0,1]³
    mesh.apply_translation([0.5, 0.5, 0.5])
    path = str(tmp / 'cube_gt.stl')
    mesh.export(path)
    return path


# CadQuery code that reproduces a unit cube centred at the origin (same as box(1,1,1))
_CUBE_CODE = "import cadquery as cq\nr = cq.Workplane('XY').box(1, 1, 1)\n"
_INVALID_CODE = "this is not python"
_EMPTY_CODE = ""


# ---------------------------------------------------------------------------
# RENDER tests
# ---------------------------------------------------------------------------

class TestRender:

    @requires_render
    @requires_deepcad
    def test_render_img_shape(self):
        """render_img returns 268×268 RGB image (4-view grid with 3px borders)."""
        from common.meshio import render_img
        stl = os.path.join(_DEEPCAD_TEST, '00000093.stl')
        result = render_img(stl)
        assert 'video' in result
        img = result['video'][0]
        assert img.mode == 'RGB'
        assert img.size == (268, 268), f'unexpected size {img.size}'

    @requires_render
    @requires_deepcad
    def test_render_img_not_blank(self):
        """render_img output must not be all-white or all-black."""
        from common.meshio import render_img
        stl = os.path.join(_DEEPCAD_TEST, '00000093.stl')
        result = render_img(stl)
        arr = np.array(result['video'][0])
        # At least 5% of pixels should differ from white (255,255,255)
        white_frac = np.mean(np.all(arr == 255, axis=-1))
        assert white_frac < 0.95, f'image is {white_frac:.0%} white — render likely failed'
        black_frac = np.mean(np.all(arr == 0, axis=-1))
        assert black_frac < 0.95, f'image is {black_frac:.0%} black — render likely failed'

    @requires_render
    @requires_cadrecode
    def test_render_img_cadrecode_sample(self):
        """render_img works on a cad-recode STL (raw mm scale, not normalised)."""
        from common.meshio import render_img
        stl = os.path.join(_CAD_RECODE, '0.stl')
        result = render_img(stl)
        arr = np.array(result['video'][0])
        white_frac = np.mean(np.all(arr == 255, axis=-1))
        assert white_frac < 0.95


# ---------------------------------------------------------------------------
# REWARD / METRICS tests  (subprocess path)
# ---------------------------------------------------------------------------

class TestReward:

    @requires_cadquery
    def test_gt_cube_iou_high(self, cube_stl):
        """GT cube code on GT cube mesh → IoU ≥ 0.95 via compute_reward."""
        from common.metrics import compute_reward
        iou = compute_reward(_CUBE_CODE, cube_stl, timeout=30.0)
        assert iou >= 0.95, f'expected IoU ≥ 0.95, got {iou:.4f}'

    @requires_cadquery
    def test_invalid_code_returns_minus1(self, cube_stl):
        """Invalid Python code → reward = -1.0."""
        from common.metrics import compute_reward
        assert compute_reward(_INVALID_CODE, cube_stl) == -1.0

    @requires_cadquery
    def test_empty_code_returns_minus1(self, cube_stl):
        """Empty string → reward = -1.0."""
        from common.metrics import compute_reward
        assert compute_reward(_EMPTY_CODE, cube_stl) == -1.0

    @requires_cadquery
    def test_metrics_iou_matches_reward(self, cube_stl):
        """compute_metrics IoU == compute_reward IoU (same subprocess path)."""
        from common.metrics import compute_reward, compute_metrics
        iou_r = compute_reward(_CUBE_CODE, cube_stl, timeout=30.0)
        iou_m, cd = compute_metrics(_CUBE_CODE, cube_stl, timeout=30.0, use_pool=False)
        assert abs(iou_r - iou_m) < 0.01, (
            f'compute_reward={iou_r:.4f} vs compute_metrics={iou_m:.4f}')
        assert cd is not None, 'compute_metrics should return Chamfer Distance'
        assert cd >= 0.0

    @requires_cadquery
    @requires_cadrecode
    def test_gt_cadrecode_pair_iou_high(self):
        """GT .py code on its own .stl from cad-recode → IoU ≥ 0.95."""
        from common.metrics import compute_reward
        py_path  = os.path.join(_CAD_RECODE, '0.py')
        stl_path = os.path.join(_CAD_RECODE, '0.stl')
        code = open(py_path).read()
        iou = compute_reward(code, stl_path, timeout=30.0)
        assert iou >= 0.95, f'GT code on GT mesh gave IoU={iou:.4f} (expected ≥ 0.95)'


# ---------------------------------------------------------------------------
# EVAL_PY consistency test
# ---------------------------------------------------------------------------

class TestEvaluatePyConsistency:

    @requires_cadquery
    def test_evaluate_py_agrees_with_compute_reward(self, cube_stl, tmp_path):
        """evaluate.py IoU ≈ compute_reward IoU on the same (code, mesh) pair.

        Allowed difference: ≤ 0.02  (tessellation + normalisation floating-point).
        """
        from common.metrics import compute_reward

        # 1. IoU via compute_reward (reward.py subprocess)
        iou_reward = compute_reward(_CUBE_CODE, cube_stl, timeout=30.0)
        assert iou_reward >= 0.90, f'reward baseline too low: {iou_reward:.4f}'

        # 2. IoU via evaluate.py
        #    evaluate.py expects:
        #      --gt-mesh-path DIR  containing <stem>.stl  (pre-normalised [0,1]³)
        #      --pred-py-path DIR  containing <stem>+0.py
        gt_dir   = tmp_path / 'gt'
        pred_dir = tmp_path / 'pred'
        gt_dir.mkdir(); pred_dir.mkdir()

        stem = 'cube'
        shutil.copy(cube_stl, gt_dir / f'{stem}.stl')
        (pred_dir / f'{stem}+0.py').write_text(_CUBE_CODE)

        evaluate_py = os.path.join(_REPO_ROOT, 'evaluate.py')
        results_csv = str(tmp_path / 'results.csv')
        proc = subprocess.run(
            [sys.executable, evaluate_py,
             '--gt-mesh-path', str(gt_dir),
             '--pred-py-path', str(pred_dir),
             '--n-points', '4096',
             '--results-csv', results_csv],
            capture_output=True, text=True, timeout=60,
            cwd=_REPO_ROOT,
        )
        assert proc.returncode == 0, (
            f'evaluate.py failed:\nstdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}')
        assert os.path.exists(results_csv), 'evaluate.py did not write results.csv'

        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        assert rows, 'results.csv is empty'
        iou_eval = float(rows[0]['iou'])

        assert abs(iou_reward - iou_eval) <= 0.02, (
            f'compute_reward={iou_reward:.4f} vs evaluate.py={iou_eval:.4f} '
            f'(diff={abs(iou_reward-iou_eval):.4f}, limit=0.02)')
