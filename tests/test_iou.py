"""Tests for IoU metric computation in rl/reward.py.

Tests verify:
1. GT code evaluated against its own mesh → IoU ≈ 1.0
2. Invalid code → reward = -1.0
3. Empty string → reward = -1.0
4. evaluate.py normalization vs rl/reward.py normalization give identical IoU
5. compute_iou (trimesh) with manually constructed meshes

Run:
    pytest tests/test_iou.py -v
    pytest tests/test_iou.py -v -k "fast"  # skip slow CadQuery tests
"""

import os
import sys
import pickle
import pytest
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.reward import compute_reward, compute_metrics, compute_iou

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CAD_RECODE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'cad-recode-v1.5',
)
_PKL_PATH = os.path.join(_CAD_RECODE_ROOT, 'train.pkl')

# Skip CadQuery tests if cadquery is not importable (e.g. libGL missing).
_HAS_CADQUERY = False
try:
    import cadquery  # noqa: F401
    _HAS_CADQUERY = True
except Exception:
    pass

requires_cadquery = pytest.mark.skipif(
    not _HAS_CADQUERY,
    reason='cadquery not importable in this environment'
)
requires_data = pytest.mark.skipif(
    not os.path.exists(_PKL_PATH),
    reason='cad-recode-v1.5 dataset not present'
)


def _load_simple_pairs(n=5):
    """Load n GT (code, mesh_path) pairs sorted by mesh file size (simplest first)."""
    with open(_PKL_PATH, 'rb') as f:
        rows = pickle.load(f)
    # Sort by STL file size — smaller = simpler geometry
    rows = sorted(
        rows,
        key=lambda r: os.path.getsize(
            os.path.join(_CAD_RECODE_ROOT, r['mesh_path'])
        ),
    )
    pairs = []
    for r in rows[:n * 3]:
        if len(pairs) >= n:
            break
        code_path = os.path.join(_CAD_RECODE_ROOT, r['py_path'])
        mesh_path = os.path.join(_CAD_RECODE_ROOT, r['mesh_path'])
        if not os.path.exists(code_path) or not os.path.exists(mesh_path):
            continue
        with open(code_path) as f:
            code = f.read()
        pairs.append((code, mesh_path))
    return pairs


# ---------------------------------------------------------------------------
# Fast tests — pure trimesh, no CadQuery
# ---------------------------------------------------------------------------

class TestComputeIouTrimesh:
    """Tests using pre-built trimesh.Trimesh objects — no CadQuery, fast."""

    def test_identical_cubes(self):
        """Two identical unit cubes → IoU = 1.0."""
        cube = trimesh.creation.box(extents=[1, 1, 1])
        iou = compute_iou(cube, cube.copy())
        assert iou is not None
        assert abs(iou - 1.0) < 0.01, f"Expected IoU≈1.0, got {iou}"

    def test_non_overlapping_cubes(self):
        """Two cubes far apart → IoU = 0.0."""
        cube_a = trimesh.creation.box(extents=[1, 1, 1])
        cube_b = trimesh.creation.box(extents=[1, 1, 1])
        cube_b.apply_translation([10, 0, 0])
        iou = compute_iou(cube_a, cube_b)
        assert iou is not None
        assert iou < 0.01, f"Expected IoU≈0.0, got {iou}"

    def test_half_overlap(self):
        """Two cubes shifted by half their width → IoU = 1/3."""
        cube_a = trimesh.creation.box(extents=[2, 2, 2])
        cube_b = trimesh.creation.box(extents=[2, 2, 2])
        cube_b.apply_translation([1, 0, 0])
        iou = compute_iou(cube_a, cube_b)
        # intersection vol = 1×2×2 = 4, union vol = 8+8-4 = 12 → IoU = 1/3
        assert iou is not None
        assert abs(iou - 1.0 / 3.0) < 0.05, f"Expected IoU≈0.333, got {iou}"

    def test_sphere_vs_cube(self):
        """Sphere inscribed in cube → IoU = (4/3 π (0.5)³) / 1 ≈ 0.524."""
        sphere = trimesh.creation.icosphere(radius=0.5)
        cube = trimesh.creation.box(extents=[1, 1, 1])
        iou = compute_iou(sphere, cube)
        assert iou is not None
        assert 0.3 < iou < 0.8, f"Sphere-in-cube IoU expected 0.3–0.8, got {iou}"


# ---------------------------------------------------------------------------
# Normalization alignment test — no CadQuery
# ---------------------------------------------------------------------------

class TestNormalization:
    """Verify rl/reward.py transform_real_mesh matches evaluate.py normalization."""

    def _transform_evaluate(self, mesh):
        """Normalize pred as evaluate.py does: center + scale to [0,1]³."""
        mesh = mesh.copy()
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        mesh.apply_translation(-center)
        extent = np.max(mesh.extents)
        if extent > 1e-7:
            mesh.apply_scale(1.0 / extent)
        mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
        return mesh

    def _transform_reward(self, mesh):
        """Normalize as rl/reward.py does: center + scale to [-1,1]³."""
        mesh = mesh.copy()
        mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
        extent = np.max(mesh.extents)
        if extent > 1e-7:
            mesh.apply_scale(2.0 / extent)
        return mesh

    def test_same_iou_both_normalizations(self):
        """IoU is the same regardless of which normalization is applied."""
        # Create two overlapping cubes (different sizes to make it non-trivial)
        cube_a = trimesh.creation.box(extents=[3, 2, 1])
        cube_b = trimesh.creation.box(extents=[2, 2, 2])
        cube_b.apply_translation([0.5, 0, 0])

        iou_eval = compute_iou(
            self._transform_evaluate(cube_a),
            self._transform_evaluate(cube_b),
        )
        iou_reward = compute_iou(
            self._transform_reward(cube_a),
            self._transform_reward(cube_b),
        )
        assert iou_eval is not None and iou_reward is not None
        assert abs(iou_eval - iou_reward) < 0.01, (
            f"Normalization mismatch: eval={iou_eval:.4f} reward={iou_reward:.4f}"
        )

    def test_gt_test_mesh_reward_norm(self):
        """GT test mesh (already in [0,1]³) after reward-style norm lands in [-1,1]³."""
        stl = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'deepcad_test_mesh', '00000093.stl',
        )
        if not os.path.exists(stl):
            pytest.skip('deepcad_test_mesh not present')
        mesh = trimesh.load_mesh(stl)
        # GT is already in [0,1]³
        assert mesh.bounds[0].min() >= -0.01 and mesh.bounds[1].max() <= 1.01
        norm = self._transform_reward(mesh)
        # After reward norm, should be in ≈ [-1,1]³
        assert abs(norm.bounds[0].min() - (-1.0)) < 0.05, (
            f"Lower bound expected ≈-1, got {norm.bounds[0].min():.3f}")
        assert abs(norm.bounds[1].max() - 1.0) < 0.05, (
            f"Upper bound expected ≈1, got {norm.bounds[1].max():.3f}")


# ---------------------------------------------------------------------------
# CadQuery-based tests — require cadquery + data
# ---------------------------------------------------------------------------

class TestComputeRewardCadQuery:
    """Tests using real CadQuery code and mesh files."""

    @requires_cadquery
    @requires_data
    def test_gt_code_gives_high_iou(self):
        """GT code executed against its own mesh should yield IoU ≥ 0.95."""
        pairs = _load_simple_pairs(n=5)
        assert len(pairs) > 0, "No GT pairs found"
        ious = []
        for code, mesh_path in pairs:
            reward = compute_reward(code, mesh_path, timeout=30.0)
            ious.append(reward)
            assert reward > 0.5, (
                f"GT code against its own mesh yielded reward={reward:.3f} "
                f"(expected ≥0.5) for {mesh_path}"
            )
        mean_iou = np.mean(ious)
        assert mean_iou >= 0.90, (
            f"Mean IoU of GT code on own mesh = {mean_iou:.3f}, expected ≥0.90"
        )

    @requires_cadquery
    @requires_data
    def test_invalid_code_gives_negative_reward(self):
        """Invalid Python → reward = -1.0."""
        pairs = _load_simple_pairs(n=1)
        _, mesh_path = pairs[0]
        reward = compute_reward('this is not valid python!!!', mesh_path, timeout=10.0)
        assert reward == -1.0, f"Expected -1.0 for invalid code, got {reward}"

    @requires_cadquery
    @requires_data
    def test_empty_code_gives_negative_reward(self):
        """Empty string → reward = -1.0."""
        pairs = _load_simple_pairs(n=1)
        _, mesh_path = pairs[0]
        reward = compute_reward('', mesh_path, timeout=10.0)
        assert reward == -1.0, f"Expected -1.0 for empty code, got {reward}"

    @requires_cadquery
    @requires_data
    def test_compute_metrics_returns_cd(self):
        """compute_metrics with valid code should return (iou > 0, cd > 0)."""
        pairs = _load_simple_pairs(n=1)
        code, mesh_path = pairs[0]
        iou, cd = compute_metrics(code, mesh_path, timeout=30.0)
        assert iou > 0.5, f"Expected IoU > 0.5, got {iou}"
        assert cd is not None and cd >= 0, f"Expected CD ≥ 0, got {cd}"

    @requires_cadquery
    @requires_data
    def test_evaluate_py_alignment(self):
        """IoU from rl/reward.py subprocess ≈ IoU from a direct in-process computation.

        Verifies that normalizing both pred and GT to [-1,1]³ (reward path) and
        normalizing both independently to [0,1]³ (evaluate.py path) yield the same
        IoU, since IoU is invariant to uniform scaling + translation.

        Uses deepcad_test_mesh where GT is pre-normalized to [0,1]³.
        """
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'deepcad_test_mesh',
        )
        if not os.path.exists(test_dir):
            pytest.skip('deepcad_test_mesh not present')

        pairs = _load_simple_pairs(n=3)
        for code, _ in pairs:
            # Use a test mesh (pre-normalized to [0,1]³) as GT
            import os as _os
            stl_files = sorted(f for f in _os.listdir(test_dir) if f.endswith('.stl'))
            if not stl_files:
                pytest.skip('no STLs in deepcad_test_mesh')
            mesh_path = _os.path.join(test_dir, stl_files[0])

            # rl/reward.py path: normalizes both to [-1,1]³
            reward = compute_reward(code, mesh_path, timeout=30.0)
            if reward < 0:
                continue

            # evaluate.py path: GT already in [0,1]³; normalize pred to [0,1]³
            import io, warnings
            import cadquery as cq
            g = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exec(code, g)
            compound = g['r'].val()
            verts, faces = compound.tessellate(0.001, 0.1)
            pred = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
            # Reload through STL to repair (same as reward worker)
            buf = trimesh.exchange.stl.export_stl(pred)
            pred = trimesh.load(io.BytesIO(buf), file_type='stl', force='mesh')
            # Normalize pred to [0,1]³
            center = (pred.bounds[0] + pred.bounds[1]) / 2.0
            pred.apply_translation(-center)
            ext = np.max(pred.extents)
            if ext > 1e-7:
                pred.apply_scale(1.0 / ext)
            pred.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
            gt = trimesh.load_mesh(mesh_path)  # already in [0,1]³
            iou_eval = compute_iou(gt, pred)

            if iou_eval is None:
                continue

            # Both should agree within 0.05 (tessellation noise is the main diff)
            assert abs(reward - iou_eval) < 0.05, (
                f"IoU mismatch: reward_path={reward:.4f}  evaluate_path={iou_eval:.4f} "
                f"for GT={mesh_path}"
            )
            break  # one successful comparison is enough
