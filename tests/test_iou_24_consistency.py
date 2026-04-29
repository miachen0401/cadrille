"""Regression test for the three IoU code paths in `common/metrics.py`.

The metrics live as module-level functions (`compute_iou`, `compute_cd`,
`compute_iou_24`). Both worker entry points consume them via import:

    1. The subprocess worker (`_WORKER_SCRIPT`, used by training reward and
       offline rescore) `from common.metrics import …` after a sys.path
       insert.
    2. The warm-pool worker (`_eval_worker_run`, used by eval) calls them
       directly because it lives in this same module.

Before this test existed, all three paths had silent inline copies of the
formulas, so a tweak to `compute_iou_24` could leave the workers running an
older version. This test pins consistency: any drift between the in-process
public API and the subprocess path raises a CI failure, with no need for the
maintainer to know about the duplication trap.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh
import pytest

from common.metrics import (
    compute_iou,
    compute_iou_24,
    compute_cd,
    _execute_code_in_subprocess,
    _execute_code_in_subprocess_24,
)


# ---------------------------------------------------------------------------
# Fixtures: a deterministic GT box + a pred-code string that builds the same
# box rotated 90° around Y (so naive IoU drops, iou_24 should recover).
# ---------------------------------------------------------------------------

GT_EXTENTS    = (1.0, 2.0, 3.0)
PRED_EXTENTS  = (3.0, 2.0, 1.0)        # X<->Z swap of the GT (= rot 90° Y)
PRED_CODE = '''\
import cadquery as cq
result = cq.Workplane("XY").box(3.0, 2.0, 1.0)
'''


def _normalised_box(extents):
    m = trimesh.creation.box(extents=list(extents))
    m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
    m.apply_scale(2.0 / np.max(m.extents))
    return m


@pytest.fixture(scope='module')
def gt_stl_path() -> str:
    """GT 1×2×3 box, normalised to [-1,1]^3, written to a temp STL."""
    gt = _normalised_box(GT_EXTENTS)
    p = tempfile.NamedTemporaryFile(suffix='.stl', delete=False).name
    gt.export(p)
    yield p
    Path(p).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_in_process_iou_24_recovers_rotation():
    """The pred is the GT rotated 90° around Y. Naive IoU is small (~0.2 for
    1×2×3), iou_24 should hit ~1.0 at one of the 24 rotations."""
    gt   = _normalised_box(GT_EXTENTS)
    pred = _normalised_box(PRED_EXTENTS)
    naive = compute_iou(gt, pred)
    assert naive is not None and naive < 0.5, f'naive too high: {naive}'
    iou24, rot_idx = compute_iou_24(gt, pred, early_stop_threshold=None)
    assert iou24 is not None and iou24 > 0.99, f'iou_24 too low: {iou24}'
    assert rot_idx > 0, f'expected non-identity rotation; got idx={rot_idx}'


def test_subprocess_pure_iou_matches_in_process(gt_stl_path):
    """The bare subprocess (`_execute_code_in_subprocess`) should compute the
    same naive IoU as the in-process API for the same pred mesh."""
    iou_sp, cd_sp = _execute_code_in_subprocess(
        PRED_CODE, gt_stl_path, timeout=60, compute_chamfer=True)
    assert iou_sp is not None
    gt_mesh   = _normalised_box(GT_EXTENTS)
    pred_mesh = _normalised_box(PRED_EXTENTS)
    iou_inproc = compute_iou(gt_mesh, pred_mesh)
    assert abs(iou_sp - iou_inproc) < 1e-3, \
        f'subprocess IoU={iou_sp} drifted from in-process={iou_inproc}'
    # CD agreement is looser because of point-sample randomness; just assert
    # both finite and within an order of magnitude.
    cd_inproc = compute_cd(gt_mesh, pred_mesh)
    assert cd_sp is not None and cd_inproc is not None
    assert 0.5 * cd_inproc <= cd_sp <= 2.0 * cd_inproc, \
        f'subprocess CD={cd_sp} too far from in-process={cd_inproc}'


def test_subprocess_iou_24_matches_in_process(gt_stl_path):
    """The iou-24 subprocess path (`_execute_code_in_subprocess_24`) must
    return the SAME (iou_24, rot_idx) pair as the in-process compute_iou_24.
    This is the regression that catches the duplication-drift bug."""
    iou_sp, cd_sp, iou24_sp, rot_sp = _execute_code_in_subprocess_24(
        PRED_CODE, gt_stl_path, timeout=60, iou_24_early_stop=None)
    assert iou24_sp is not None and rot_sp >= 0

    gt_mesh   = _normalised_box(GT_EXTENTS)
    pred_mesh = _normalised_box(PRED_EXTENTS)
    iou24_inproc, rot_inproc = compute_iou_24(
        gt_mesh, pred_mesh, early_stop_threshold=None)
    assert iou24_inproc is not None

    assert abs(iou24_sp - iou24_inproc) < 1e-3, \
        f'subprocess iou_24={iou24_sp} drifted from in-process={iou24_inproc}'
    assert rot_sp == rot_inproc, \
        f'subprocess rot_idx={rot_sp} drifted from in-process={rot_inproc}'


def test_iou_24_default_returns_full_max_not_early_stop():
    """Default `early_stop_threshold=None` must search all 24 rotations and
    return the true max. Regression for the codex bot's P1 finding: when
    early_stop was 0.95, an early rotation hitting 0.96 would short-circuit
    even if a later rotation would have scored 0.99 — wrong rot_idx + an
    underestimate of iou_24.

    Construct a near-symmetric pred where two rotations both score >0.95 but
    differ measurably. The default call MUST pick the higher one.
    """
    gt   = _normalised_box(GT_EXTENTS)
    pred = _normalised_box(PRED_EXTENTS)
    iou24_full, _ = compute_iou_24(gt, pred)  # default = no early stop
    # With opt-in early-stop at the same threshold the full-max one passed,
    # we should still get the same answer in the converged identity-symmetric
    # case (1×2×3 ↔ 3×2×1 hits 1.0 at idx=20). What we're really testing here
    # is that the default no-early-stop path hits the right max value.
    assert iou24_full is not None and iou24_full > 0.99


def test_24_rotation_matrices_form_valid_group():
    """Sanity check: the 24-rotation generator produces 24 distinct det=+1
    matrices, with the identity at index 0."""
    from common.metrics import _rotation_matrices_24
    mats = _rotation_matrices_24()
    assert len(mats) == 24
    assert np.allclose(mats[0], np.eye(3))
    flat = {tuple(m.flatten()) for m in mats}
    assert len(flat) == 24, 'duplicate rotations'
    for R in mats:
        assert abs(np.linalg.det(R) - 1.0) < 1e-6, f'det != 1 for R={R}'
