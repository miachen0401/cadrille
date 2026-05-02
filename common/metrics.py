"""Mesh-to-mesh metrics + worker pools shared by train/, eval/, and tools/.

Originally lived at rl/reward.py. Moved here so eval/ can import without
depending on rl/. rl/reward.py re-exports all public names as a shim during
the refactor; drop the shim once no caller references `rl.reward.*`.

Two execution paths:

  Training (hot path, `compute_rewards_parallel`):
    subprocess.run() per call + ThreadPoolExecutor.
    Fast; 30 s timeout; CadQuery crash → only that job fails.

  Eval (reliability path, `compute_metrics`):
    Warm ProcessPoolExecutor(spawn) — workers pre-import cadquery/trimesh
    once, run at nice=10, execute with SIGALRM timeout (Fix 2+3).
    Results cached by SHA-256(code)+mesh_path so the same model
    never re-executes the same code (Fix 4).
    Falls back to subprocess if pool was never initialised.

Uses subprocess.run() for training because:
  1. CadQuery memory leaks stay in the subprocess (not the training process)
  2. The CUDA context in the main training process is not corrupted by fork()

Reward formula (from paper):
    R(τ) = IoU        if code is valid    (range [0, 1])
    R(τ) = -1         if code is invalid
"""

import os
import sys
import json
import hashlib
import textwrap
import tempfile
import subprocess
import multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Optional, Tuple

# Allow standalone execution from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Training worker (subprocess-per-call)
# ---------------------------------------------------------------------------

# `_WORKER_SCRIPT` runs in a fresh Python interpreter — no CUDA context, no
# memory leaks. It is intentionally thin: the *metrics* live as top-level
# functions in this same module (compute_iou / compute_cd / compute_iou_24)
# and the worker imports them so there is a single source of truth.
#
# Build the script as `header_with_repo_path + body` so the body can be a
# plain triple-string (no f-string {} escaping needed for the algo code).
_REPO_ROOT_FOR_WORKER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_WORKER_HEADER = textwrap.dedent(f'''\
    """CadQuery mesh executor — spawned by rl/reward.py as a subprocess.

    Metrics (compute_iou, compute_cd, compute_iou_24) are imported from the
    parent project's common.metrics module — DO NOT redefine them here.
    """
    import sys
    import json
    sys.path.insert(0, {_REPO_ROOT_FOR_WORKER!r})
    import numpy as np
    from common.metrics import compute_iou, compute_cd, compute_iou_24

''')

_WORKER_BODY = textwrap.dedent('''\
    def run_worker(code_str, gt_mesh_path, compute_chamfer=False, iou_24=False,
                   iou_24_early_stop=None):
        import io
        import trimesh
        import cadquery as cq  # noqa: F401 (used implicitly via exec)

        # Stub BenchCAD-style show_object(obj) so it also captures the result
        # — benchcad training corpus ends with `show_object(result)` which would
        # otherwise NameError inside our exec context.
        _captured = {}
        g = {'show_object': lambda obj, *a, **kw: _captured.setdefault('r', obj)}
        exec(code_str, g)
        _res = g.get('r') or g.get('result') or _captured.get('r')
        if _res is None:
            raise KeyError("no 'r' or 'result' variable in generated code")
        compound = _res.val()

        # Tessellate to mesh
        vertices, faces = compound.tessellate(0.001, 0.1)
        pred_mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
        assert len(pred_mesh.faces) > 2

        # Export → reload through STL to trigger trimesh's repair pipeline
        # (merge vertices, fix winding, fill holes → watertight mesh).
        # evaluate.py does the same via disk; we do it in memory.
        # Without this, complex .union() tessellations are non-manifold and
        # trimesh's boolean intersection silently returns None → iou=None.
        buf = trimesh.exchange.stl.export_stl(pred_mesh)
        pred_mesh = trimesh.load(io.BytesIO(buf), file_type='stl', force='mesh')

        def transform_real_mesh(mesh):
            """Center and scale mesh to [-1, 1]^3 (matches ref transform_real_mesh)."""
            mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
            extent = np.max(mesh.extents)
            if extent > 1e-7:
                mesh.apply_scale(2.0 / extent)
            return mesh

        pred_mesh = transform_real_mesh(pred_mesh)
        gt_mesh = transform_real_mesh(trimesh.load_mesh(gt_mesh_path))
        iou = compute_iou(gt_mesh, pred_mesh)
        cd  = compute_cd(gt_mesh, pred_mesh) if compute_chamfer else None
        iou24, rot_idx = (None, -1)
        if iou_24:
            iou24, rot_idx = compute_iou_24(
                gt_mesh, pred_mesh, early_stop_threshold=iou_24_early_stop)
        return iou, cd, iou24, rot_idx


    if __name__ == '__main__':
        payload = json.loads(sys.stdin.read())
        try:
            iou, cd, iou24, rot_idx = run_worker(
                payload['code_str'],
                payload['gt_mesh_path'],
                compute_chamfer=payload.get('compute_chamfer', False),
                iou_24=payload.get('iou_24', False),
                iou_24_early_stop=payload.get('iou_24_early_stop', None))
            print(json.dumps({'iou': iou, 'cd': cd,
                              'iou_24': iou24, 'rot_idx': rot_idx,
                              'error': None}))
        except Exception as e:
            print(json.dumps({'iou': None, 'cd': None,
                              'iou_24': None, 'rot_idx': -1,
                              'error': str(e)}))
        sys.stdout.flush()
''')

# Final script written to the temp file = header (with sys.path + imports)
# followed by the body (run_worker + main entry).
_WORKER_SCRIPT = _WORKER_HEADER + _WORKER_BODY


# Module-level cache: written once per process lifetime
_worker_path: Optional[str] = None
# Log first N worker errors to help diagnose reward = -1 issues
_error_log_count = 0
_MAX_ERROR_LOGS = 5


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


def _execute_code_in_subprocess_24(
    code_str: str,
    gt_mesh_path: str,
    timeout: float = 300.0,
    iou_24_early_stop: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """Like `_execute_code_in_subprocess` but also returns rotation-invariant IoU.

    Returns (iou, cd, iou_24, rot_idx). iou_24 is the max volumetric IoU over
    the 24 cube rotations of pred_mesh; rot_idx ∈ [0, 23] is the winning
    rotation (0 ≡ identity). Use this for offline rescoring of saved
    generations where wall-clock matters less than score quality.

    Default timeout is 300s because 24 boolean intersections at ~1-3s each.
    """
    payload = json.dumps({
        'code_str': code_str,
        'gt_mesh_path': gt_mesh_path,
        'compute_chamfer': True,
        'iou_24': True,
        'iou_24_early_stop': iou_24_early_stop,
    })
    global _error_log_count
    try:
        proc = subprocess.run(
            [sys.executable, _get_worker_path()],
            input=payload, capture_output=True, text=True, timeout=timeout,
        )
        if not proc.stdout.strip():
            return None, None, None, -1
        data = json.loads(proc.stdout.strip())
        return (data.get('iou'), data.get('cd'),
                data.get('iou_24'), data.get('rot_idx', -1))
    except subprocess.TimeoutExpired:
        return None, None, None, -1
    except Exception:
        return None, None, None, -1


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
    global _error_log_count
    try:
        proc = subprocess.run(
            [sys.executable, _get_worker_path()],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if not proc.stdout.strip():
            if _error_log_count < _MAX_ERROR_LOGS:
                _error_log_count += 1
                print(f'[reward worker] empty stdout (returncode={proc.returncode})'
                      f'\n  stderr: {proc.stderr[:300].strip()}', flush=True)
            return None, None
        data = json.loads(proc.stdout.strip())
        if data.get('error') and _error_log_count < _MAX_ERROR_LOGS:
            _error_log_count += 1
            print(f'[reward worker] error: {data["error"]}', flush=True)
        return data.get('iou'), data.get('cd')
    except subprocess.TimeoutExpired:
        if _error_log_count < _MAX_ERROR_LOGS:
            _error_log_count += 1
            print(f'[reward worker] timeout after {timeout}s', flush=True)
        return None, None
    except Exception as exc:
        if _error_log_count < _MAX_ERROR_LOGS:
            _error_log_count += 1
            print(f'[reward worker] exception: {exc}', flush=True)
        return None, None


# ---------------------------------------------------------------------------
# Fix 4: Result cache  (eval path only — training uses subprocess directly)
# ---------------------------------------------------------------------------

# Plain dict used as an ordered LRU (Python 3.7+ dicts preserve insertion order).
_RESULT_CACHE: dict = {}
_RESULT_CACHE_MAX = 512


def _cache_key(code_str: str, mesh_path: str, compute_chamfer: bool) -> str:
    digest = hashlib.sha256(code_str.encode('utf-8', errors='replace')).hexdigest()[:24]
    return f'{digest}|{os.path.abspath(mesh_path)}|{int(compute_chamfer)}'


def _cache_get(key: str) -> Optional[Tuple]:
    return _RESULT_CACHE.get(key)


def _cache_set(key: str, value: Tuple) -> None:
    if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
        oldest = next(iter(_RESULT_CACHE))
        del _RESULT_CACHE[oldest]
    _RESULT_CACHE[key] = value


# ---------------------------------------------------------------------------
# Warm reward process pool  (spawn, pre-imported deps, training hot path)
# ---------------------------------------------------------------------------
# Eliminates per-call Python startup + cadquery/trimesh import overhead (~1-2s each).
# Uses spawn (not fork) so CUDA context in parent is never inherited by workers.

_reward_pool: Optional[ProcessPoolExecutor] = None
_reward_pool_crashes: int = 0


def get_and_reset_pool_crashes() -> int:
    """Return pool crash count since last call and reset to 0."""
    global _reward_pool_crashes
    n, _reward_pool_crashes = _reward_pool_crashes, 0
    return n


def _reward_worker_init() -> None:
    """Pre-import heavy deps once per worker at pool startup."""
    import trimesh          # noqa: F401
    import cadquery         # noqa: F401
    try:
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError:
        pass


def _reward_worker_run(
    code_str: str,
    gt_mesh_path: str,
    timeout: float,
    soft_invalid: float = -1.0,
    family: Optional[str] = None,
    ess_mode: str = 'fractional',
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Run CadQuery + IoU inside a warm worker process.

    Returns:
        (iou, cd, ess_score)
          iou:      None on SyntaxError/timeout, soft_invalid on CQ/mesh fail,
                    else float in [0, 1].
          cd:       always None in this code path (legacy slot, kept for back-compat).
          ess_score: None when `family` lacks an essential_ops spec, else
                    float in [0, 1] computed from raw code via regex
                    (essential_score for ess_mode='fractional', binary 0/1
                    for ess_mode='binary'). Independent of CQ exec status —
                    even a soft_invalid run can carry a usable ess signal.

    SyntaxError → ess_score is also None (raw code is too garbled to trust).
    """
    import signal
    import io
    import warnings
    import numpy as np
    import trimesh
    import cadquery as cq  # noqa: F401

    def _on_alarm(signum, frame):
        raise TimeoutError(f'CadQuery exceeded {timeout:.0f}s')

    # Compute essential-ops score from RAW code via regex (no exec needed).
    # Done up-front so it survives SyntaxError / CQ failure paths.
    ess_score: Optional[float] = None
    if family:
        try:
            from common.essential_ops import (
                ESSENTIAL_BY_FAMILY,
                find_ops as _find_ops,
                essential_score as _ess_score,
                essential_pass as _ess_pass,
            )
            if family in ESSENTIAL_BY_FAMILY:
                _ops = _find_ops(code_str or '')
                if ess_mode == 'binary':
                    p = _ess_pass(family, _ops)
                    ess_score = None if p is None else (1.0 if p else 0.0)
                else:
                    ess_score = _ess_score(family, _ops)
        except Exception:
            ess_score = None

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(max(1, int(timeout) + 2))
    try:
        # Compile first to catch SyntaxErrors/SyntaxWarnings without exec noise.
        # SyntaxError = model produced completely garbled output → hard penalty.
        try:
            code_obj = compile(code_str, '<string>', 'exec')
        except SyntaxError:
            signal.alarm(0)
            # Drop ess_score on SyntaxError — raw text is too garbled to trust.
            return None, None, None
        # Code is syntactically valid — any failure from here gets soft_invalid.
        _captured = {}
        g = {'show_object': lambda obj, *a, **kw: _captured.setdefault('r', obj)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exec(code_obj, g)
        _res = g.get('r') or g.get('result') or _captured.get('r')
        if _res is None:
            raise KeyError("no 'r' or 'result' variable in generated code")
        compound = _res.val()
        vertices, faces = compound.tessellate(0.001, 0.1)
        pred_mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
        assert len(pred_mesh.faces) > 2
        buf = trimesh.exchange.stl.export_stl(pred_mesh)
        pred_mesh = trimesh.load(io.BytesIO(buf), file_type='stl', force='mesh')

        def _transform(mesh):
            mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
            ext = np.max(mesh.extents)
            if ext > 1e-7:
                mesh.apply_scale(2.0 / ext)
            return mesh

        pred_mesh = _transform(pred_mesh)
        gt_mesh   = _transform(trimesh.load_mesh(gt_mesh_path))
        iou = None
        try:
            intersection_volume = 0.0
            for gt_i in gt_mesh.split():
                for pred_i in pred_mesh.split():
                    sect = gt_i.intersection(pred_i)
                    intersection_volume += sect.volume if sect is not None else 0.0
            gt_vol  = sum(m.volume for m in gt_mesh.split())
            pred_vol = sum(m.volume for m in pred_mesh.split())
            union_vol = gt_vol + pred_vol - intersection_volume
            if union_vol > 0:
                iou = float(intersection_volume / union_vol)
        except Exception:
            pass
        signal.alarm(0)
        # iou=None means mesh existed but boolean failed → still better than syntax error
        return (iou if iou is not None else soft_invalid), None, ess_score
    except Exception:
        signal.alarm(0)
        # Runtime CQ error (e.g. invalid operation) → soft_invalid
        # ess_score (regex-based) survives this branch — code structure exists.
        return soft_invalid, None, ess_score


def shutdown_pools() -> None:
    """Shut down reward + eval pools and free their RSS (~400 MB per worker).

    Call before model.cpu() during img eval to reclaim CPU RAM on memory-
    constrained systems (4080 / 16 GB).  Pools are NOT restarted afterwards;
    subsequent reward/eval calls fall back to fresh subprocesses automatically.
    """
    global _reward_pool, _eval_pool
    for pool, name in [(_reward_pool, 'reward'), (_eval_pool, 'eval')]:
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    _reward_pool = None
    _eval_pool   = None


def init_reward_pool(n_workers: int = 8) -> None:
    """Spawn the warm reward process pool (call once at training startup).

    Workers pre-import cadquery/trimesh so per-call overhead is just computation,
    not Python startup + library import (~1-2s savings per rollout).
    Idempotent: safe to call multiple times.
    """
    global _reward_pool
    if _reward_pool is not None:
        return
    ctx = multiprocessing.get_context('spawn')
    _reward_pool = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_reward_worker_init,
    )
    print(f'[reward pool] Started {n_workers} warm worker(s) (spawn)', flush=True)


# ---------------------------------------------------------------------------
# Fix 2+3: Warm eval process pool  (spawn, nice=10, pre-imported deps)
# ---------------------------------------------------------------------------

_eval_pool: Optional[ProcessPoolExecutor] = None


def _eval_worker_init() -> None:
    """Initialiser run once per worker process at pool startup.

    Fix 3: Sets low CPU priority (nice=10) so eval workers don't starve
    the training reward workers.
    Pre-imports cadquery/trimesh/scipy to warm up the worker (Fix 2).
    """
    os.nice(10)
    # Pre-import heavy deps so the first real job has no cold-start latency
    import trimesh          # noqa: F401
    import cadquery         # noqa: F401
    try:
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError:
        pass


def _eval_worker_run(
    code_str: str,
    gt_mesh_path: str,
    timeout: float,
    compute_chamfer: bool,
) -> Tuple[Optional[float], Optional[float]]:
    """Execute CadQuery code inside a warm worker process.

    Uses SIGALRM (Linux/macOS) for hard timeout so the worker returns
    promptly on slow shapes and becomes immediately available for the next job.
    """
    import signal
    import io
    import warnings
    import numpy as np
    import trimesh
    import cadquery as cq  # noqa: F401 (used implicitly via exec)

    def _on_alarm(signum, frame):
        raise TimeoutError(f'CadQuery exceeded {timeout:.0f}s')

    signal.signal(signal.SIGALRM, _on_alarm)
    # +2 s buffer: SIGALRM fires slightly after the stated timeout so that
    # the caller's future.result(timeout+5) guard never triggers first.
    signal.alarm(max(1, int(timeout) + 2))

    try:
        try:
            code_obj = compile(code_str, '<string>', 'exec')
        except SyntaxError:
            signal.alarm(0)
            return None, None
        _captured = {}
        g = {'show_object': lambda obj, *a, **kw: _captured.setdefault('r', obj)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exec(code_obj, g)
        _res = g.get('r') or g.get('result') or _captured.get('r')
        if _res is None:
            raise KeyError("no 'r' or 'result' variable in generated code")
        compound = _res.val()

        vertices, faces = compound.tessellate(0.001, 0.1)
        pred_mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
        assert len(pred_mesh.faces) > 2

        buf = trimesh.exchange.stl.export_stl(pred_mesh)
        pred_mesh = trimesh.load(io.BytesIO(buf), file_type='stl', force='mesh')

        def _transform(mesh):
            mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
            ext = np.max(mesh.extents)
            if ext > 1e-7:
                mesh.apply_scale(2.0 / ext)
            return mesh

        pred_mesh = _transform(pred_mesh)
        gt_mesh   = _transform(trimesh.load_mesh(gt_mesh_path))

        # Single source of truth: call the module-level metric implementations
        # rather than re-inlining the formulas (used to be a third silent copy).
        iou = compute_iou(gt_mesh, pred_mesh)
        cd  = compute_cd(gt_mesh, pred_mesh) if (compute_chamfer and iou is not None) else None

        signal.alarm(0)
        return iou, cd

    except Exception:
        signal.alarm(0)
        return None, None


def init_eval_pool(n_workers: int = 2) -> None:
    """Spawn the warm eval process pool (call once at training startup).

    Workers pre-import cadquery/trimesh at startup (Fix 2) and run at
    nice=10 (Fix 3).  Uses 'spawn' to avoid inheriting the parent's
    CUDA context, GPU memory allocation, or file descriptors.

    Idempotent: safe to call multiple times.
    """
    global _eval_pool
    if _eval_pool is not None:
        return
    ctx = multiprocessing.get_context('spawn')
    _eval_pool = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_eval_worker_init,
    )
    print(f'[eval pool] Started {n_workers} warm worker(s) (nice=10, spawn)', flush=True)


def _execute_in_eval_pool(
    code_str: str,
    gt_mesh_path: str,
    timeout: float,
    compute_chamfer: bool,
) -> Tuple[Optional[float], Optional[float]]:
    """Execute via warm pool with cache; fall back to subprocess if pool missing."""
    # Fix 4: cache lookup
    key = _cache_key(code_str, gt_mesh_path, compute_chamfer)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    if _eval_pool is None:
        # Pool not initialised (e.g. standalone script) — use subprocess
        result = _execute_code_in_subprocess(code_str, gt_mesh_path, timeout, compute_chamfer)
        _cache_set(key, result)
        return result

    try:
        future = _eval_pool.submit(
            _eval_worker_run, code_str, gt_mesh_path, timeout, compute_chamfer)
        # +5 s over the stated timeout: SIGALRM fires at timeout+2, worker
        # catches it and returns None; this outer guard is a safety net only.
        result = future.result(timeout=timeout + 5)
    except Exception:
        result = (None, None)

    _cache_set(key, result)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_iou(gt_mesh, pred_mesh) -> Optional[float]:
    """Volumetric IoU between two trimesh objects.

    Single source of truth for the IoU formula. The subprocess worker
    (`_WORKER_SCRIPT`) and the warm-pool worker (`_eval_worker_run`) both
    call back into this function instead of re-implementing it.
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


def compute_cd(gt_mesh, pred_mesh, n_points: int = 8192) -> Optional[float]:
    """Bidirectional L2 Chamfer Distance via 8192 surface samples per mesh.

    Single source of truth — both subprocess and warm-pool workers import this
    rather than re-implementing it. Matches evaluate.py:compute_chamfer_distance.
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


def _rotation_matrices_24() -> List[np.ndarray]:
    """The 24 rotational symmetries of an axis-aligned cube as 3x3 matrices.

    Equivalent to all signed axis-permutation matrices with determinant +1.
    Index 0 is always the identity. Cached on first call.
    """
    cache = getattr(_rotation_matrices_24, '_cache', None)
    if cache is not None:
        return cache
    from itertools import permutations, product
    mats: List[np.ndarray] = [np.eye(3)]  # identity first so idx 0 ≡ no rotation
    seen = {tuple(np.eye(3).flatten())}
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[i, p] = signs[i]
            if abs(np.linalg.det(R) - 1.0) >= 1e-6:
                continue
            key = tuple(R.flatten())
            if key in seen:
                continue
            seen.add(key)
            mats.append(R)
    assert len(mats) == 24, f'expected 24 rotations, got {len(mats)}'
    _rotation_matrices_24._cache = mats  # type: ignore[attr-defined]
    return mats


def compute_iou_24(
    gt_mesh,
    pred_mesh,
    early_stop_threshold: Optional[float] = None,
) -> Tuple[Optional[float], int]:
    """Rotation-invariant IoU under the 24 cube symmetries.

    Tries each of the 24 axis-aligned rotations on pred_mesh and returns the
    maximum volumetric IoU vs gt_mesh, plus the index of the winning rotation
    (0 = identity, 1..23 = the 90°/180°/270° axis-permutations + sign flips).

    Use this when the prediction may be a correct shape but rotated by a
    multiple of 90° on some axis — common for CAD generations whose output
    base_plane / orientation drifts from GT.

    Both meshes should already be centred at the origin and scaled to a
    common cube (the existing `transform_real_mesh` normalisation to
    [-1, 1]^3 satisfies this); rotating around the origin then keeps the
    mesh inside the same cube.

    Args:
        early_stop_threshold: optional. If set, abandons the rotation search
            as soon as some rotation reaches this IoU. THIS IS LOSSY — the
            returned value is the *first* IoU above the threshold, not the
            true maximum, and `best_rotation_idx` may be wrong if a later
            rotation would have scored higher. Default `None` = full search,
            so the function honours its "max over 24" contract by default;
            opt in only when you genuinely value wall-clock over correctness
            (e.g. training reward where the gradient signal of "good enough"
            is what matters).

    Returns:
        (best_iou, best_rotation_idx). best_iou is None if every rotation's
        boolean intersection failed (non-manifold mesh, etc.); idx is -1 in
        that case.
    """
    best_iou: Optional[float] = None
    best_idx = -1
    for i, R in enumerate(_rotation_matrices_24()):
        if i == 0:
            pred_rot = pred_mesh
        else:
            pred_rot = pred_mesh.copy()
            T = np.eye(4)
            T[:3, :3] = R
            pred_rot.apply_transform(T)
        iou = compute_iou(gt_mesh, pred_rot)
        if iou is None:
            continue
        if best_iou is None or iou > best_iou:
            best_iou = iou
            best_idx = i
            if early_stop_threshold is not None and best_iou >= early_stop_threshold:
                break
    return best_iou, best_idx


def compute_reward(code_str: str, gt_mesh_path: str, timeout: float = 10.0) -> float:
    """Compute IoU-based reward for a single generated code string.

    Returns IoU in [0, 1] on success, or -1.0 on invalid/failed code.
    """
    iou, _ = _execute_code_in_subprocess(code_str, gt_mesh_path, timeout=timeout)
    if iou is None:
        return -1.0
    if float(iou) < 0:
        return 0.0
    return float(iou)


def compute_metrics_24(
    code_str: str,
    gt_mesh_path: str,
    timeout: float = 300.0,
    iou_24_early_stop: Optional[float] = None,
) -> Tuple[float, Optional[float], Optional[float], int]:
    """Compute (iou_naive, cd, iou_24, rot_idx) for one sample.

    iou_naive matches `compute_metrics` exactly (no rotation). iou_24 is the
    max IoU over the 24 axis-aligned rotations of pred_mesh; rot_idx is the
    winning rotation index (0 ≡ identity). Returned IoU values follow the
    same convention as compute_metrics: -1.0 on subprocess/exec failure,
    0.0 on zero-overlap, otherwise the float in [0, 1].

    iou_24_early_stop defaults to None (full search) — see compute_iou_24
    for the lossy-but-faster opt-in semantics.
    """
    iou, cd, iou_24, rot_idx = _execute_code_in_subprocess_24(
        code_str, gt_mesh_path,
        timeout=timeout, iou_24_early_stop=iou_24_early_stop)
    iou_out    = -1.0 if iou    is None else max(0.0, float(iou))
    iou24_out  = None if iou_24 is None else max(0.0, float(iou_24))
    return iou_out, cd, iou24_out, rot_idx


def compute_metrics(
    code_str: str,
    gt_mesh_path: str,
    timeout: float = 30.0,
    use_pool: bool = False,
) -> Tuple[float, Optional[float]]:
    """Compute both IoU reward and Chamfer Distance for a single code string.

    Args:
        use_pool: If True, use the warm eval pool + result cache (Fix 2/4).
                  Set True for eval; leave False for training reward computation.
                  Falls back to subprocess if init_eval_pool() was never called.

    Returns:
        (iou_reward, cd)
        iou_reward: float in [0, 1] on success, -1.0 on failure
        cd:         float or None  (None if code is invalid)
    """
    if use_pool:
        iou, cd = _execute_in_eval_pool(code_str, gt_mesh_path, timeout, compute_chamfer=True)
    else:
        iou, cd = _execute_code_in_subprocess(
            code_str, gt_mesh_path, timeout=timeout, compute_chamfer=True)
    if iou is None:
        return -1.0, None
    if float(iou) < 0:
        return 0.0, cd
    return float(iou), cd


def compute_rewards_parallel(
    codes: List[str],
    gt_paths: List[str],
    workers: int = 4,
    timeout: float = 10.0,
    soft_invalid: float = -1.0,
    families: Optional[List[Optional[str]]] = None,
    ess_mode: str = 'fractional',
    return_pairs: bool = False,
):
    """Compute rewards for multiple codes in parallel.

    Uses the warm reward pool if init_reward_pool() was called — eliminates
    per-call Python startup + cadquery/trimesh import overhead (~1-2s each).
    Falls back to ThreadPoolExecutor + fresh subprocesses if pool is unavailable
    or if a worker crashes (BrokenProcessPool). Automatically restarts the pool
    after a crash so future batches are warm again.

    Args:
        soft_invalid: Reward for code that is syntactically valid but fails at
            CadQuery build/mesh level.  Default -1.0 (backward-compat).
        families:    Optional per-row family hint. When provided AND a row's
                     family has an `essential_ops` spec, the worker also returns
                     an ess_score for that row (else None). When omitted, ess
                     is None for every row — caller can ignore the second tuple
                     element to preserve pure-IoU behavior.
        ess_mode:    'fractional' (default) → essential_score in [0, 1];
                     'binary' → 0/1 from essential_pass.
        return_pairs: When False (default), returns a List[float] of IoU-only
                     rewards (back-compat with mining + tests + bench_workers).
                     When True, returns List[Tuple[float, Optional[float]]] of
                     (iou, ess_score) — caller composes the final scalar.

    Returns:
        Either List[float] (return_pairs=False, default) or
        List[Tuple[float, Optional[float]]] (return_pairs=True).
        IoU semantics: -1.0 on SyntaxError/timeout (hard penalty),
        soft_invalid (typically -1.0 or -0.5) on CQ/mesh runtime failure,
        else float in [0, 1]. ess_score (when paired): None when row's family
        lacks a spec or on SyntaxError; else float in [0, 1] (regex-based,
        survives CQ runtime failures).
    """
    from concurrent.futures.process import BrokenProcessPool

    assert len(codes) == len(gt_paths), "codes and gt_paths must have the same length"
    if families is not None:
        assert len(families) == len(codes), \
            "families must align with codes/gt_paths"
    else:
        families = [None] * len(codes)

    def _pack(iou: Optional[float], ess: Optional[float]):
        if iou is None:
            pair = (-1.0, None)        # SyntaxError / timeout — hard penalty, no ess
        else:
            pair = (float(iou), ess)   # IoU ∈ [0,1] on success; soft_invalid < 0 on CQ fail
        return pair if return_pairs else pair[0]

    global _reward_pool
    if _reward_pool is not None:
        try:
            futures = [
                _reward_pool.submit(
                    _reward_worker_run, code, path, timeout, soft_invalid, fam, ess_mode)
                for code, path, fam in zip(codes, gt_paths, families)
            ]
            results: list = []
            for f in futures:
                try:
                    iou, _cd, ess = f.result(timeout=timeout + 5)
                    results.append(_pack(iou, ess))
                except BrokenProcessPool:
                    raise  # bubble up to outer handler
                except Exception:
                    results.append((-1.0, None) if return_pairs else -1.0)
            return results
        except BrokenProcessPool:
            global _reward_pool_crashes
            _reward_pool_crashes += 1
            print('[reward pool] worker crashed — restarting pool, '
                  'falling back to subprocess for this batch', flush=True)
            try:
                _reward_pool.shutdown(wait=False)
            except Exception:
                pass
            _reward_pool = None
            init_reward_pool(n_workers=workers)  # warm pool for next batch
            # fall through to subprocess for this batch

    # Fallback: fresh subprocess per call (no ess support — pure IoU only).
    # Worker pool is the only path that computes ess; the subprocess fallback
    # is a rare crash-recovery path so losing ess for one batch is acceptable.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(compute_reward, code, path, timeout)
            for code, path in zip(codes, gt_paths)
        ]
        if return_pairs:
            return [(f.result(), None) for f in futures]
        return [f.result() for f in futures]
