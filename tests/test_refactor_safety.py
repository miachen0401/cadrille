"""Refactor safety guardrail — keep green on every refactor step.

Two concerns locked down here:
1. *Import smoke* — every package/module in the repo must import. Catches broken
   moves, missing __init__.py, circular imports, typos in `from … import …`.
2. *passk math* — two implementations of the unbiased pass@k estimator exist
   (eval/passk.py and rl/eval_passk.py). Locking the boundary cases here means
   a subsequent merge cannot silently change semantics.

Runs in ~2 s. No GPU, no cadquery exec. Faster tests belong here; anything
slow or GPU-bound lives in the existing test_iou / test_pipeline / test_cppo.
"""
from __future__ import annotations
import importlib
import math
import pkgutil

import pytest


# ── 1. Import smoke ──────────────────────────────────────────────────────────

_PACKAGES_TO_WALK = ["train", "eval", "tools", "common", "data_prep", "bench", "experiments", "scripts"]
_EXCLUDE = {
    # Stand-alone scripts with heavy imports that are out-of-scope for the
    # refactor. Re-evaluate if they become library code.
    "tools.check_env.check_model",                 # loads Qwen2-VL-2B
    "experiments.data_prep_cadlib.deepcad2mesh",   # requires `cadlib`
    "experiments.cadevolve.eval",                  # requires `pyvista`
    "experiments.cadevolve.render",                # requires `pyvista`
}


def _discover_modules() -> list[str]:
    mods: list[str] = []
    for pkg_name in _PACKAGES_TO_WALK:
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue  # package may not exist yet (e.g. common/ before step 1)
        if not hasattr(pkg, "__path__"):
            mods.append(pkg_name)
            continue
        for m in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
            if m.name in _EXCLUDE:
                continue
            mods.append(m.name)
    mods.extend(["dataset", "cadrille"])  # top-level modules
    return mods


@pytest.mark.parametrize("mod", _discover_modules())
def test_imports(mod: str) -> None:
    importlib.import_module(mod)


# ── 2. passk math ────────────────────────────────────────────────────────────

def _try_pass_at_k():
    """Find whichever _pass_at_k exists — the two will be merged later."""
    try:
        from eval.passk import _pass_at_k, pass_at_k_mean
        return _pass_at_k, pass_at_k_mean, "eval.passk"
    except ImportError:
        pass
    from rl.eval_passk import _pass_at_k, pass_at_k_mean  # type: ignore
    return _pass_at_k, pass_at_k_mean, "rl.eval_passk"


def test_passk_boundaries() -> None:
    pak, _, _ = _try_pass_at_k()

    # n < k → undefined (NaN)
    assert math.isnan(pak(3, 1, 5))
    assert math.isnan(pak(0, 0, 1))

    # All samples pass (c == n) → 1.0
    assert pak(5, 5, 1) == 1.0
    assert pak(5, 5, 5) == 1.0

    # c == 0 → 0.0 exactly (nothing correct, can't sample a correct one)
    assert pak(10, 0, 1) == 0.0
    assert pak(10, 0, 5) == 0.0

    # n - c < k → 1.0 (any k-subset must hit a correct sample)
    # n=10, c=8, k=3 → n-c=2 < 3 → 1.0
    assert pak(10, 8, 3) == 1.0

    # k=1 reduces to c/n
    assert pak(10, 3, 1) == pytest.approx(0.3)
    assert pak(100, 47, 1) == pytest.approx(0.47)


def test_passk_mid_range() -> None:
    """Chen et al. 2021: 1 - C(n-c, k) / C(n, k)."""
    pak, _, _ = _try_pass_at_k()

    # n=10, c=5, k=3 — numeric sanity
    # 1 - prod_{i=0..2}((n-c-i)/(n-i)) = 1 - (5/10 * 4/9 * 3/8)
    expected = 1.0 - (5 / 10 * 4 / 9 * 3 / 8)
    assert pak(10, 5, 3) == pytest.approx(expected)

    # n=20, c=2, k=5
    p = 1.0 - (18 / 20 * 17 / 19 * 16 / 18 * 15 / 17 * 14 / 16)
    assert pak(20, 2, 5) == pytest.approx(p)


def test_passk_mean_skips_nan() -> None:
    _, mean, _ = _try_pass_at_k()

    # Three items: two valid (k<=n), one NaN (k>n)
    n_list = [10, 10, 3]
    c_list = [3, 0, 1]
    k = 5
    # item 0 : n=10 c=3 k=5 → valid
    # item 1 : n=10 c=0 k=5 → 0.0
    # item 2 : n=3  c=1 k=5 → NaN (skipped)
    pak, _, _ = _try_pass_at_k()
    v0 = pak(10, 3, 5)
    v1 = pak(10, 0, 5)
    assert math.isfinite(v0) and math.isfinite(v1)
    assert mean(n_list, c_list, k) == pytest.approx((v0 + v1) / 2)

    # All-NaN → NaN
    assert math.isnan(mean([1, 2], [0, 1], 5))
