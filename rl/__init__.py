"""Top-level `rl` package — backwards-compat shim.

Real implementation lives at `train.rl` (moved in refactor step 4). This
shim pre-aliases every submodule under `sys.modules['rl.<name>']` so
`from rl.<name> import X` imports hit the already-cached module.

Drop once every caller migrates to `train.rl` or `common.*`
(step 6 of docs/repo_simplification.md).
"""
from __future__ import annotations

import importlib
import sys


# Make sure `rl` itself points at the train.rl package so attribute look-ups
# (e.g. `rl.dataset.X`) resolve without triggering a fresh import.
_trainrl = importlib.import_module('train.rl')
for attr in dir(_trainrl):
    if not attr.startswith('_'):
        globals()[attr] = getattr(_trainrl, attr)

# Pre-alias every submodule (including subpackages like rl.algorithms)
# so `from rl.dataset import X` reuses train.rl.dataset from sys.modules.
for _name in (
    'dataset', 'reward', 'eval', 'eval_passk', 'config',
    'mine', 'filter_scores', 'train', 'algorithms',
    'algorithms.cppo', 'algorithms.dpo',
):
    try:
        _mod = importlib.import_module(f'train.rl.{_name}')
    except ImportError:
        continue
    sys.modules[f'rl.{_name}'] = _mod
