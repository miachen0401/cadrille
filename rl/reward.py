"""Reward module — re-export shim.

The actual implementation lives in `common.metrics` so eval/ and tools/ can
use it without pulling in rl/. This shim keeps `from rl.reward import ...`
working during the refactor; drop it once every caller imports from
`common.metrics` directly (see step 6 of docs/repo_simplification.md).
"""
# Star-import re-exports every public name including the internal helpers
# (e.g. _get_worker_path, _reward_worker_run) that tools/ currently imports.
from common.metrics import *  # noqa: F401,F403
from common.metrics import (  # private names used by tools/infer_cases.py, etc.
    _get_worker_path,
    _reward_worker_init,
    _reward_worker_run,
    _eval_worker_init,
    _eval_worker_run,
    _execute_in_eval_pool,
    _execute_code_in_subprocess,
    _cache_key,
    _cache_get,
    _cache_set,
)
