"""Pass@k evaluation — re-export shim.

The single implementation now lives in `eval/passk.py`. This shim preserves
`from rl.eval_passk import ...` imports during the refactor and will be dropped
in step 6 (docs/repo_simplification.md).
"""
from eval.passk import (  # noqa: F401
    _pass_at_k,
    pass_at_k_mean,
    load_val_examples,
    _generate_one_batch,
    eval_passk,
    _find_checkpoints,
    main,
)
