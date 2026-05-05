"""Auto-load .env from repo root so entry points don't need `source .env` first.

Usage (always at the very top of a __main__ script):

    from common.env import load_repo_env
    load_repo_env()

Idempotent: safe to call many times. Uses os.environ.setdefault so an
explicit `export FOO=bar` from the shell still wins over the .env value.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOADED = False


def load_repo_env(env_path: str | os.PathLike | None = None) -> bool:
    """Read .env at repo root (or *env_path*) and setdefault into os.environ.

    Returns True if a file was found and parsed, False otherwise.
    """
    global _LOADED
    if _LOADED:
        return True
    p = Path(env_path) if env_path else (_REPO_ROOT / '.env')
    if not p.exists():
        return False
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        # Strip optional matching surrounding quotes
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)
    _LOADED = True
    return True
