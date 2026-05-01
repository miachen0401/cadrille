"""Single source for holdout-family list, BC val uid->family map, and IID/OOD
split helper. Used by all offline scripts in scripts/analysis/.

Usage:
    from common.holdout import HOLDOUT_FAMILIES, uid2fam, is_ood, split_label

The constants/maps are loaded once at import time. To change the holdout list,
edit configs/sft/holdout_families.yaml.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_CFG = REPO_ROOT / 'configs/sft/holdout_families.yaml'
_BC_VAL_PKL = REPO_ROOT / 'data/benchcad/val.pkl'


def _load() -> set[str]:
    if not _CFG.exists():
        return set()
    cfg = yaml.safe_load(_CFG.read_text()) or {}
    return set(cfg.get('holdout_families', []))


HOLDOUT_FAMILIES: set[str] = _load()


def _load_uid2fam() -> dict[str, str]:
    if not _BC_VAL_PKL.exists():
        return {}
    rows = pickle.load(_BC_VAL_PKL.open('rb'))
    return {r['uid']: r['family'] for r in rows}


uid2fam: dict[str, str] = _load_uid2fam()


def is_ood(uid: str, bucket: str = 'BenchCAD val') -> bool:
    """True iff uid is in a held-out family. Only meaningful for BenchCAD val."""
    if bucket != 'BenchCAD val':
        return False
    fam = uid2fam.get(uid)
    return fam in HOLDOUT_FAMILIES if fam else False


def split_label(uid: str, bucket: str = 'BenchCAD val') -> str:
    """Return '[OOD]' / '[IID]' tag for BC val uids; '' for other buckets."""
    if bucket != 'BenchCAD val':
        return ''
    fam = uid2fam.get(uid)
    if not fam:
        return ''
    return '[OOD]' if fam in HOLDOUT_FAMILIES else '[IID]'
