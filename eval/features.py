"""Detect BenchCAD feature preservation in generated CadQuery code.

BenchCAD rows carry a `feature_tags` dict, e.g.
    {"has_hole": true, "has_fillet": false, "has_chamfer": false,
     "has_slot": false, "rotational": true}

For every feature flagged True in the GT, we check whether the prediction's
code contains the corresponding CadQuery op. The check is a coarse regex
match — good enough for a first pass. User signalled the hole detection
is the roughest of the bunch ("先用找有没有hole实现") and may be replaced
with a mesh-based detector later.

Usage:
    from eval.features import detect_features, feature_recall
    present = detect_features(pred_code)         # dict[str, bool]
    scores  = feature_recall(gt_tags, pred_code) # dict[feat, {'gt': bool, 'hit': bool}]
"""
from __future__ import annotations
import json
import re
from typing import Mapping

# Patterns chosen to tolerate whitespace + method-call syntax. All keyed by
# the feature name used in BenchCAD metadata.
_FEATURE_PATTERNS: dict[str, re.Pattern] = {
    # User: "先用找有没有hole实现". Coarse — any boolean cut or hole-op.
    'has_hole':    re.compile(r'\.(cut|hole|cboreHole|cskHole)\b'),
    'has_fillet':  re.compile(r'\.fillet\b'),
    'has_chamfer': re.compile(r'\.chamfer\b'),
    # slot: cadquery has .slot2D in sketch mode; sometimes people use .rect for slots,
    # so this recall will be conservative.
    'has_slot':    re.compile(r'\.(slot2D|slot)\b'),
    # rotational → revolve, sphere, cylinder (loose).
    'rotational':  re.compile(r'\.(revolve|sphere|cylinder)\b'),
}


def _parse_tags(feature_tags) -> dict[str, bool]:
    """BenchCAD stores feature_tags as JSON-encoded string. Normalize to dict[str, bool]."""
    if feature_tags is None:
        return {}
    if isinstance(feature_tags, str):
        try:
            feature_tags = json.loads(feature_tags)
        except json.JSONDecodeError:
            return {}
    if not isinstance(feature_tags, Mapping):
        return {}
    return {k: bool(v) for k, v in feature_tags.items()}


def detect_features(pred_code: str) -> dict[str, bool]:
    """Return {feature_name: code_contains_pattern} for every tracked feature."""
    return {name: bool(pat.search(pred_code)) for name, pat in _FEATURE_PATTERNS.items()}


def feature_recall(feature_tags, pred_code: str) -> dict[str, dict[str, bool]]:
    """Per-feature recall for one GT/pred pair.

    Returns {feat: {'gt': gt_present, 'hit': pred_has_it}} for every tracked feature.
    If the GT does not have the feature the 'hit' value is still reported (so callers
    can compute precision if desired), but the summary stage usually only averages
    over rows where gt == True.
    """
    tags = _parse_tags(feature_tags)
    present = detect_features(pred_code)
    out: dict[str, dict[str, bool]] = {}
    for name in _FEATURE_PATTERNS:
        out[name] = {'gt': tags.get(name, False), 'hit': present[name]}
    return out


def aggregate_feature_recall(rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-row feature_recall dicts into per-feature summaries.

    Each row must have: row['feature_recall'] from this module.
    """
    summary: dict[str, dict] = {}
    for feat in _FEATURE_PATTERNS:
        n_gt = sum(1 for r in rows if r.get('feature_recall', {}).get(feat, {}).get('gt'))
        n_hit = sum(1 for r in rows
                    if r.get('feature_recall', {}).get(feat, {}).get('gt')
                    and r.get('feature_recall', {}).get(feat, {}).get('hit'))
        summary[feat] = {
            'n_gt': n_gt,
            'n_hit': n_hit,
            'recall': (n_hit / n_gt) if n_gt else None,
        }
    return summary
