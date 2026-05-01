"""Single source of truth for "the representative case per family".

Every per-family deep-dive should call `pick_canonical_case(family)` and
NOT pick its own stem — otherwise the same family shows different stems
across analyses and cross-referencing breaks down.

Selection rule (deterministic, unbiased):
  1. Filter to cases where BOTH CADEvolve v3 AND Q3VL-v3 exec successfully.
  2. Compute family median Q3VL IoU.
  3. Pick the case whose Q3VL IoU is closest to that median.
  4. Tie-break: alphabetical stem.

This gives a "typical" case for the family — not the best, not the worst,
not the maximum-contrast case. Suitable as a reference point used
consistently across overview / weak-family / group / code-deepdive
visualisations.

Scripts whose PURPOSE is to surface extremes (e.g. group A "high IoU
ESS=False", group B "low IoU ESS=False", gears "best CE", high-IoU-no-
ESS) intentionally use their own selection rule and do NOT call this
function — but they should label themselves clearly so the reader
knows the case won't match the canonical per-family one.
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT = REPO / 'eval_outputs/cad_bench_722'
META_PATH = {
    'ce': EVAL_ROOT / 'cadevolve_rl1' / 'metadata.jsonl',
    'q':  EVAL_ROOT / 'cadrille_qwen3vl_v3' / 'metadata.jsonl',
}


@lru_cache(maxsize=1)
def _load_metas():
    out = {}
    for k, p in META_PATH.items():
        out[k] = {}
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line); out[k][r['stem']] = r
                except Exception:
                    pass
    return out


@lru_cache(maxsize=1)
def _load_canonical():
    """Build the canonical per-family pick once and cache."""
    metas = _load_metas()
    # Group stems by family using whichever metadata has the family field.
    by_family = defaultdict(list)
    for stem, q_rec in metas['q'].items():
        fam = q_rec.get('family')
        if not fam:
            ce_rec = metas['ce'].get(stem)
            if ce_rec: fam = ce_rec.get('family')
        if not fam: continue
        ce_rec = metas['ce'].get(stem) or {}
        if (q_rec.get('error_type') != 'success'
                or ce_rec.get('error_type') != 'success'):
            continue
        if q_rec.get('iou') is None or ce_rec.get('iou') is None:
            continue
        by_family[fam].append((stem, q_rec['iou'], ce_rec['iou']))

    canonical = {}
    for fam, items in by_family.items():
        items.sort(key=lambda t: t[1])  # by Q3VL IoU
        median_iou = items[len(items) // 2][1]
        items.sort(key=lambda t: (abs(t[1] - median_iou), t[0]))
        canonical[fam] = items[0][0]
    return canonical


def pick_canonical_case(family: str) -> str | None:
    """Return the canonical stem for `family`, or None if no both-exec-ok case."""
    return _load_canonical().get(family)


def all_canonical() -> dict[str, str]:
    """family → canonical stem mapping."""
    return dict(_load_canonical())


if __name__ == '__main__':
    # CLI: print all family → canonical stem mappings for inspection.
    canonical = all_canonical()
    print(f'Per-family canonical cases ({len(canonical)} families):')
    for fam in sorted(canonical):
        print(f'  {fam:<26}  {canonical[fam]}')
