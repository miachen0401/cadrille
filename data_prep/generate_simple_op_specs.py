"""Auto-generate essential_ops spec for benchcad-simple op-pattern families.

Each family in benchcad-simple is named `simple_<op>` or `simple_<op>_<op>`
where the op tokens directly correspond to the cadquery operations a valid
generation must invoke. We parse the name → AND-list of canonical OP_PATTERNS
keys → write to common/essential_ops_simple.yaml.

The output yaml is loaded alongside common/essential_ops.yaml at import time
(see common/essential_ops.py merge logic).

Token canonicalization (family name → OP_PATTERNS key):
  - revolve / cut / hole / fillet / chamfer / loft / sweep / shell / spline /
    polygon / polyline / lineTo / threePointArc / polarArray / rarray
    → identical match (modulo case)
  - taper / taper_extrude → 'taper=' (cadquery kwarg signal)
  - extrude / extrudeBlind / extrudeUntil → DROPPED (implicit in nearly all
    cadquery code, would over-trigger)
  - box / cyl / cyl_hole / cone / wedge / hemisphere → PRIMITIVE — dropped
    unless it's the *only* token (then we use 'Sketch' as the catch-all)
  - sphere → 'sphere' (a real OP_PATTERNS key)
  - rev → 'revolve' (sometimes abbreviated)
  - compose / union / cyl → no spec (skipped)

Usage:
    uv run python -m data_prep.generate_simple_op_specs
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.essential_ops import OP_PATTERNS  # noqa: E402

# Map family-name tokens (post split) → canonical OP_PATTERNS key.
# None means "drop this token" (primitive shape / implicit op).
TOKEN_MAP: dict[str, str | None] = {
    # canonical ops (case-insensitive identity match)
    'revolve':   'revolve',
    'rev':       'revolve',
    'cut':       'cut',
    'hole':      'hole',
    'fillet':    'fillet',
    'chamfer':   'chamfer',
    'loft':      'loft',
    'sweep':     'sweep',
    'shell':     'shell',
    'spline':    'spline',
    'polygon':   'polygon',
    'polyline':  'polyline',
    'lineTo':    'lineTo',
    'sphere':    'sphere',
    'arc':       'threePointArc',
    'rarray':    'rarray',
    'polar':     'polarArray',
    # taper-extrude → cadquery `extrude(..., taper=N)` signal
    'taper':     'taper=',
    # implicit ops / primitive shapes — drop, leave spec to other tokens
    'extrude':   None,
    'box':       None,
    'cyl':       None,
    'cone':      None,
    'wedge':     None,
    'hemisphere': None,
    'compose':   None,
    'union':     None,
    'array':     None,   # paired with 'polar' or 'r' which already maps
    'simple':    None,   # the prefix
}


def family_to_spec(fam: str) -> list[str] | None:
    """Map a family name like 'simple_revolve_cut' → ['revolve', 'cut'].

    Returns None if no canonical ops can be derived (caller should skip).
    """
    if not fam.startswith('simple_'):
        return None
    raw = fam[len('simple_'):]

    # Greedy multi-token matches first (no false splits inside compound tokens)
    # e.g. 'taper_extrude' is "taper" + "extrude" — but extrude is implicit,
    # so we want spec to contain just 'taper='. The default token-by-token
    # split handles this correctly because TOKEN_MAP['extrude'] = None.
    tokens = raw.split('_')

    ops: list[str] = []
    for tok in tokens:
        if tok in TOKEN_MAP:
            mapped = TOKEN_MAP[tok]
            if mapped is not None and mapped not in ops:
                ops.append(mapped)
        # else: silently skip unknown token

    return ops if ops else None


def main() -> None:
    # Read all bench-simple family names
    rows = pickle.load(open(REPO / 'data/benchcad-simple/train.pkl', 'rb'))
    sys.path.insert(0, str(REPO))
    from train.rl.dataset import _extract_family

    fams: set[str] = set()
    for r in rows:
        f = r.get('family') or _extract_family(r.get('uid', ''))
        if f:
            fams.add(f)

    print(f'[1/2] {len(fams)} bench-simple families found')

    # Build spec
    specs: dict[str, list[str]] = {}
    skipped: list[str] = []
    for fam in sorted(fams):
        spec = family_to_spec(fam)
        if spec:
            specs[fam] = spec
        else:
            skipped.append(fam)

    print(f'[2/2] {len(specs)} families mapped, {len(skipped)} skipped')

    # Sanity: every canonical key should be in OP_PATTERNS
    bad = []
    for fam, ops in specs.items():
        for op in ops:
            if op not in OP_PATTERNS:
                bad.append((fam, op))
    if bad:
        print(f'WARN: {len(bad)} entries reference unknown OP_PATTERNS keys:')
        for fam, op in bad[:5]:
            print(f'    {fam} → {op}')

    # Write yaml alongside the canonical spec
    out = REPO / 'common/essential_ops_simple.yaml'
    with out.open('w') as f:
        f.write(
            '# Auto-generated bench-simple op-pattern specs.\n'
            '# DO NOT EDIT BY HAND — regenerate via:\n'
            '#     uv run python -m data_prep.generate_simple_op_specs\n'
            '#\n'
            '# Each family name encodes its own composition spec; tokens map to\n'
            '# canonical OP_PATTERNS keys via data_prep.generate_simple_op_specs.TOKEN_MAP.\n'
            '# extrude / box / cyl / cone tokens are dropped as implicit primitives.\n'
            '\n'
        )
        for fam in sorted(specs):
            ops = specs[fam]
            f.write(f'{fam}:\n')
            for op in ops:
                f.write(f'  - {op}\n')
            f.write('\n')

    print(f'wrote {out}')
    print(f'\nsample specs:')
    for fam in sorted(specs)[:8]:
        print(f'  {fam:35s} → {specs[fam]}')
    if skipped:
        print(f'\nskipped families (no recognizable op tokens):')
        for fam in skipped[:10]:
            print(f'  {fam}')


if __name__ == '__main__':
    main()
