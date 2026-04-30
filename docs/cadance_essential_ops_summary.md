# Cadance essential-ops scoring — summary

Source: HaozheZhang6/Cadance PR #7 (`ops-score` branch, merged).
Captured 2026-04-30 for cadrille paper writeup.

## Final score formula (single source of truth)

```text
score = 0.60·max(iou, iou_rot24)   ← geometry headline
      + 0.20·essential_pass         ← per-family op check (anti-shortcut)
      + 0.10·feature_f1             ← has_chamfer / has_fillet / has_hole
      + 0.05·cd_score
      + 0.05·hd_score
```

Total = 1.0. N/A handling: when `essential_pass = None` (62/106 families),
drop 0.20 term and rescale remaining sum ×1.25.

## essential_pass

Per-family `essential = [<element>, ...]` in `canonical_ops.yaml`:
- outer list = AND (every element must be satisfied)
- inner string = single required op
- inner list = OR-tuple of equivalents (e.g. `[loft, twistExtrude]`)

Example: `helical_gear: [[loft, twistExtrude], [polyline, spline, threePointArc, Sketch]]`
→ pass iff `(loft OR twistExtrude)` AND (any profile op).

Coverage: 44/106 families have essential, 62 are N/A.

## Recognized ops in Cadance (vs our online_eval `_OPS`)

**Structural** (in essentials):
`sweep+helix`, `sweep`, `revolve`, `loft`, `shell`, `taper=`,
`polarArray`, `rarray`, `twistExtrude`, `makeTorus`, `sphere`, `cut`

**Profile** (in essentials):
`polyline`, `spline`, `threePointArc`, `Sketch`, `polygon`, `lineTo`

**Feature class** (independent — for `feature_f1`, NOT essential):
`chamfer`, `fillet`, `hole`

Notable additions vs ours:
- `sweep+helix` (sweep with `makeHelix`/`.helix(...)` anywhere) — order-agnostic regex
- `taper=` (taper kwarg in extrude) — distinct from `loft`
- `twistExtrude` (cadquery method)
- `Sketch` (cq.Sketch object — vs free-hand workplane wires)

Our current `_OPS` is missing:
`sweep+helix`, `taper=`, `twistExtrude`, `Sketch`, `polarArray`, `rarray`,
`makeTorus`. These are all NEEDED for the rare-op story we want to tell
(helix → torsion_spring/coil_spring/worm_screw, twistExtrude → twisted_drill/
twisted_bracket, etc.).

## What this gives us for the paper

A scoring metric that is **defensible against the "model just picked a
substitute op" critique**:

> Why is this not just IoU?  
> Because a model that emits a polygonal extrusion approximation of a helix
> can score IoU = 0.6 yet `essential_pass = 0` (no `sweep+helix` /
> `makeHelix` / `twistExtrude`). The 0.20 weight on essential punishes
> that shortcut without throwing away IoU entirely.

## Edge cases (per Cadance SCORING.md)

| Scenario | Behavior |
|---|---|
| Generated code fails to exec | `iou=iou_rot24=cd_score=hd_score=0`. `feature_f1` and `essential_pass` STILL computed from gen_code text. Fallback: `score = 0.10·feature_f1 + 0.20·ess` capped at 0.3; N/A scaling still applies. |
| GT exec fails | Same; drop sample (rare). |
| `--rot-invariant` not used | `IoU` term uses raw `iou` instead of `max(iou, iou_rot24)`. |
| Family is N/A | Drop 0.20 essential, rescale ×1.25. |

13 N/A families in current spec: chair, dowel_pin, i_beam, parallel_key,
stepped_shaft, table, wall_anchor, clevis_pin, round_flange, t_pipe_fitting,
tee_nut, phone_stand, pull_handle.

## Adoption plan for our paper

1. **Use the same combined-score formula** — port Cadance's `bench/metrics/
   combined_score()` to `eval/cadance_score.py` so we report the same
   number Cadance does. Even better: depend on Cadance directly via a
   `pip install -e ../Cadance` or vendor the YAML.

2. **Per-family essential pass-rate** in our table — exactly what user
   asked for in #3. Replaces or augments the rare-op recall column.

3. **For families NOT in our 106 (e.g. anything from cad-iso-106 /
   benchcad-simple)** — N/A scaling kicks in automatically.

4. **Feature F1** — already trivial to compute from regex on gen_code.
   No exec needed, fast.

## Adopting essentials into our `online_eval`

Implementation sketch:

```python
# train/sft/online_eval.py

from common.canonical_ops import find_ops, essential_pass  # vendored
def _essential_pass_rate(pred_codes, gt_families):
    n_passed = n_applicable = 0
    for code, fam in zip(pred_codes, gt_families):
        ops = find_ops(code)
        ok = essential_pass(fam, ops)
        if ok is None:  # N/A family
            continue
        n_applicable += 1
        if ok: n_passed += 1
    return n_passed / max(n_applicable, 1), n_applicable

# Add to bucket eval:
out['essential_pass_rate'] = ...
out['essential_n_applicable'] = ...
```

Plus a new `feature_f1` per-bucket. Both fit alongside the existing
op_macro_recall etc.
