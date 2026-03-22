# Phase 0.4 — SFT vs RL Per-Case Delta
**Date:** 2026-03-21
**Script:** `tools/analyze_sft_rl_delta.py`
**Scope:** All 8046 (DeepCAD) / 1725 (Fusion360) cases × 2 modalities

Per-case delta categories (SFT→RL):
- `fixed`: SFT failed (IoU=0 or error), RL succeeded (IoU>0)
- `boosted`: both succeeded, RL IoU > SFT IoU + 0.05
- `stable`: both similar (|ΔIoU| ≤ 0.05)
- `regressed`: both succeeded, RL IoU < SFT IoU − 0.05
- `broken`: SFT succeeded, RL failed

---

## Summary table

| Combo | fixed | boosted | stable | regressed | broken | net ΔIoU |
|---|---|---|---|---|---|---|
| deepcad/img | 6.6% | 21.5% | 69.4% | 2.3% | 0.3% | **+7.10pp** |
| deepcad/pc | 6.0% | 11.8% | 74.0% | 6.9% | 1.2% | **+4.55pp** |
| fusion360/img | 12.0% | 23.8% | 59.2% | 4.3% | 0.6% | **+9.61pp** |
| fusion360/pc | 11.2% | 17.0% | 62.6% | 7.1% | 2.0% | **+8.35pp** |

---

## Key findings

- RL improves both modalities and both datasets.
- **img benefits more** than pc (train_modality=img — expected asymmetry).
- DeepCAD/img: 28.1% improve vs 2.6% degrade → very clean gain.
- DeepCAD/pc: 17.8% improve vs 8.1% degrade → net positive but less efficient.
- Fusion360 gains larger (12% fixed vs 6.6% for DeepCAD) — Fusion360 SFT baseline has more failures.
- `broken` rate is low (0.3–2.0%) → RL rarely destroys working solutions.
- PC regression (6.9–7.1%) is the main downside of img-only RL training.

## Implication for Phase 1

RL's main lever is **fixing failures** (fixed 6–12%) and **boosting partial successes** (boosted 11–24%).
The residual error pool is dominated by dim_error (72%) in cases where both SFT and RL succeed at low IoU.
→ Phase 1 target: reduce dim_error via better numeric grounding (process reward / verifier).
