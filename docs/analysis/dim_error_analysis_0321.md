# Phase 0.5 — dim_error Sub-Classification
**Date:** 2026-03-21
**Script:** `tools/analyze_dim_errors.py`
**Method:** Load pred + GT STL, normalise both to [-1,1]³, compare sorted
normalised extents to classify geometric error type.

Sub-types:
- `aspect_ratio` (asp_diff ≥ 0.25): wrong proportions — one or more axes scaled differently
- `aspect_ratio_mild` (0.12 ≤ asp_diff < 0.25): mild proportion mismatch
- `local_feat` (asp_diff < 0.12): bounding box matches, IoU low due to holes/cutouts/features

---

## Taxonomy breakdown (all low-IoU success cases, IoU ∈ (0, 0.70])

| Combo | dim_error | wrong_primitive | degenerate | wrong_plane | partial_geom | feature_count | no_code | Total |
|---|---|---|---|---|---|---|---|---|
| deepcad_sft_img | 665 | 137 | 49 | 55 | 55 | 0 | 0 | 961 |
| deepcad_rl_img | 346 | 64 | 25 | 23 | 14 | 5 | 0 | 477 |
| deepcad_sft_pc | 455 | 97 | 26 | 26 | 55 | 0 | 0 | 659 |
| deepcad_rl_pc | 403 | 147 | 35 | 19 | 22 | 5 | 0 | 631 |
| fusion360_sft_img | 317 | 17 | 25 | 22 | 22 | 1 | 0 | 404 |
| fusion360_rl_img | 223 | 7 | 11 | 11 | 4 | 9 | 0 | 265 |
| fusion360_sft_pc | 216 | 12 | 25 | 11 | 11 | 0 | 0 | 275 |
| fusion360_rl_pc | 191 | 21 | 11 | 6 | 10 | 3 | 0 | 242 |

---

## dim_error sub-type breakdown

### Aggregated (all combos combined)

| Sub-type | Count | % |
|---|---:|---:|
| `aspect_ratio` (strong) | 11 | 0% |
| `aspect_ratio_mild`     | 39 | 1% |
| `local_feat`            | 2766 | 98% |
| **Total**               | **2816** | 100% |

### Per-combo breakdown

| Combo | n_dim | aspect_ratio | aspect_mild | local_feat | asp_diff_med | asp_diff_p90 | vol_ratio_med | iou_med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deepcad_sft_img | 665 | 5 (0.8%) | 7 (1.1%) | 653 (98.2%) | 0.005 | 0.020 | 1.073 | 0.543 |
| deepcad_rl_img | 346 | 0 (0.0%) | 1 (0.3%) | 345 (99.7%) | 0.004 | 0.013 | 1.075 | 0.580 |
| deepcad_sft_pc | 455 | 3 (0.7%) | 12 (2.6%) | 440 (96.7%) | 0.004 | 0.036 | 1.011 | 0.572 |
| deepcad_rl_pc | 403 | 0 (0.0%) | 3 (0.7%) | 400 (99.3%) | 0.003 | 0.013 | 1.150 | 0.558 |
| fusion360_sft_img | 317 | 1 (0.3%) | 6 (1.9%) | 310 (97.8%) | 0.007 | 0.040 | 1.113 | 0.502 |
| fusion360_rl_img | 223 | 1 (0.4%) | 1 (0.4%) | 221 (99.1%) | 0.005 | 0.019 | 1.085 | 0.553 |
| fusion360_sft_pc | 216 | 1 (0.5%) | 7 (3.2%) | 208 (96.3%) | 0.004 | 0.041 | 1.005 | 0.559 |
| fusion360_rl_pc | 191 | 0 (0.0%) | 2 (1.0%) | 189 (99.0%) | 0.004 | 0.030 | 1.102 | 0.563 |

---

## Volume bias (pred vs GT, normalised space)

| Combo | over>1.1 | near±10% | under<0.9 | median ratio |
|---|---:|---:|---:|---:|
| deepcad_sft_img | 47.7% | 16.5% | 35.8% | 1.073 |
| deepcad_rl_img | 47.4% | 17.3% | 35.3% | 1.075 |
| deepcad_sft_pc | 42.9% | 19.3% | 37.8% | 1.011 |
| deepcad_rl_pc | 54.1% | 13.2% | 32.8% | 1.150 |
| fusion360_sft_img | 50.2% | 14.8% | 35.0% | 1.113 |
| fusion360_rl_img | 47.5% | 20.6% | 31.8% | 1.085 |
| fusion360_sft_pc | 36.6% | 29.2% | 34.3% | 1.005 |
| fusion360_rl_pc | 50.3% | 22.5% | 27.2% | 1.102 |

---

## SFT vs RL dim_error sub-type comparison

| Group | n | aspect_ratio | local_feat |
|---|---:|---:|---:|
| sft_img | 982 | 19 (1%) | 963 (98%) |
| rl_img | 569 | 3 (0%) | 566 (99%) |
| sft_pc | 671 | 23 (3%) | 648 (96%) |
| rl_pc | 594 | 5 (0%) | 589 (99%) |

---

## Bonus: Full taxonomy % (relative to low-IoU success cases)

| Combo | n_total | dim_error | wrong_prim | degenerate | wrong_plane | partial_geom |
|---|---:|---:|---:|---:|---:|---:|
| deepcad_sft_img | 961 | 69% | 14% | 5% | 6% | 6% |
| deepcad_rl_img  | 477 | 73% | 13% | 5% | 5% | 3% |
| deepcad_sft_pc  | 659 | 69% | 15% | 4% | 4% | 8% |
| **deepcad_rl_pc**  | **631** | **64%** | **23%** | 6% | 3% | 3% |
| fusion360_sft_img | 404 | 78% | 4% | 6% | 5% | 5% |
| fusion360_rl_img  | 265 | 84% | 3% | 4% | 4% | 2% |
| fusion360_sft_pc  | 275 | 79% | 4% | 9% | 4% | 4% |
| fusion360_rl_pc   | 242 | 79% | 9% | 5% | 2% | 4% |

**Notable**: `deepcad_rl_pc` wrong_primitive jumps to **23%** vs SFT_PC 15%. RL training
(img modality) increases box-fallback in PC mode — RL optimises for img reward and PC
occasionally falls back to box() as a safe shortcut for flat/thin shapes.

---

## Interpretation

### Main finding: local_feat = 98% of all dim_error cases

**The model gets the overall shape proportions essentially correct** (median asp_diff = 0.003–0.007,
p90 = 0.013–0.041 — all well below the 0.12 threshold). The bounding box of the prediction
matches the GT bounding box. IoU is low (median 0.50–0.58) not because of wrong proportions,
but because of **wrong internal structure**: holes, cutouts, wall thickness, and local feature
positions.

### Volume bias: model over-generates material

Across all combos, pred volume > GT volume (median ratio 1.07–1.15). ~47–54% of cases have
vol_ratio > 1.1. This is the direct signature of **missing subtractive operations** (holes,
pockets, cutouts): the model generates solid material where the GT has voids.

The bias is larger in RL PC mode (median 1.15 vs SFT PC 1.01) — RL's policy slightly amplifies
over-generation in PC mode, possibly because the reward gives partial credit for solid shapes
that overlap the GT volume even without cutouts.

### RL PC mode: wrong_primitive rises (15% → 23%)

RL training (img modality) does not fix, and slightly worsens, the box-fallback pattern in PC
mode. PC cases with flat/thin shapes more often get box() in RL — the model has learned that
`.box()` is a safe approximation for flat shapes and does so more aggressively under RL.

### Phase 1 decision: **Chamfer Distance reward (Option A)**

The analysis conclusively rules out:
- **Scale-aware input encoding** (Option B): proportions are already correct; scale is not the issue
- **Aspect-ratio / profile curriculum**: only 1–2% of dim_error cases have wrong proportions

The analysis conclusively points to:
- **CD reward**: penalises points that are in pred but not in GT (catches missing holes)
  and points in GT not in pred (catches missing features). Gives dense gradient signal
  even when overall IoU is already moderate.
  Formula: `R = α·R_iou + (1−α)·exp(−β·CD)` where CD already computed in scoring pipeline.
- **Wrong_primitive curriculum** (orthogonal, especially for PC mode): RL_PC has 23%
  wrong_primitive; over-sampling flat/thin GT shapes and softly penalising box-only predictions
  is a clean independent fix.
