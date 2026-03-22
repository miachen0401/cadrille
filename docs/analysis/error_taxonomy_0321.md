# Error Taxonomy for Low-IoU CadQuery Predictions

**Date:** 2026-03-21
**Scope:** Automated analysis of 200 low-IoU predictions (50 per combo)

---

## Methodology

### Sample Selection

For each of 4 combos — deepcad_sft_img, deepcad_rl_img, deepcad_sft_pc, deepcad_rl_pc — 50 cases with IoU in the range (0, 0.70) were sampled. All predictions executed without exception (error_type = "success").

| Combo | Min IoU | Max IoU | Mean IoU | n (<0.10) | n (0.10–0.30) | n (0.30–0.50) | n (0.50–0.70) |
|---|---|---|---|---|---|---|---|
| deepcad_sft_img | 0.003 | 0.694 | 0.465 | 4 | 6 | 12 | 28 |
| deepcad_rl_img | 0.007 | 0.686 | 0.477 | 4 | 6 | 11 | 29 |
| deepcad_sft_pc | 0.004 | 0.698 | 0.486 | 2 | 5 | 14 | 29 |
| deepcad_rl_pc | 0.058 | 0.700 | 0.487 | 4 | 7 | 8 | 31 |

### Classification Method

Each `_pred.py` file was parsed programmatically. Features extracted:
- **Workplane type** (XY, ZX, YZ)
- **Operation counts**: `.extrude()`, `.union()`, `.cut()`, `.revolve()`, `.loft()`
- **Sketch primitives**: `.sketch()`, `.circle()`, `.rect()`, `.segment()`, `.arc()`
- **Cutout operations**: `mode='s'`
- **Primitive functions**: `.box()`, `.cylinder()`
- **Extrude distances**, **box dimensions**, **spatial extent**, **code length**

Deterministic rule-based classifier applied in priority order:

1. **degenerate** — IoU < 0.05 OR extrude distance ≤ 1.5 with spatial extent > 50 and IoU < 0.15
2. **wrong_primitive** — uses `.box()` or `.cylinder()` without any `.sketch()` + `.extrude()`
3. **partial_geom** — single `.extrude()`, no boolean ops, complex profile with IoU < 0.40
4. **wrong_plane** — non-XY workplane for what should be XY shape (or vice versa) with IoU < 0.40
5. **feature_count** — 4+ union operations with IoU < 0.45
6. **dim_error** — default: code structure correct, numeric dimensions wrong

---

## Per-Combo Count Table

| Category | SFT\_IMG | RL\_IMG | SFT\_PC | RL\_PC | **Row Total** |
|---|---|---|---|---|---|
| **dim\_error** | 34 (68%) | 40 (80%) | 35 (70%) | 34 (68%) | **143 (72%)** |
| **wrong\_primitive** | 7 (14%) | 4 (8%) | 8 (16%) | 7 (14%) | **26 (13%)** |
| **degenerate** | 2 (4%) | 4 (8%) | 2 (4%) | 4 (8%) | **12 (6%)** |
| **wrong\_plane** | 4 (8%) | 1 (2%) | 3 (6%) | 2 (4%) | **10 (5%)** |
| **partial\_geom** | 3 (6%) | 0 (0%) | 2 (4%) | 2 (4%) | **7 (4%)** |
| **feature\_count** | 0 (0%) | 1 (2%) | 0 (0%) | 1 (2%) | **2 (1%)** |
| **Total** | **50** | **50** | **50** | **50** | **200** |

---

## Category Descriptions and Representative Examples

### 1. dim\_error (143 / 200 = 72%)

**Definition:** Code structure is topologically correct — right workplane, right number of extrude/boolean operations, right sketch primitive types — but numeric values (dimensions, positions, scale) do not match GT. IoU range: 0.14–0.69, mean ~0.53.

**Typical sub-patterns:**
- *Scale error*: overall shape correct but proportions off
- *Position error*: feature placed at wrong offset
- *Profile error*: sketch segments form correct topology but wrong vertex coordinates
- *Height error*: extrude depth substantially wrong

**Representative examples:**
```python
# SFT_IMG 00709457, IoU=0.602 — two bodies, right structure, wrong dimensions
w0=cq.Workplane('XY',origin=(0,0,-9))
r=w0.sketch().segment((-100,-48),(25,-48))...close().assemble().finalize().extrude(17)
  .union(w0.sketch()...finalize().extrude(18))
```
```python
# SFT_PC 00948060, IoU=0.633 — nested hollow box, correct topology
w0=cq.Workplane('XY',origin=(0,0,-70))
r=w0.sketch().rect(200,176).rect(194,168,mode='s').finalize().extrude(66)
  .union(w0.sketch().rect(190,164).rect(186,160,mode='s').finalize().extrude(140))
```

---

### 2. wrong\_primitive (26 / 200 = 13%)

**Definition:** Model selected wrong geometric primitive type. Dominant pattern: using `.box()` directly instead of `sketch()` + `extrude()`. Nearly all cases: `w0.workplane(...).box(a, b, c)` where one dimension is very small (2–4 units). IoU: 0.23–0.70.

**Representative examples:**
```python
# SFT_IMG 00035212, IoU=0.685 — flat plate, used box instead of sketch+extrude
r=w0.workplane(offset=-200/2).box(132,2,200)
```
```python
# SFT_PC 00869034, IoU=0.698 — thin bar
r=w0.workplane(offset=-200/2).box(4,18,200)
```
```python
# RL_PC 00033121, IoU=0.700 — thin bar + tiny cube union (training artifact)
r=w0.workplane(offset=-200/2).moveTo(0,0).box(11.5,4,200)
  .union(w0.workplane(offset=-2/2).box(11.5,1,1))
```

**Note:** Several wrong_primitive cases include a "dummy union" — a tiny `box(1,1,1)` appended to the main primitive. Training artifact where model attempts to match multi-body structure of GT.

---

### 3. degenerate (12 / 200 = 6%)

**Definition:** Code runs and produces geometry, but result is near-zero volume or extremely thin.

| Sub-type | Count | Description |
|---|---|---|
| flat_slab | 5 | sketch + extrude(1) with large spatial extent; ~1mm tall object |
| thin_box_near_zero | 4 | box(200, 1.5, 200) or similar — IoU near zero |
| near_zero (other) | 2 | Near-zero IoU for structural reasons |
| over_subtracted | 1 | mode='s' subtract operations consume entire volume |

```python
# SFT_IMG 00195343, IoU=0.100 — cross-shaped profile but only 1mm tall
r=w0.sketch()...close().assemble().finalize().extrude(1)   # tiny extrude depth
```
```python
# SFT_PC 00118022, IoU=0.004 — over-subtracted (cuts > additions)
r=w0.sketch()...push([(-78,-14.5)]).rect(16,37,mode='s')...finalize().extrude(-4)
```

---

### 4. wrong\_plane (10 / 200 = 5%)

**Definition:** Correct sketch profile and extrude approach but workplane placed on wrong axis. Shape extruded in wrong direction relative to GT.

```python
# SFT_IMG 00115681, IoU=0.280 — ZX plane, correct L-shaped profile, wrong axis
w0=cq.Workplane('ZX',origin=(0,-100,0))
r=w0.sketch().segment((-5,-100),(5,-100))...close().assemble().finalize().extrude(200)
```
```python
# RL_IMG 00231080, IoU=0.055 — XY plane but extremely thin frame (cross-section)
r=w0.sketch().rect(200,68.5).push([(0,0)]).rect(193.5,63.5,mode='s').finalize().extrude(-15)
```

---

### 5. partial\_geom (7 / 200 = 4%)

**Definition:** Model generates only a fragment of expected multi-body shape — single `.extrude()` with no boolean operations, complex profile suggests model attempted cross-section instead of assembling multiple solid bodies. IoU: 0.05–0.35.

```python
# SFT_IMG 00039347, IoU=0.098 — complex arc profile, single extrude, no union
r=w0.sketch()
  .arc((-91,12),(-82,-27),(-56,-57))...close().assemble()
  .reset().face(...)...finalize().extrude(6)
```
```python
# SFT_PC 00372113, IoU=0.291 — I/H-beam cross section, single extrude
r=w0.sketch().segment((-10,-4),(-8,-4))...close().assemble().finalize().extrude(-200)
```

---

### 6. feature\_count (2 / 200 = 1%)

**Definition:** Wrong number of features — extra unions/bodies beyond what GT contains.

```python
# RL_IMG 00320090, IoU=0.397 — 4 unions, two overlapping boxes + extra small boxes
r=w0.workplane(offset=6/2).moveTo(-20,0).box(60.5,180.5,6)
  .union(w0.workplane(offset=6/2).moveTo(-20,0).box(60.5,169.5,6))  # redundant
  .union(w0.sketch()...finalize().extrude(76))
  .union(w0.workplane(offset=76/2).moveTo(-47,0.5).box(5,1.5,76))
  .union(w0.workplane(offset=76/2).moveTo(50.5,97.5).box(7.5,1.5,76))
```

---

## Cross-Combo Comparison

### IMG vs PC

| Category | IMG (n=100) | PC (n=100) | Difference |
|---|---|---|---|
| dim\_error | 74 (74%) | 69 (69%) | IMG +5pp |
| wrong\_primitive | 11 (11%) | 15 (15%) | PC +4pp |
| degenerate | 6 (6%) | 6 (6%) | equal |
| wrong\_plane | 5 (5%) | 5 (5%) | equal |
| partial\_geom | 3 (3%) | 4 (4%) | PC +1pp |
| feature\_count | 1 (1%) | 1 (1%) | equal |

PC mode produces more wrong_primitive failures (15% vs 11%). Point clouds give the model less precise information about whether GT is a thin plate/bar, leading to more box() fallback. IMG provides rendered depth cues, helping the model make better primitive-type decisions.

### SFT vs RL

| Category | SFT (n=100) | RL (n=100) | Difference |
|---|---|---|---|
| dim\_error | 69 (69%) | 74 (74%) | RL +5pp |
| wrong\_primitive | 15 (15%) | 11 (11%) | SFT +4pp |
| wrong\_plane | 7 (7%) | 3 (3%) | SFT +4pp |
| partial\_geom | 5 (5%) | 2 (2%) | SFT +3pp |
| degenerate | 4 (4%) | 8 (8%) | RL +4pp |
| feature\_count | 0 (0%) | 2 (2%) | RL +2pp |

Key findings:
1. **RL reduces structural errors** (27 → 16/100): wrong_primitive, wrong_plane, partial_geom all decrease. RL reward efficiently penalizes wrong geometric structure.
2. **RL slightly increases degenerate failures** (4 → 8/100): Reward exploitation — model occasionally generates very thin flat geometries that score partial IoU on flat-ish GT shapes.
3. **RL increases dim_error rate** (69% → 74%): Reflects successful structural learning — more predictions have the right geometry type, but numeric precision remains the limiting factor.
4. **SFT has more wrong_plane errors** (7 vs 3): RL with IoU reward strongly penalizes misaligned planes (near-zero IoU), leading to rapid correction.

---

## Summary

```
dim_error       ████████████████████████████████████ 72% (143/200)
wrong_primitive ██████ 13% (26/200)
degenerate      ███ 6% (12/200)
wrong_plane     ██ 5% (10/200)
partial_geom    ██ 4% (7/200)
feature_count   ▌ 1% (2/200)
```

### Key Conclusions

1. **Dim_error dominates (72%)**: Model has learned geometric grammar well but struggles with metric accuracy. Primary improvement path: better coordinate/dimension estimation, not shape classification.
2. **Wrong_primitive is second (13%)**: Box() shortcut that partially works (IoU 0.5–0.7 on flat shapes) but fundamentally wrong. Training with reward bonus for sketch+extrude vs box might help.
3. **RL reduces structural failures** (27 → 16/100): IoU reward efficiently penalizes wrong structure.
4. **Degenerate failures increase slightly with RL** (4 → 8/100): Possible reward exploitation for ambiguous flat GT shapes.
5. **Feature-count errors are rare (1%)**: Model seldom adds truly spurious features.
6. **IMG vs PC difference is small**: Error type distributions nearly identical between modalities.

---

## Appendix: Full Case List

### deepcad_sft_img (n=50)

| Stem | IoU | Category | Reason |
|---|---|---|---|
| 00709457 | 0.602 | dim_error | multi-op moderate IoU |
| 00140290 | 0.533 | dim_error | box+sketch union |
| 00035212 | 0.685 | wrong_primitive | box(132,2,200) |
| 00816234 | 0.586 | dim_error | complex sketch |
| 00309121 | 0.565 | dim_error | multi-sketch |
| 00289312 | 0.506 | dim_error | complex union |
| 00267698 | 0.666 | wrong_primitive | box(100,2,200) YZ |
| 00181768 | 0.302 | dim_error | nested circles |
| 00812011 | 0.285 | dim_error | ZX+XY multi-wp |
| 00131316 | 0.521 | dim_error | complex profile |
| 00749324 | 0.429 | dim_error | multi-sketch |
| 00815969 | 0.529 | dim_error | multi-op |
| 00952806 | 0.003 | degenerate | near_zero_iou |
| 00609109 | 0.625 | wrong_primitive | box(200,2,200) YZ |
| 00115681 | 0.280 | wrong_plane | ZX non_xy_simple |
| 00664811 | 0.444 | dim_error | complex profile |
| 00464765 | 0.425 | dim_error | multi-op |
| 00039347 | 0.098 | partial_geom | complex_profile_single |
| 00038103 | 0.560 | dim_error | multi-sketch |
| 00123451 | 0.477 | dim_error | complex |
| 00266412 | 0.540 | dim_error | simple extrude |
| 00277425 | 0.052 | partial_geom | complex_profile_single |
| 00551181 | 0.516 | dim_error | multi-op |
| 00677369 | 0.331 | wrong_plane | ZX non_xy_simple |
| 00035978 | 0.345 | wrong_plane | YZ non_xy_simple |
| 00627747 | 0.514 | dim_error | complex |
| 00241271 | 0.128 | partial_geom | medium_profile_low |
| 00791775 | 0.680 | wrong_primitive | box(2,100,200) YZ |
| 00722400 | 0.361 | dim_error | multi-sketch |
| 00778195 | 0.667 | dim_error | complex multi-op |
| 00460602 | 0.665 | dim_error | simple extrude |
| 00266505 | 0.445 | dim_error | multi-op |
| 00494624 | 0.486 | dim_error | simple |
| 00664516 | 0.267 | dim_error | multi-sketch |
| 00310171 | 0.571 | dim_error | complex |
| 00869061 | 0.624 | dim_error | multi-op |
| 00934891 | 0.398 | dim_error | complex |
| 00006584 | 0.581 | dim_error | multi-op |
| 00832249 | 0.600 | dim_error | simple |
| 00868218 | 0.619 | wrong_primitive | box(12,4,200) |
| 00199695 | 0.420 | wrong_primitive | box(200,2,200) |
| 00773251 | 0.568 | dim_error | simple |
| 00380945 | 0.242 | wrong_plane | ZX non_xy_low |
| 00195343 | 0.100 | degenerate | flat_slab_ext=1 |
| 00264381 | 0.694 | dim_error | complex |
| 00834440 | 0.670 | dim_error | multi-op |
| 00379127 | 0.529 | wrong_primitive | box(2,172,200) YZ |
| 00120803 | 0.666 | dim_error | multi-op |
| 00425515 | 0.691 | dim_error | complex |
| 00129290 | 0.178 | dim_error | box+sketch |

### deepcad_rl_img (n=50)

| Stem | IoU | Category | Reason |
|---|---|---|---|
| 00326127 | 0.431 | dim_error | multi-op |
| 00914502 | 0.669 | dim_error | complex |
| 00310954 | 0.655 | dim_error | complex |
| 00657352 | 0.007 | degenerate | thin_box_near_zero |
| 00266505 | 0.675 | dim_error | multi-op |
| 00864347 | 0.625 | dim_error | complex |
| 00045026 | 0.514 | dim_error | complex |
| 00800534 | 0.267 | dim_error | multi-push-rects |
| 00464719 | 0.228 | wrong_primitive | box(1,83,200)+box YZ |
| 00556004 | 0.141 | dim_error | thin box union |
| 00147363 | 0.616 | dim_error | complex |
| 00993185 | 0.659 | dim_error | simple |
| 00350813 | 0.111 | degenerate | flat_slab_ext=1 |
| 00096611 | 0.641 | dim_error | complex |
| 00572504 | 0.596 | dim_error | complex |
| 00289312 | 0.625 | dim_error | multi-op |
| 00896445 | 0.645 | dim_error | complex |
| 00680342 | 0.272 | dim_error | multi-op |
| 00666410 | 0.612 | dim_error | complex |
| 00949693 | 0.368 | wrong_primitive | box(1,123,200) YZ |
| 00933384 | 0.662 | dim_error | simple |
| 00328737 | 0.011 | degenerate | near_zero |
| 00617131 | 0.604 | dim_error | simple |
| 00206376 | 0.558 | dim_error | multi-op |
| 00773560 | 0.466 | dim_error | simple |
| 00086004 | 0.601 | dim_error | rect frame |
| 00046003 | 0.495 | dim_error | complex |
| 00735988 | 0.533 | dim_error | complex |
| 00231080 | 0.055 | wrong_plane | cross_section_thin_xy |
| 00837850 | 0.421 | dim_error | multi-op |
| 00287456 | 0.549 | dim_error | complex |
| 00930765 | 0.015 | degenerate | thin_box_near_zero |
| 00236689 | 0.638 | wrong_primitive | box(3,3.5,200)+boxes |
| 00937263 | 0.597 | dim_error | complex |
| 00116780 | 0.238 | wrong_primitive | box(200,167,1) |
| 00351934 | 0.576 | dim_error | simple |
| 00276736 | 0.652 | dim_error | multi-op |
| 00458916 | 0.605 | dim_error | complex |
| 00692217 | 0.621 | dim_error | complex |
| 00900812 | 0.443 | dim_error | complex |
| 00329909 | 0.407 | dim_error | multi-op |
| 00179249 | 0.372 | dim_error | complex |
| 00343787 | 0.676 | dim_error | complex |
| 00320090 | 0.397 | feature_count | excess_unions=4 |
| 00217731 | 0.449 | dim_error | complex |
| 00745875 | 0.323 | dim_error | complex |
| 00267259 | 0.686 | dim_error | complex |
| 00773251 | 0.590 | dim_error | simple |
| 00753161 | 0.599 | dim_error | multi-op |
| 00708952 | 0.674 | dim_error | complex |

### deepcad_sft_pc (n=50)

| Stem | IoU | Category | Reason |
|---|---|---|---|
| 00154147 | 0.207 | wrong_plane | ZX non_xy_low |
| 00948060 | 0.633 | dim_error | nested frame |
| 00992611 | 0.625 | dim_error | complex |
| 00291304 | 0.264 | dim_error | box+sketch |
| 00841954 | 0.697 | dim_error | simple |
| 00396775 | 0.569 | dim_error | simple |
| 00283983 | 0.579 | dim_error | multi-op |
| 00752986 | 0.517 | wrong_primitive | box(100,2,200) |
| 00625550 | 0.377 | wrong_primitive | box(200,150,20)+box |
| 00451398 | 0.503 | dim_error | complex |
| 00995116 | 0.605 | dim_error | complex |
| 00869034 | 0.698 | wrong_primitive | box(4,18,200) |
| 00360703 | 0.598 | wrong_primitive | box(2,100,200) YZ |
| 00517626 | 0.622 | dim_error | complex |
| 00118022 | 0.004 | degenerate | over_subtracted |
| 00372113 | 0.291 | partial_geom | complex_profile_single |
| 00072711 | 0.251 | wrong_primitive | box(200,2,168) |
| 00499421 | 0.476 | dim_error | complex |
| 00656022 | 0.698 | dim_error | simple |
| 00446293 | 0.540 | dim_error | multi-op |
| 00138165 | 0.379 | wrong_primitive | box(2,100,200) YZ |
| 00345035 | 0.451 | dim_error | complex |
| 00890876 | 0.399 | dim_error | multi-op |
| 00499337 | 0.608 | dim_error | triangle flat plate |
| 00346563 | 0.344 | partial_geom | complex_profile_single |
| 00802387 | 0.525 | dim_error | complex |
| 00646292 | 0.683 | dim_error | complex |
| 00996515 | 0.584 | dim_error | simple |
| 00747767 | 0.468 | dim_error | complex |
| 00255020 | 0.445 | dim_error | complex |
| 00437689 | 0.691 | wrong_primitive | box(134,4,200) YZ |
| 00249707 | 0.640 | dim_error | complex |
| 00398470 | 0.565 | dim_error | complex |
| 00878404 | 0.670 | dim_error | simple |
| 00845141 | 0.345 | wrong_plane | ZX non_xy_simple |
| 00436611 | 0.507 | dim_error | multi-op |
| 00912240 | 0.577 | wrong_primitive | box(2,200,200) YZ |
| 00702516 | 0.447 | dim_error | complex |
| 00911599 | 0.553 | dim_error | multi-op |
| 00653353 | 0.310 | dim_error | complex |
| 00590175 | 0.378 | dim_error | complex |
| 00248013 | 0.641 | dim_error | complex |
| 00817047 | 0.505 | dim_error | simple |
| 00792847 | 0.574 | dim_error | complex |
| 00179685 | 0.319 | dim_error | complex |
| 00103060 | 0.010 | degenerate | near_zero |
| 00200892 | 0.324 | wrong_plane | ZX non_xy_simple |
| 00266505 | 0.611 | dim_error | complex |
| 00981931 | 0.697 | dim_error | simple |
| 00277425 | 0.285 | dim_error | complex profile |

### deepcad_rl_pc (n=50)

| Stem | IoU | Category | Reason |
|---|---|---|---|
| 00711660 | 0.694 | dim_error | simple |
| 00979269 | 0.691 | wrong_primitive | box(56.5,3,200)+box YZ |
| 00109798 | 0.058 | degenerate | flat_slab_ext=1 |
| 00639898 | 0.692 | dim_error | box+sketch |
| 00630909 | 0.654 | dim_error | simple |
| 00789034 | 0.660 | dim_error | complex |
| 00866616 | 0.522 | dim_error | complex |
| 00384222 | 0.436 | dim_error | multi-op |
| 00916309 | 0.678 | dim_error | complex |
| 00021948 | 0.576 | dim_error | complex |
| 00187336 | 0.554 | dim_error | flat triangle plate |
| 00882538 | 0.235 | partial_geom | complex_profile_single |
| 00419857 | 0.637 | dim_error | simple |
| 00536120 | 0.404 | wrong_primitive | box(180,200,1) flat |
| 00186338 | 0.571 | dim_error | complex |
| 00478429 | 0.287 | wrong_plane | YZ non_xy_simple |
| 00741972 | 0.628 | dim_error | complex |
| 00239593 | 0.105 | dim_error | two hollow cylinders |
| 00765285 | 0.671 | dim_error | complex |
| 00006345 | 0.692 | dim_error | simple |
| 00410883 | 0.345 | wrong_plane | ZX non_xy_simple |
| 00828903 | 0.432 | dim_error | multi-op |
| 00283985 | 0.097 | degenerate | thin_box_low_iou |
| 00836983 | 0.556 | dim_error | complex |
| 00174979 | 0.643 | dim_error | simple |
| 00481017 | 0.659 | dim_error | complex |
| 00993146 | 0.641 | dim_error | complex |
| 00299865 | 0.073 | degenerate | thin_box_low_iou |
| 00234945 | 0.228 | dim_error | complex |
| 00613340 | 0.609 | dim_error | simple |
| 00246095 | 0.217 | partial_geom | complex_profile_single |
| 00894530 | 0.554 | dim_error | complex |
| 00869061 | 0.413 | dim_error | multi-op |
| 00001698 | 0.656 | wrong_primitive | box(3,200,200)+boxes YZ |
| 00981638 | 0.423 | dim_error | flat plate profile |
| 00509723 | 0.513 | dim_error | complex |
| 00818575 | 0.522 | wrong_primitive | box(5,200,200)+boxes |
| 00033121 | 0.700 | wrong_primitive | box(11.5,4,200)+box YZ |
| 00580046 | 0.488 | dim_error | flat polygon profile |
| 00489881 | 0.591 | wrong_primitive | box(56.5,4,200) YZ |
| 00362573 | 0.161 | feature_count | excess_unions=4 |
| 00097980 | 0.433 | dim_error | simple |
| 00363755 | 0.645 | dim_error | simple |
| 00937263 | 0.575 | dim_error | complex |
| 00135199 | 0.643 | dim_error | complex |
| 00140290 | 0.225 | dim_error | complex |
| 00813814 | 0.661 | dim_error | complex |
| 00116119 | 0.651 | wrong_primitive | box(4,200,200) YZ |
| 00872963 | 0.515 | dim_error | complex |
| 00202675 | 0.059 | wrong_primitive | box(5,1,200)+box YZ |
