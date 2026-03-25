# Research Plan — Cadrille NeurIPS 2026

## Goal

Push Cadrille beyond the CPPO baseline (img/DeepCAD 92.2%) with better multimodal grounding.
**Deadline**: ~50 days (submission ~May 2026).
**Strategy**: empirical-first — understand failure modes before proposing fixes.

---

## Phase 0 — Error Taxonomy (CURRENT)

**Focus: official `cadrille-sft` vs `cadrille-rl` only.**
Our H100 training runs are just steps-limited (~360 steps vs official ~3600+); treat them as future work.
The research question here is: what are the residual bottlenecks in the *official* RL model?

### Step 0.1 — Full inference run ✅ COMPLETE (2026-03-21)

Script: `tools/analyze_errors.py`. All 8046/1725 cases × 2 models × 2 modalities.
Output: `data/analysis/{dataset}_{model}_{modality}/metadata.jsonl` + `{stem}_pred.py` + `{stem}_pred.stl`.

### Step 0.2 — IoU distribution analysis ✅ COMPLETE (2026-03-21)

See `docs/analysis/phase0_analysis_0321.md`. Key finding: official RL img/DC = **92.7% > 92.2% paper target**.
RL's main gain is failure elimination (runtime_error −18×), not precision improvement.

### Step 0.3 — Error taxonomy (n=200, automated rule-based) ✅ COMPLETE (2026-03-21)

See `docs/analysis/error_taxonomy_0321.md`. Key finding:

```
dim_error       72%   ← model gets topology right, numbers wrong
wrong_primitive 13%   ← box() fallback instead of sketch+extrude (flat plates)
degenerate       6%
wrong_plane      5%
partial_geom     4%
feature_count    1%
```

RL reduces structural errors (wrong_plane 7→3, partial_geom 5→2) but dim_error fraction grows
(69%→74%): as structural failures are fixed, numeric precision is now the bottleneck.

### Step 0.4 — SFT vs RL per-case delta ✅ COMPLETE (2026-03-21)

See `docs/analysis/sft_rl_delta_0321.md`. All 8046/1725 cases.

| Combo | fixed | boosted | stable | regressed | broken | net ΔIoU |
|---|---|---|---|---|---|---|
| deepcad/img | 6.6% | 21.5% | 69.4% | 2.3% | 0.3% | **+7.10pp** |
| deepcad/pc | 6.0% | 11.8% | 74.0% | 6.9% | 1.2% | **+4.55pp** |
| fusion360/img | 12.0% | 23.8% | 59.2% | 4.3% | 0.6% | **+9.61pp** |
| fusion360/pc | 11.2% | 17.0% | 62.6% | 7.1% | 2.0% | **+8.35pp** |

RL improves both modalities. img benefits more (train_modality=img).
PC: 17.8% improve vs 8.1% degrade → still net positive, but RL is less efficient at improving PC.

### Step 0.5 — dim_error sub-classification ✅ COMPLETE (2026-03-21)

Script: `tools/analyze_dim_errors.py`. Full run: n=2816 dim_error cases across all 8 combos.
See `docs/analysis/dim_error_analysis_0321.md`.

**Key findings:**
- **local_feat = 98%** — normalised bounding box matches; IoU low due to internal structure (holes, cutouts, features)
- aspect_ratio (wrong proportions): only **1%** → rules out Option B (scale cue)
- vol_ratio breakdown (split for local_feat cases):
  - over_material (>1.1): **47–54%** — pred more solid than GT, consistent with missing subtractive ops
  - under_material (<0.9): **32–38%** — pred less solid, missing solid features or over-carved
  - near_volume_match (±10%): **14–20%** — vol matches but IoU low, feature position/size wrong
- RL PC mode wrong_primitive: **15% → 23%** — RL img training amplifies box() fallback on PC

**Note on vol_ratio**: "missing holes" is the most common story (over_material 47–54%) but NOT the only one.
Under_material (35%) means some cases the model generates too little material (missing bosses/features
or over-subtracted). CD reward handles all three sub-types because it is bidirectional (penalises both
extra pred surfaces and missing GT surfaces), including position-offset cases.

**Decision locked**: Option A (CD reward) is the right Phase 1 main route.
Option B (scale cue) ruled out. Option C (wrong_primitive curriculum) confirmed as orthogonal second line.

### Step 0.6 — pred code subtractive analysis ← OPTIONAL quick win (2h)

For over_material dim_error cases (vol_ratio > 1.1), parse pred `.py`:
- If **no `mode='s'`** in pred → model completely skipped subtractive op → over_material from missing cut
- If **has `mode='s'`** → model tried to subtract but got dimensions/position wrong

This tells us whether the problem is "model forgot to make a hole" vs "made hole with wrong params".
Informs curriculum design for Phase 1 (targeted vs general).
Not required before starting Phase 1 — can be done in parallel with implementation.

---

## Phase 1 — Targeted Fixes (direction confirmed by Step 0.5)

**Principle**: one minimal change at a time for clean ablation. Two parallel lines:
- **Main line**: CD precision reward (addresses 98% local_feat dim_error)
- **Side line**: wrong_primitive curriculum (addresses RL PC mode 23% box fallback)

### Option A ✅ SELECTED: Chamfer Distance precision reward

**Why**: local_feat = 98% of dim_error; vol distribution shows over/under/near — CD handles all three.
CD is bidirectional: penalises extra pred surfaces (over_material), missing GT surfaces (under_material),
and position-offset local features (near_volume_match). IoU alone gives weak signal when topology is right.

Formula: `R = α·R_iou + (1−α)·exp(−β·CD)` where CD already computed in scoring pipeline.
CD already available in metadata.jsonl (`mean_cd` field from evaluate.py).

**Hyperparams to sweep**: α ∈ {0.8, 0.9}, β ∈ {5, 10, 20}. Start with α=0.9, β=10.
Change: `rl/reward.py` only. No architecture, data, or curriculum changes.

### Option B ❌ RULED OUT: Scale cue in input

aspect_ratio only 1% of dim_error → proportions are already correct. Scale is not the bottleneck.

### Option C ✅ SELECTED (side line): Wrong_primitive curriculum

RL PC mode wrong_primitive 15% → 23% (RL img training amplifies box fallback on PC).
Implementation: over-sample thin/flat GT shapes in RL training data; soft penalty if pred has no `.sketch()`.
Orthogonal to Option A — can implement and ablate independently.
Change: `rl/dataset.py` (sampling weights) + minor soft-reward in `rl/reward.py`.

### Option D: View augmentation (do before first H100 training run)

1-line change in `rl/dataset.py`. Random ±30° camera perturbation during rollout generation.
Reduces wrong_plane (5% of residual). Free, no compute overhead.
Add to first Phase 1 training run as baseline enhancement.

---

## Eval Results

### Full-set analysis (analyze_errors.py, n=8046/1725, 2026-03-21)


| Checkpoint             | img/DeepCAD | img/Fusion360 | pc/DeepCAD | pc/Fusion360 |
| ---------------------- | ----------- | ------------- | ---------- | ------------ |
| cadrille-sft           | 87.9%       | 79.6%         | 90.1%      | 83.8%        |
| cadrille-rl (official) | **92.7%**   | **85.6%**     | **90.7%**  | **86.0%**    |


Error breakdown (full test set):


| Combo             | success  | zero_iou | runtime_err | syntax_err |
| ----------------- | -------- | -------- | ----------- | ---------- |
| deepcad_sft_img   | 7785     | 127      | 132         | 2          |
| deepcad_rl_img    | **8001** | **38**   | **7**       | 0          |
| deepcad_sft_pc    | 7631     | 218      | 178         | 19         |
| deepcad_rl_pc     | **7986** | **46**   | **10**      | **4**      |
| fusion360_sft_img | 1625     | 54       | 46          | 0          |
| fusion360_rl_img  | **1705** | **18**   | **2**       | 0          |
| fusion360_sft_pc  | 1566     | 76       | 68          | 15         |
| fusion360_rl_pc   | **1693** | **22**   | **5**       | **5**      |


### Key findings from Step 0.2

1. **RL improves BOTH img and pc** — contrary to earlier (smaller-sample) eval showing img regression.
  Official rl checkpoint: +4.8pp img/DC, +6.0pp img/F360, +0.6pp pc/DC, +2.2pp pc/F360.
2. **RL dramatically reduces failures** — runtime_error drops 18× on img/DC (132→7), zero_iou drops 3×.
  This is the main driver of IoU gain: model generates executable code far more reliably.
3. **img > pc after RL on DeepCAD** (92.7% vs 90.7%) — surprising reversal from SFT (87.9% vs 90.1%).
  RL training (train_modality=img) specifically optimises img mode, possibly at minor pc cost.
4. **Fusion360 gap persists** — RL img 85.6% vs pc 86.0% (near parity). SFT img 79.6% vs pc 83.8%.
  F360 shapes are generally harder; both modalities near parity after RL.
5. **Paper target gap** — official rl img/DC = 92.7%. Our training target = 92.2% (Table 2). Already met.

---

## Key Observations (from 0319 session)

- `clip_fraction ≈ 0.001–0.005` in run p0ui4ehg: policy barely moving (lr too low)
- Training reward rising (0.33→0.46) but validation flat → overfitting to 22,970 hard examples
- img/DeepCAD −0.8pp after RL (small-sample eval was misleading — full set shows +4.8pp)
- **REVISED**: img and pc both improve under RL; img improves MORE than pc on DeepCAD

## Code Changes Already Made (in Docker /workspace, 2026-03-19)

- `rl/dataset.py`: `CurriculumRLDataset` (3-phase difficulty expansion)
- `rl/algorithms/cppo.py`: `adv_std_norm`, `soft_invalid` param
- `rl/reward.py`: `soft_invalid` distinction (syntax=-1.0, runtime=configurable)
- `rl/config.py`: new config fields for curriculum/adv_std_norm/soft_invalid_reward
- `configs/rl/h100.yaml`: `eps_high=0.2`, curriculum options
- **Note**: these changes are in Docker container, may not be on host

---

## Data & Checkpoints


| Item            | Location                                 |
| --------------- | ---------------------------------------- |
| DeepCAD test    | `data/deepcad_test_mesh/` (8,047 STLs)   |
| Fusion360 test  | `data/fusion360_test_mesh/` (1,726 STLs) |
| cadrille-sft    | `checkpoints/cadrille-sft/`              |
| cadrille-rl     | `checkpoints/cadrille-rl/`               |
| Analysis output | `data/analysis/` (gitignored)            |
| W&B             | `hula-the-cat/cadrille-rl`               |


---

## Immediate Next Steps

**Phase 0 (analysis):**
- [x] Step 0.1: Full inference run (8046/1725 cases × 2 models × 2 modalities)
- [x] Step 0.2: IoU distribution + failure breakdown
- [x] Step 0.3: Error taxonomy (200 cases, automated, 6 categories)
- [x] Step 0.4: SFT vs RL per-case delta (all cases, full breakdown)
- [x] **Step 0.5**: dim_error sub-classification → `tools/analyze_dim_errors.py`, results in `docs/analysis/dim_error_analysis_0321.md`
  - **local_feat = 98%** of all dim_error cases (n=2816); aspect_ratio only 1%
  - Model gets overall proportions right; fails on holes/cutouts/internal features
  - Vol ratio > 1.0 (median 1.07–1.15): model over-generates material → missing subtractive ops
  - RL PC mode: wrong_primitive rises from 15% → 23% (box fallback worsens in PC under img RL)
- [ ] **Step 0.6** (optional): Predicted parameter distribution analysis (SFT vs RL)

**Phase 1 (training — direction confirmed, ready to implement):**
- [ ] **Step 1a**: Implement CD reward in `rl/reward.py` (Option A, main line)
- [ ] **Step 1b**: Add view augmentation in `rl/dataset.py` (Option D, free win)
- [ ] **Step 1c**: wrong_primitive curriculum in `rl/dataset.py` (Option C, side line, can be parallel)
- [ ] Run H100 training with 1a+1b, eval at 360/720/1080 steps vs baseline
- [ ] Ablate: CD-only vs IoU-only to measure CD's contribution
- [ ] Step 0.6 (optional, 2h): pred code subtractive analysis — can run in parallel with training
- [ ] H100 runs: continue lr1e-5 runs to convergence (~3600 steps), eval at 720/1080/1440 step milestones

---

## Archive — Original RL Reproduction Plan

*(Kept for reference; reproduction is done, focus is now NeurIPS research)*

- Mining results: DeepCAD 4,173 hard / Fusion360 2,688 hard at R_th=0.75
- GC bug root cause confirmed and fixed in cppo.py:232-300
- H100 entropy explosion on restart was due to `gradient_checkpointing_disable()` missing in `generate_rollouts()`

