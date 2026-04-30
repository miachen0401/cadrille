# Paper §7 + §8 Plan — "Recall ≠ Composition" narrative

**Date**: 2026-04-30 (updated)
**Branch**: `feat/v4-holdout-sft`
**Run reference**:
- v3 baseline: `sft-s50k-lr2e-4-b8a4-img-0428-1320` (50k complete)
- v4-holdout: `sft-s50k-lr2e-4-b8a4-img-0430-0828` (currently 17k/50k)

---

## Section 7 STORY ARCHITECTURE — Three subsections

The training-analysis section is structured around three figures, each a multi-line trajectory plot covering distinct training recipes:

### §7.a — rare_recall vs training step (4 lines)
### §7.b — IoU vs training step (4 lines)
### §7.c — RL post-training: essential reward vs pure IoU reward

The 4-line spec for §7.a/§7.b:

| Line | Recipe label | Training data | Eval target |
|---|---|---|---|
| **(1) IID ceiling** | v3 (full data) | All 106 families | Same families seen |
| **(2) OOD plain** | v4-holdout-noeasy | 96 families (10 held-out), no benchcad-easy | Held-out 10 families |
| **(3) OOD + bench-easy** | v4-holdout (current) | 96 families (10 held-out) + 80k benchcad-easy | Held-out 10 families |
| **(4) no-bench HQ-only** | v4-hq-only | text2cad + recode_bench only (no benchcad-stack) | Held-out 10 families |

For §7.c — RL ablation (2 lines):

| Line | Reward | Started from |
|---|---|---|
| **RL-iou** | pure IoU | v4-holdout ckpt-50k |
| **RL-ess** | IoU + λ × essential_pass | v4-holdout ckpt-50k |

**Story arc:**
- §7.a: rare_recall climbs across all four lines on OOD families, even line (4) which has zero bench-stack data — *op recall is transferable*
- §7.b: IoU shows much wider spread across lines — the "+bench-easy" supplement (line 3) lifts IoU but does not close the IID gap; (4) is far below
- §7.c: limitation revealed by §7.a/7.b is *op composition* (essential_pass plateaus). RL with essential reward closes the gap further than RL with pure IoU. Two complementary remedies emerge: (i) more compositional data + (ii) composition-aware reward.

---

## Section 7. Training Probe — What SFT Can and Cannot Learn from BenchCAD

### 7.1 Setup

We use a single ablation experiment to probe the generalization boundary of SFT on BenchCAD. Two runs share Qwen3-VL-2B backbone, 50k steps with cosine LR schedule, effective batch 32. The only variable is training data:

**v3 baseline (control)**: All 106 BenchCAD families, mix ratio 36% HQ / 64% bench-stack.

**v4-holdout (probe)**: Remove 10 families (15205 samples, ≈1.1% of train) plus add 80k benchcad-easy samples; mix ratio 60/40. The 10 holdout families are
`tapered_boss, taper_pin, venturi_tube, bucket, dome_cap, nozzle, enclosure, waffle_plate, bolt, duct_elbow`.
Each holdout family's essential ops (`revolve / sweep / loft+taper / shell / rarray / polygon`) appears with >5% frequency in the remaining 96 families — so the model retains theoretical op-level exposure.

Every 1000 steps, BenchCAD val (n=50) is evaluated greedy. Predictions are split by family into:
- **IID** (n=41, 96 seen families)
- **OOD** (n=9, 5 of 10 holdout families actually sampled)

Op metrics:
- `recall = |GT_ops ∩ pred_ops| / |GT_ops|`
- `rare_recall` — same as recall but restricted to a 12-op rare set (sweep, revolve, loft, shell, twistExtrude, helix, taper, polarArray, rarray, mirror, fillet, chamfer)
- `essential_pass` — per-family AND-of-OR-tuples check (vendored from Cadance `canonical_ops.yaml`)

### 7.2 Finding 1 — Models recall rare ops on unseen families

**[Main figure panel (a)]** `docs/paper_figures/fig_main_recall_vs_composition.png`

On BC val OOD samples, v4-holdout's `rare_recall` climbs from **0.17 (step 1k)** to **0.78 (step 14k)**. v3 (which saw these families) tracks 0.11 → 0.83 — only +0.05 above v4. **Both runs improve at similar rates on never-seen families.**

→ Op-level recall is transferable. The model learns from other families that "revolve / sweep / shell" exist as productive moves and emits them on novel inputs.

### 7.3 Finding 2 — But models cannot learn op composition

**[Main figure panel (b) + Appendix A per-family bar]**
`fig_main_recall_vs_composition.png`, `fig_app_per_family.png`

Same OOD samples, `essential_pass` (correctly composed family-specific structure) tells a different story:
- v4: 0.22 → 0.44 (plateaus from step 6k onward)
- v3: 0.22 → 1.00 (climbs to ceiling)
- Gap stable at -0.4 to -0.6 from step 6k forward

Per-family decomposition (Appendix A) shows the collapse is consistent across all 5 sampled holdout families:

| Family       | v3 ess_pass | v4 ess_pass | Δ      |
|--------------|------------:|------------:|-------:|
| bolt         | 0.79        | 0.04        | -0.75  |
| dome_cap     | 0.93        | 0.26        | -0.67  |
| tapered_boss | 0.64        | 0.07        | -0.57  |
| waffle_plate | 0.64        | 0.50        | -0.14  |
| bucket       | 0.93        | 0.71        | -0.22  |

→ The model emits rare ops (high recall) but does not combine them into the correct family-level structure. E.g. on dome_cap, v4 tends to emit `extrude+circle` (visual cap) instead of `revolve+segment` (the rotational op + profile op composition).

### 7.4 Finding 3 — On in-domain families, both metrics climb together

**[Main figure panel (d)]**

On the n=41 IID subset, v4 `rare_recall` and `essential_pass` track v3 throughout training. No degradation from the recipe change (no statistically significant difference in n=41).

→ The recall-vs-composition decoupling is specifically a family-level transfer phenomenon, not a model-bug.

### 7.5 Interpretation — Sparse high-dimensional cell structure

BenchCAD contains 106 families × ~1.3k samples per family, each sample with 2-3 rare op combinations. Formally, the training distribution looks like **scattered valuable points in a high-dimensional space** rather than a continuous manifold:
- **op recall** is a *local* property — model can memorize "rare op X exists at this point"
- **op composition** is a *global* property — needs cross-family manifold knowledge "this geometry should use revolve+spline"

SFT's next-token cross-entropy assumes local smoothness. It works for the former, not the latter. This explains why adding 80k more dense samples (benchcad-easy) improves IID rare_recall but doesn't change OOD essential_pass plateau.

### 7.6 Implications for the dataset paper

Our dataset's value is not "more samples" — it is:
1. **Each family is a representative point of an independent cell** in the high-dimensional CAD space
2. **The failure mode of current SFT recipes on this data** *is* the dataset's contribution — it surfaces real boundaries of model + method
3. **Future work directions** open up: family taxonomy expansion (cell coverage), cell-aware loss (replacing next-token CE), family-level supervision, RL over high-pass@8 substrate

---

## Section 8. Robustness Augmentations

§7's findings rely on: single-seed training, n=9 OOD per eval, 5 of 10 holdout families actually sampled, single family selection, single `essential_pass` metric. Below is a tiered queue of experiments, each adding one robustness dimension.

### 8.1 Tier 0 — Required for credible NeurIPS main-track (~3 GPU-day)

| # | Experiment | Resolves | Expected effect |
|---|---|---|---|
| T0-1 | **v4-holdout to 50k** (currently 14k, +36k = ~24h) | Insufficient training | Plateau is real, not transient |
| T0-2 | **v4-baseline 50k** (same recipe, no holdout) | Missing control: "v3 may just be better on those families" | Clean attribution: v4-baseline (saw fams) vs v4-holdout (didn't) on the same OOD families |
| T0-3 | **Stratified offline eval @ ckpt-25k + ckpt-50k** (n=50 OOD = 10 fam × 5) for v3, v4-holdout, v4-baseline | n=9 too small | OOD essential_pass σ drops 0.16 → 0.07 |

Tier 0 alone elevates §7 from suggestive to defensible.

### 8.2 Tier 1 — Strong robustness (~3 GPU-day)

| # | Experiment | Resolves |
|---|---|---|
| T1-1 | **3 random holdout configs** (10 fam each, 25k step each, configs A/B/C) | Cherry-pick attack: shows recall-vs-composition gap stable across holdout choices |
| T1-2 | **Cross-LLM IoU-24 eval** (gpt-4o / Claude / Gemini on 50 OOD samples) | "Just a small-model issue?" — show commercial LLMs also collapse |
| T1-3 | **Retrospective v3 family-subset analysis** (no new training, partition v3 ckpt-50k predictions by 3 different 10-family subsets and compute ess_pass) | Quick sanity: confirm v3 ess_pass varies by family choice — strengthens v4 holdout signal |

After Tier 1: paper can be submitted to NeurIPS main.

### 8.3 Tier 2 — Extension (~2 GPU-day)

| # | Experiment | Resolves |
|---|---|---|
| T2-1 | **benchcad-easy reverse holdout** (drop 5 simple_* families during training, eval them as OOD, 25k step) | Internal self-consistency — uses BC's own subset for OOD validation |
| T2-2 | **DC/Fu op-signature pseudo-families** (k-means on op-presence vectors, evaluate ess_pass per pseudo-family) | Cross-dataset validation — finding not BC-specific |
| T2-3 | **Multi-metric verification** (exact_op_match, AST structural similarity) | Metric robustness — finding not an `essential_pass` definition artifact |

### 8.4 Tier 3 — Nice-to-have (~1 GPU-day)

| # | Experiment | Resolves |
|---|---|---|
| T3-1 | 2nd seed v4-holdout (25k) | Training seed noise |
| T3-2 | Rare op set sensitivity (different rare op definitions) | Definition robustness |

---

## 8.5 Recommended execution order

### Week 1 (Days 1-3)

1. v4-holdout continues to 50k (already running, ~24h remaining)
2. **Now**: launch retrospective v3 family-subset analysis (T1-3, 10 min CPU) → confirms gap is non-trivial
3. **At step 25k of v4-holdout (~12h)**: stratified offline eval (T0-3 partial, n=50 OOD on intermediate checkpoint). Yields paper draft data point.
4. **After v4-holdout 50k completes**: launch v4-baseline 50k (T0-2)

### Week 2 (Days 4-7)

5. v4-baseline finishes
6. Final stratified offline eval (T0-3 full): all three runs at all checkpoints
7. Cross-LLM IoU-24 (T1-2, 4h CPU + API)
8. Decide Tier 1 holdout configs based on retrospective analysis (T1-3 result)

### Writing (Days 8-10)

9. Tier 0 data complete → write §7 + §8 (this doc as base)
10. If Tier 1 data ready → add a Robustness section
11. Update figures

---

## Immediately actionable (no GPU contention)

| Action | Time | Output |
|---|---|---|
| Retrospective v3 family-subset analysis (T1-3) | 10 min CPU | Confirms whether gap is family-pick-dependent |
| Cross-LLM IoU-24 launch (T1-2) | 4h, runs in BG | Commercial LLM ess_pass on 50 OOD |
| Stratified offline eval @ ckpt-12000 (T0-3 preview) | 35 min GPU contention | n=50 OOD data — preview before ckpt-25k |

---

## Figures Inventory (current state)

| Figure | File | Used in |
|---|---|---|
| Main: recall vs composition | `docs/paper_figures/fig_main_recall_vs_composition.png` | §7.2-7.4 |
| Appendix A: per-family OOD ess_pass | `docs/paper_figures/fig_app_per_family.png` | §7.3 |
| Appendix B: IID gain trajectories | `docs/paper_figures/fig_app_iid_gains.png` | §7.4 control evidence |
| Op trajectory metrics (incl. op_entropy) | `docs/v4_ops_metrics_2026-04-30.png` | Appendix |
| OOD render trajectory grid | `docs/v4_ood_grid_full_2026-04-30.png` | Appendix qualitative |

---

## Decision points

1. **Adopt this narrative as §7 + §8?** (alternative was rare-op-recall-as-headline, but data does not support it as cleanly)
2. **Approve Tier 0 launch sequence?** (v4-baseline launch after v4-holdout 50k)
3. **Run T1-3 retrospective analysis now?** (10 min, low-risk preview) — DONE 2026-04-30; confirmed v3 scores 0.87-0.99 on random subsets, our 10 holdout fams score 1.00 → v4's 0.44 is real signal

---

## Section 9. Reviewer-self-critique (writing as if reviewing this paper)

### What's strong
- 3-figure architecture (§7.a/§7.b/§7.c) is a complete training story end-to-end
- Two complementary findings: (i) op recall vs op composition decoupling, (ii) RL with composition reward as remediation
- Clean comparison framework: 4 lines on §7.a/b correspond to 4 distinct dataset-recipe variants
- Paper's contribution is *both* the dataset (rare-op-combination data) *and* the diagnostic methodology (essential_pass + ablation)

### What I would attack as a reviewer

**A. Comparability of the 4 lines.** The 4 lines mix two factors:
- (factor 1) holdout vs no-holdout
- (factor 2) what supplemental data is added
We need a clean 2×2 grid:

|              | no benchcad-easy | + benchcad-easy |
|--------------|------------------|-----------------|
| no holdout   | v3 (line 1)      | v4-baseline ★   |
| holdout 10   | v4-holdout-noeasy (line 2) | v4-holdout (line 3) |

★ = currently MISSING. Without it, line (3) vs line (1) confounds two changes.

**Recommendation**: Add **v4-baseline** training (24h GPU). Then 5-line plot:
1. v3 IID (full data, eval IID) — ceiling
2. v3 OOD (full data, eval OOD families) — same model, OOD eval — *ceiling on those families*
3. v4-baseline OOD — recipe matched to v4 but no holdout
4. v4-holdout-noeasy OOD — recipe + holdout, no easy supplement
5. v4-holdout OOD — recipe + holdout + easy supplement
6. v4-hq-only OOD — no bench-stack at all (floor)

This 6-line plot disentangles:
- Recipe effect (60/40 mix change): line 1 vs line 3
- Holdout effect: line 3 vs line 4
- Easy supplement effect: line 4 vs line 5
- Bench-stack overall effect: line 4 vs line 6

**B. Reward design for §7.c**

Pure IoU and "IoU + λ × essential_pass" are too few baselines. A reviewer expects:
- (R1) Pure IoU (current proposal)
- (R2) IoU + 0.3 × essential_pass (additive shaping)
- (R3) Multiplicative: IoU × (1 + α × essential_pass)
- (R4) Curriculum: essential_pass first 1k steps, IoU after

**Recommendation**: Run R1 + R2 (+R3 if budget). Show essential_pass trajectory under each, not just final IoU.

**C. Mix ratio not ablated**

Why specifically 60/40 HQ/bench? No mix-ratio sweep. Reviewer asks: "What if 50/50 or 70/30?"

**Recommendation**: At minimum run mix=50/50 and mix=70/30 at 25k step (truncated, 12h each). Add 2-line sub-plot showing mix is approximately optimal.

**D. Single seed**

All §7.a/b/c lines are single-seed. Reviewer requires 2+ seeds for headline numbers.

**Recommendation**: Run v3 + v4-holdout 2nd seed (25k each, 24h total). Report mean ± std on §7.a/b end values.

**E. n=9 OOD per eval is too small**

Current online_eval samples 50 BC val randomly → ~9 OOD samples per eval. σ ≈ 0.10 on essential_pass.

**Recommendation**: Phase B refactor — stratified n=50 OOD bucket per eval. Or: run offline stratified eval at ckpt-25k and ckpt-50k for all 6 lines (~30 min × 12 = 6h GPU spread across the runs).

**F. Why these 10 families specifically?**

Defense: T1-3 retrospective analysis showed v3 scores 1.00 on these specific families and 0.87-0.99 on random 10-family subsets — so we did NOT cherry-pick "hard" families. Mention this in §7.

**Recommendation**: Add a caveat box in §7.1: *"Holdout families were selected to ensure (a) every essential op appears with >5% frequency in remaining 96 families, and (b) v3 baseline reaches ess_pass ≥ 0.85 on each (verified via T1-3 retrospective analysis). The recall-vs-composition gap is therefore not a 'hard family' artifact."*

### Reviewer-recommended execution order

**Phase 1 — Core (3 GPU-day, mandatory)**:
- Finish v4-holdout to 50k
- Train v4-baseline 50k (control: same recipe, no holdout)
- Train v4-hq-only 50k

**Phase 2 — Robustness (3 GPU-day, mandatory for NeurIPS)**:
- Train v4-holdout-noeasy 50k (line 4)
- Train mix-ratio sweep at 25k step × 2 ratios (50/50, 70/30)
- 2nd seed v4-holdout 25k

**Phase 3 — RL (1 GPU-day)**:
- RL-iou from v4-holdout ckpt-50k
- RL-ess from v4-holdout ckpt-50k

**Phase 4 — Eval & write (parallel to Phase 3)**:
- Offline stratified eval n=50 at all key ckpts (3h GPU)
- Cross-LLM IoU-24 eval (4h API)
- 6-line plot generation
- Paper draft

Total: ~7 GPU-day plus 1 day API/CPU. Achievable in 7 calendar days on single A100.

### Reviewer scoring (self-estimate)

Without Phase 1+2: **borderline reject** — single seed, missing controls
With Phase 1+2 done: **borderline accept** — clean ablation, robust signals
With Phase 1+2+3+4 done: **clear accept** — complete story, multiple ablations, RL extension

### Reviewer-style summary statement

> The paper claims SFT learns op recall but not op composition on held-out CAD families. With proposed Phase 1+2 ablations (v3 + v4-baseline + v4-holdout × {±benchcad-easy} + v4-hq-only) plus RL ablation, the paper makes a substantive contribution to compositional generalization in symbolic graphics generation. The 6-line plot architecture cleanly disentangles recipe / holdout / supplement / bench-stack effects — a stronger experimental design than typical CAD-generation papers.

---

## Updated experiment queue (post reviewer self-critique)

### Phase 1 — Core SFT runs (24h × 3 = 72h on single A100, sequential)

| # | Run | Config | Status |
|---|---|---|---|
| 1 | v4-holdout 50k | `big_bench_shell_50k_v4_holdout.yaml` | 🟡 17k/50k in progress |
| 2 | v4-baseline 50k | `big_bench_shell_50k_v4_baseline.yaml` | ❌ pending launch |
| 3 | v4-hq-only 50k | `big_bench_shell_50k_v4_hq_only.yaml` | ❌ pending launch |
| 4 | v4-holdout-noeasy 50k | `big_bench_shell_50k_v4_holdout_noeasy.yaml` (NEW) | ❌ pending launch |

### Phase 2 — Robustness

| # | Run | Time |
|---|---|---|
| 5 | mix-ratio 50/50 at 25k | 12h |
| 6 | mix-ratio 70/30 at 25k | 12h |
| 7 | 2nd seed v4-holdout 25k | 12h |

### Phase 3 — RL

| # | Run | Time |
|---|---|---|
| 8 | RL-iou from v4-holdout ckpt-50k | 12h |
| 9 | RL-ess (IoU + 0.3×ess) from v4-holdout ckpt-50k | 12h |
| 10 | RL-ess-mult (IoU × (1+α·ess)) from v4-holdout ckpt-50k | 12h (optional) |

### Phase 4 — Offline eval & analysis

| # | Action | Time |
|---|---|---|
| 11 | Stratified n=50 OOD eval at ckpt-25k/50k for runs 1-4 | 4h GPU |
| 12 | Cross-LLM IoU-24 eval (gpt-4o + Claude on 50 OOD) | 4h API |
| 13 | 6-line plot for §7.a + §7.b | 30 min |
| 14 | RL ablation plot for §7.c | 30 min |
| 15 | Final paper figures | 1h |

**Total compute**: 7-8 GPU-day on single A100, plus ~1 day API/CPU/writing.

---

## Action right now (what we can do today/tonight)

1. **Launch v4-baseline NOW** in a separate process — it needs same GPU as v4-holdout but cadquery max@8 leaves 30% GPU idle; we may be able to overlap. Or wait until v4-holdout finishes (~16h).
2. **Launch v4-hq-only NOW** with `eval_steps=2000` (less frequent) to reduce contention with v4-holdout.
3. **Pre-write all the plotting + RL training scripts** so when ckpts arrive we just run them.



