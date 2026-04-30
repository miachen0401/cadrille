# Paper §7 + §8 Plan — "Recall ≠ Composition" narrative

**Date**: 2026-04-30
**Branch**: `feat/v4-holdout-sft`
**Run reference**:
- v3 baseline: `sft-s50k-lr2e-4-b8a4-img-0428-1320` (50k complete)
- v4-holdout: `sft-s50k-lr2e-4-b8a4-img-0430-0828` (currently 14k/50k)

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
3. **Run T1-3 retrospective analysis now?** (10 min, low-risk preview)

