# §7 v2 — pre-registered storylines per result outcome

The 5-line ablation sets up four data axes:
- **bench-stack vs HQ**:    baseline → ood / ood_enhanced (bench helps?)
- **mech context**:         ood vs ood_enhanced (10 mech in/out)
- **easy supplement**:      baseline → iid_enhanced (plate primitives help?)
- **atomic op coverage**:   any line → iid_v2 (saw 44 of 54 op patterns)

Each combination of (ess_pass, IoU) outcomes → a different paper narrative.

## Reading the figure (5 lines, 2 metrics → 10 numbers per ckpt)

x-axis: training step.
y-axis: §7.a essential_pass / §7.b IoU on the SAME stratified 50-OOD
(10 simple_op patterns × 5 cases each, online_eval seed=42).

```
                   essential_pass        IoU
                   on 50 OOD             on 50 OOD
                   ─────────────────     ─────────────────
   iid_v2          (1) saw 44/54         (a) saw 44/54
   ood_enhanced_v2 (2) bench + mech 10   (b) bench + mech 10
   ood_v2          (3) bench, no mech    (c) bench, no mech
   iid_enhanced_v2 (4) HQ + plates only  (d) HQ + plates only
   baseline_v2     (5) HQ only           (e) HQ only
```

---

## Outcome A: clean monotonic ladder (1) > (2) > (3) > (4) > (5)

> "Each data ingredient contributes independently — atomic op coverage > mech
> context > bench-stack > plates > nothing."

### Story (paper-ready)
- Composition is unlocked by **atomic op coverage** (iid_v2's 44 fams of
  bench-simple). Holding out the 10 OOD patterns still lets the model
  generalise *across patterns* — direct evidence for "ops are
  compositional units the model learns to recombine".
- Removing op-pattern coverage (ood_enhanced) drops ess_pass by Δ₁ — quantifies
  the value of op-pattern training data.
- Removing the 10 mech context (ood) drops by Δ₂ — quantifies value of
  in-domain shape context for op composition.
- Removing bench-stack entirely (iid_enhanced) drops by Δ₃ — easy plates
  alone are insufficient.
- baseline floor → magnitude of total data contribution.

### Confidence: HIGH (clean ablation, sums to ceiling)
This is the publication-quality outcome. §7 main figure in the paper.

---

## Outcome B: enhance helps neither metric (iid_enhanced ≈ baseline)

> "Plate primitives (benchcad-easy) provide no transfer to op-pattern
> composition; bench-stack alone is the active ingredient."

### Implication
- Drop iid_enhanced from the §7 main figure (or relegate to appendix as
  a negative control).
- Strengthens the framing: "the dataset's *value* is bench-stack
  (mechanical parts and op-pattern primitives), not raw row count."
- Reviewer-proof against "they just added more data" attack.

### Risk
- Dataset paper claim weakens — we now claim "of our 4 sub-datasets,
  benchcad-easy is decoration". Ok if the other 3 carry the story.

---

## Outcome C: enhance HURTS (iid_enhanced < baseline OR ood_enhanced < ood)

> "Wrong augment is a distractor — easy plate primitives or extra mech
> samples can dilute the rare-op signal that drives op composition."

### Implication
- This is **already partially observed**: ood_enhance < ood on ess_pass at
  step 24k by ~0.2 (50-stratified verified). v2 will confirm + extend.
- Story flips from "more data is better" to "**aligned data is better**".
- Add a §7.5 paragraph: "Why benchcad-easy hurts" — visual op-profile
  argument (`paper/figures/easy_vs_ood_render_grid.png`).
- Strong paper claim: "BenchCAD's contribution is not size but
  *op-composition coverage* — adding off-profile data hurts."

### Risk
- Reviewer asks: "did you screen your augment data? what's the principle?"
  → answer: yes, we measured op-profile (figure already in paper); easy
  is plate-extrude profile, OOD is revolve-cut-shell profile.

---

## Outcome D: ess_pass UP, IoU DOWN (or vice versa) — metric divergence

This is the **most interesting failure mode** for the paper.

### D1: enhance pushes ess_pass UP but IoU DOWN
> "Augment teaches the right ops but not the right shapes."

Means: model sees more op composition examples, learns to emit `revolve`
+ `cut` (ess passes), but the *parameters* (radius, position) drift away
from training distribution → IoU drops.

**Story**: separate the two signals at the metric level — paper's §7
becomes a **two-axis claim**: data unlocks composition; shape precision
needs RL or more data. → bridges directly into §8.

### D2: enhance pushes IoU UP but ess_pass DOWN
> "Augment teaches shape memorization at the cost of op generalization."

Means: model learns to memorize plate shapes from easy data → IoU on
plate-y OOD samples climbs, but stops emitting the held-out op patterns
(stays in the easy plate-extrude regime).

**Story**: warning about **mode collapse on augment data**. ess_pass is
the canary — without it, IoU alone misleads. paper §7 conclusion:
"a dataset paper for compositional generalisation cannot evaluate by
IoU alone — need explicit op composition metric like essential_pass."

This is actually a **more compelling paper** than Outcome A.

### D3: ood ≠ ood_enhanced on different axes
e.g. ood has higher ess_pass, ood_enhanced has higher IoU.

> "Mech context buys shape precision; op-pattern density buys composition."

Each ablation isolates one improvement axis. Strongest paper.

---

## Outcome E: all bench-using lines saturate (ess ≈ ceiling for all but baseline)

> "BenchCAD's 60% bench-stack is enough; specific op-pattern coverage adds
> nothing beyond ~80% ess_pass."

### Implication
- Surprising! Means bench-stack alone (without op-pattern primitives)
  generalises to atomic op patterns.
- Paper claim becomes: "the mechanical part data implicitly contains
  enough op composition to transfer to atomic patterns".
- Counterevidence to the "atomic primitives are needed" hypothesis.
- Still sells: "Our dataset's mechanical-part subset alone is sufficient
  for op-composition transfer; op-pattern primitives are belt-and-braces."

### Risk
- This makes benchcad-simple look redundant in paper — but argue:
  benchcad-simple is the *evaluation* set that lets us *prove* this.

---

## Decision tree (pseudocode)

```
ess_pass at step 50k:
    iid_v2 ≥ 0.80?  → ablation has signal headroom (good)
        ood_v2  ≥ 0.40?  → bench-stack alone unlocks composition (Outcome A or E)
            iid_enhanced_v2 ≥ ood_v2?    → easy is enough (rare; reframe)
            iid_enhanced_v2 < baseline   → easy hurts (Outcome C)
            iid_enhanced_v2 ≈ baseline   → easy is null (Outcome B)
        ood_v2 ≪ 0.20?   → bench-stack insufficient; mech context essential
            ood_enhanced_v2 > ood_v2 by Δ → Outcome A
            ood_enhanced_v2 ≈ ood_v2     → mech context redundant (Outcome E variant)
    iid_v2 < 0.50?  → ablation saturated low; metric or eval pool issue
        → diagnostic: is essential_pass spec too strict? bump to 'binary' off
```

---

## Action items — locked in NOW (before we have results)

1. **§7.a + §7.b figures**: 5 lines, same 50 OOD, all post-train. Already
   coded (`scripts/analysis/plot_main_appendix.py`).
2. **§7 appendix**: per-family ess_pass breakdown (10 fams × 5 ckpts = 50
   bars per line). Distinguish Outcome D1 vs D2 by which families gain/lose.
3. **§7.5 reflection**: dataset selection rationale (with
   `easy_vs_ood_render_grid.png`).
4. **§7.6 limitation**: single seed; 2 seed re-run for headline numbers
   (Outcome A or D needs CI proof).
5. **§8 follow-up**: pure-IoU RL vs ess+IoU RL on the BEST §7 ckpt.
   Direct measurement of Outcome D's "shape precision needs RL" claim.

The framework code accommodates all 5 outcomes — only the figure caption
text changes per outcome.
