# Paper Storyline — §7 Training Probe

> **One-liner:** Adding our BenchCAD shell-style data improves rare-op recall and essential_pass on seen families, but cannot teach family-level op composition on held-out families. This delineates what SFT can/cannot learn from the dataset.

## Paper §7 main text (proposed)

### §7.1 Setup
Two SFT runs, Qwen3-VL-2B, 50k steps, batch 32, cosine LR 2e-4:
- **v3 baseline**: 106 families, 36% HQ / 64% bench-stack mix
- **v4-holdout**: 10 families held out (15205 samples ≈ 1.1% of train), +80k benchcad-easy supplement, 60% HQ / 40% bench-stack mix

10 holdout families selected by criterion: v3 baseline `essential_pass ≥ 0.80` (i.e., families demonstrably learnable when seen). Single source: `configs/sft/holdout_families.yaml`.

Online eval at every 1000 steps splits BenchCAD val into stratified IID (n=50) + OOD (n=50 = 10 families × 5).

### §7.2 Finding 1 — Op recall transfers
On held-out families, v4 `rare_recall` climbs **0.17 → 0.78** during training. v3 (which saw these families) tracks 0.11 → 0.83. **Both runs improve at similar rates on the same OOD set.** *[Figure: §7.a 4-line rare_recall vs step]*

→ The model learns from other families that "revolve / sweep / shell" exist as productive moves and emits them on novel inputs.

### §7.3 Finding 2 — Op composition does not transfer
Same OOD samples, `essential_pass` (per-family AND-of-OR check):
- v4: 0.22 → 0.44 (plateaus from step 6k onward)
- v3: 0.22 → 1.00 (climbs to ceiling)
- **Gap stable at -0.4 to -0.6 from step 6k forward.**

Per-family decomposition (e.g., dome_cap v3=0.93 → v4=0.26, bolt v3=0.79 → v4=0.04). The model emits rare ops but does not combine them into the correct family-level structure. *[Figure: §7.b 4-line ess_pass vs step + per-family bar]*

### §7.4 Finding 3 — IID metrics matched
On the IID subset, v4 `rare_recall`, `essential_pass`, IoU all track v3 throughout training. No degradation from the recipe change. *[Figure: §7 panel (d) IID control overlay]*

→ The recall-vs-composition decoupling is specifically a family-level transfer phenomenon, not a model bug.

### §7.5 Interpretation
BenchCAD = scattered valuable points in high-dimensional op-composition space (vs continuous manifolds in dense datasets like DeepCAD).
- **op recall** is local (memorize "rare op X exists at this point")
- **op composition** is global (need cross-family manifold knowledge)

SFT's next-token CE assumes local smoothness — works for the former, not the latter. Adding 80k more benchcad-easy samples (denser local sampling) improves IID rare_recall but doesn't change OOD essential_pass plateau.

### §7.6 What this means for the dataset
Our dataset's value is not "more samples" — it is:
1. Each family is a representative point of an independent cell in op-composition space.
2. The failure mode of current SFT recipes on this data **is** the dataset's contribution — it surfaces real model+method boundaries.
3. Future directions: family taxonomy expansion, cell-aware loss, family-level supervision, RL with composition reward.

## Figures (in `paper/figures/`)

| Figure | File | Shows |
|---|---|---|
| Main §7.a | `fig_7_4line_ess_pass.png` | 4-line `essential_pass` rate × step (IID / OOD-noeasy / OOD+easy / no-bench) |
| Main §7.b | `fig_7_ood_iou_4line.png` | 4-line OOD IoU × step |
| Main §7.c | `fig_main_recall_vs_composition.png` | 4-panel: OOD rare_recall climb / OOD ess_pass plateau / final-step gap / IID control |
| Appendix A | `fig_app_per_family.png` | Per-OOD-family ess_pass (5 families × v3/v4 bars) |
| Appendix B | `fig_app_iid_gains.png` | IID rare_recall + feat_F1 + recall trajectories |
| Supp. ops | `v4_ops_metrics_2026-04-30.png` | 12-panel ess_pass + feat_F1 + op_entropy × {IID,OOD,DC,Fu} |
| Supp. cases | `v4_ood_grid_full_2026-04-30.png` | 9 OOD samples × 9 eval steps mesh trajectory |
| Supp. failure | `v4_failure_per_family_2026-05-01.png` | per-OOD-family IoU + exec + ess bars |

## Limitations (§7.6 + §8 caveats)

- Single seed; current OOD bucket n=50 stratified (10 fams × 5) — not yet n=200+ for tight CI
- Comparison vs v3 confounds three changes (mix ratio, +benchcad-easy, holdout). Disentangling needs v4-baseline (control: same recipe, no holdout) — pending.
- `essential_pass` is one specific metric; corroborated by `feature_F1` and `IoU` showing same OOD pattern.
- Single benchmark (BenchCAD families). DeepCAD/Fusion360 lack GT codes so can't compute essential_pass on those buckets.

## Pending experiments (post-merge)

1. **v4-baseline 50k** — control: same 60/40 mix + benchcad-easy, no holdout. Tests whether OOD gap is from holdout vs from recipe change.
2. **v4-holdout-noeasy 50k** — same holdout, no benchcad-easy. Tests benchcad-easy supplement effect on OOD.
3. **v4-hq-only 50k** — text2cad + recode_bench only (no bench-stack). Floor baseline.

After (1)-(3), the §7.a/§7.b 4-line plots become fully populated (current state has placeholder zeros for lines 2 and 4).
