# Status: KILLED at step 4000 (2026-04-27 09:05 UTC)

This run was stopped early (8% through) and replaced by `big_bench_shell_50k_phase2_20260427` with a recode-heavy mix.

## Why killed

At steps 1k–3k, big-50k consistently underperformed curriculum on Fusion360 by 6–17 IoU points (worst at step 2k: −0.144). Root cause was mix imbalance: 78.6% of every step's draws came from bench-style geometry (benchcad + cad_iso_106 + benchcad_simple = 11/14), but Fusion360's geometry distribution doesn't match bench-shell. Curriculum's phase-2 ran 90% recode-v1.5 and that's the source whose geometry lines up with DC/FU eval.

## Final IoU snapshot (step 4000)

| bucket          | step 4000 IoU | vs curriculum step 4000 |
|-----------------|----------------|---------------------------|
| BenchCAD val    | 0.356          | −0.033 🟡                 |
| DeepCAD test    | 0.384          | +0.021 🟢                 |
| Fusion360 test  | 0.405          | −0.006 🟡 (had recovered) |

By step 4000 the FU gap was nearly closed (−0.006), but BC val started oscillating downward (0.405 → 0.356 from step 3k → 4k). Combined with the early bench-overfitting trajectory, decision was to start fresh with a 50/50 high-quality mix rather than wait it out.

## What we learned (for future mix design)

1. **Bench-stack overrepresentation hurts FU early.** Need ≥40% non-bench data from step 0 to keep FU on a healthy trajectory.
2. **Per-item exposure matters more than total weight.** benchcad got 25 epochs while recode_bench got only 2.4 — even though their step-share weight was 4 vs 2. Exposure imbalance creates a model that's confident on bench but undertrained on recode.
3. **Eval order: BC val IoU is leading indicator** of overfitting (oscillates first). FU IoU is leading indicator of distribution mismatch (drops linearly).
