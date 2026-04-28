# Big BenchCAD-shell SFT 50k — phase-2 mix (20260427)

Run started 2026-04-27 09:08 UTC. **From scratch** (replaces v1 killed at step 4000). Target 50,000 steps.

Run name: `sft-s50k-lr2e-4-b8a4-img-0427-0908`
WandB: https://wandb.ai/hula-the-cat/cadrille-sft/runs/21sz2vje
HF repo: https://huggingface.co/Hula0401/cadrille-qwen3vl-2b-50pct-recode-50k

## Goal (per user 2026-04-27)

- 40–60% of every step's draws come from `text2cad_bench + recode_bench` (high-quality / cadrille-paper-style geometry, closest to DC/FU eval distribution)
- equal per-item frequency within `(text2cad_bench, recode_bench)`
- equal per-item frequency within `(benchcad, cad_iso_106, benchcad_simple)`

## Per-source items (verified 2026-04-27 after recode_bench refetch from 94 k → 142 k)

| source          | items   | weight | step share | per-item (×1e-3) |
|-----------------|---------|--------|------------|--------------------|
| benchcad        | 18 167  | 18     | 3.4 %      | 0.99               |
| cad_iso_106     | 162 145 | 162    | 30.3 %     | 1.00               |
| benchcad_simple | 85 597  | 86     | 16.1 %     | 1.00               |
| text2cad_bench  | 76 238  | 94     | 17.6 %     | 1.23               |
| recode_bench    | 142 411 | 175    | 32.7 %     | 1.23               |
| **total**       | 484 558 | 535    | 100 %      |                    |

→ Combined high-quality (`text2cad+recode`) = **50.3 %**.
→ Within bench-stack: per-item exposure all ≈ 1.00 e-3 (equal, max diff ±0.01).
→ Within high-quality: per-item exposure both ≈ 1.23 e-3 (equal).

## vs v1 (killed)

|                       | v1 mix       | v2 mix |
|-----------------------|--------------|--------|
| benchcad              | 28.6 %       | 3.4 %  |
| cad_iso_106           | 28.6 %       | 30.3 % |
| benchcad_simple       | 21.4 %       | 16.1 % |
| **text2cad_bench**    | 7.1 %        | **17.6 %** |
| **recode_bench**      | 14.3 %       | **32.7 %** |
| high-quality combined | 21.4 %       | **50.3 %** |
| benchcad per-item exp.| 25 epochs    | 1.06 epochs |
| recode per-item exp.  | 2.4 epochs   | 1.97 epochs |

→ benchcad exposure cut by 24× (from over-saturation to barely 1 epoch).
→ recode_bench gets 2.3× more step share AND 50 % more items (refetch).
→ Bench-stack still gets ~50 % combined (still useful data, just not dominating).

## Schedule

50 000 steps × eff_bs 32 = 1.6 M total samples drawn:
- benchcad: 1.6 M × 3.4 % = 54 k → 3.0 epochs
- cad_iso_106: 1.6 M × 30.3 % = 485 k → 3.0 epochs
- benchcad_simple: 1.6 M × 16.1 % = 258 k → 3.0 epochs
- text2cad_bench: 1.6 M × 17.6 % = 282 k → 3.7 epochs
- recode_bench: 1.6 M × 32.7 % = 523 k → 3.7 epochs

A100 ETA: ~25–30 hours at 30 steps/min.

## Eval-on-start metric (step 0, base Qwen3-VL-2B before SFT)

To be filled in after step-0 eval completes.
