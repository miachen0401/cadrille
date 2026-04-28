# Big BenchCAD-shell SFT 50k (20260427) — data mix

Run started 2026-04-27 06:13 UTC, target 50,000 steps. **In progress.**

## Config (big_bench_shell_50k.yaml)

Single-phase weighted mix, 5 sources. `eff_bs = 32` (batch_size 8 × accum 4).

```yaml
sft_mix_weights:
  benchcad:         4
  cad_iso_106:      4
  benchcad_simple:  3
  recode_bench:     2
  text2cad_bench:   1
```

## Per-source items loaded (verified from local train.pkl)

| source              | items   | weight | step share | total samples (50k × 32) | per-item exposures |
|---------------------|---------|--------|------------|---------------------------|---------------------|
| benchcad            | 18,167  | 4      | 28.6%      | 457 k                     | **25.2 epochs** ⚠️  |
| cad_iso_106         | 162,145 | 4      | 28.6%      | 457 k                     | 2.82 epochs         |
| benchcad_simple     | 85,597  | 3      | 21.4%      | 343 k                     | 4.01 epochs         |
| recode_bench        | 94,894  | 2      | 14.3%      | 229 k                     | 2.41 epochs         |
| text2cad_bench      | 76,238  | 1      | 7.1%       | 114 k                     | 1.49 epochs         |
| **total**           | 437,041 |        | 100%       | 1,600 k                   |                     |

## Diagnosis (at step ~3000, ~6% of run)

big-50k vs curriculum, IoU at the same step (greedy / t=0):

| step | BenchCAD val      | DeepCAD test      | **Fusion360 test** |
|------|-------------------|-------------------|---------------------|
| 1000 | 0.196 (+0.076 🟢) | 0.270 (+0.058 🟢) | 0.284 (+0.010 🟡)  |
| 2000 | 0.348 (+0.091 🟢) | 0.301 (+0.010 🟡) | 0.257 (**−0.144 🔴**) |
| 3000 | 0.405 (+0.081 🟢) | 0.326 (+0.021 🟡) | 0.381 (**−0.057 🔴**) |

→ Big-50k is **comfortably ahead on bench-style data** (BenchCAD val +0.08
consistently) but **trailing curriculum on Fusion360** by 0.06–0.14 IoU.
The step-2000 dip on FU was real (−0.144) and the model is still recovering
(−0.057 at step 3000). Cause: 78.6% of step exposure is bench-style
geometry (benchcad + iso106 + simple = 11/14). Bench-style ≠ Fusion360
distribution. Need to up-weight recode_bench (cadrille-paper-style
geometry, closest to DC/FU eval) to fix this.

## Planned: phase-2 mix at step 10000

Goal: 50% combined exposure on (text2cad_bench + recode_bench),
equal per-item frequency within each group.

Item counts (after refetching cad-recode-bench to ~140k):

| source              | items   | proposed weight | step share | per-item (×1e-3) |
|---------------------|---------|-----------------|------------|--------------------|
| benchcad            | 18,167  | 18              | 3.4%       | 0.99               |
| cad_iso_106         | 162,145 | 162             | 30.5%      | 1.00               |
| benchcad_simple     | 85,597  | 86              | 16.2%      | 1.00               |
| text2cad_bench      | 76,238  | 94              | 17.7%      | 1.23               |
| recode_bench        | 140,000 | 172             | 32.3%      | 1.23               |
| **total**           |         | 532             | 100%       |                    |

Combined high-quality (text2cad+recode) = 50.0%. Within bench-stack:
per-item exposure equal at ~0.001. Within high-quality: per-item exposure
equal at ~0.0012. → Refer to `configs/sft/big_bench_shell_50k_phase2.yaml`
when written.
