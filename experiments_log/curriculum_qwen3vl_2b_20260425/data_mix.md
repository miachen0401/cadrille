# Curriculum Qwen3-VL-2B (20260425) — data mix

Run started 2026-04-25 19:29 UTC, completed 20,000 steps.

## Config (curriculum_qwen3vl_2b.yaml)

Two-phase curriculum via `curriculum_phases` — same total 20k steps, two
weight schedules.

| step range | benchcad | recode (cad-recode-v1.5) | text2cad (legacy) |
|-----------|---------|--------------------------|-------------------|
| 0  – 5k   | 5       | 1                        | 1                 |
| 5k – 20k  | 1       | 9                        | 0                 |

(curriculum mass shifts from BenchCAD-warmup → recode-heavy fine-tune.)

## Per-source items loaded

| source              | items   | notes                                                     |
|---------------------|---------|-----------------------------------------------------------|
| benchcad            | 18,167  | BenchCAD train (synthetic shapes from Hula0401/cad-sft)   |
| recode (cad-recode-v1.5) | ~107k | cadrille paper-style geometry, rendered via cadquery     |
| text2cad (legacy)   | 18,987  | Text-to-CAD train, used only in phase 1                   |

## Effective per-item exposure

`eff_bs = 32, total_steps = 20000 → 640k samples drawn`

Approximate (averaged across phases):

| source     | step share avg | total samples | per-item exposures |
|-----------|-----------------|----------------|---------------------|
| benchcad  | (5/7)·25% + (1/10)·75% = 25% | 160k     | 8.8 epochs |
| recode    | (1/7)·25% + (9/10)·75%       | 467k     | 4.4 epochs |
| text2cad  | (1/7)·25% + 0·75%            | 23k      | 1.2 epochs |

## Final eval IoU @ step 20000

| bucket          | IoU   | notes                                              |
|-----------------|-------|----------------------------------------------------|
| BenchCAD val    | 0.546 | Strong on bench shapes                             |
| DeepCAD test    | 0.466 | Decent — recode-heavy fine-tune helps DC geometry  |
| Fusion360 test  | 0.565 | **Best of any run on FU** — recode+text2cad mix    |

See `eval_metrics.csv` for full step-by-step.
