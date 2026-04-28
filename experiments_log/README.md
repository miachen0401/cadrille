# cadrille SFT — experiment log

This folder is the canonical record of every cadrille SFT run we've
finalized or are actively running. Each subfolder = one run, with
`config.yaml`, `eval_metrics.csv`, `data_mix.md`, and (post-hoc) `notes.md`.

The point of this log is **quantitative comparability**: every run uses
the same 5 eval buckets (BenchCAD val, recode20k train, text2cad train,
DeepCAD test, Fusion360 test), so per-step IoU + op metrics are
directly comparable across runs.

## Runs so far

| run                                       | model         | steps  | total items | mix style                                       | status                        |
|-------------------------------------------|---------------|--------|-------------|-------------------------------------------------|-------------------------------|
| `curriculum_qwen3vl_2b_20260425`          | Qwen3-VL-2B   | 20 k   | ~144 k      | Two-phase: benchcad-warmup → recode-heavy       | finished                      |
| `big_bench_shell_50k_20260427` (v1)       | Qwen3-VL-2B   | 50 k   | 437 k       | Single-phase, bench-heavy 5-way mix (4:4:3:2:1) | KILLED at step 4 k            |
| `big_bench_shell_50k_phase2_20260427` (v2)| Qwen3-VL-2B   | 50 k   | 484 k       | Single-phase, 50% high-quality (text2cad+recode), equal per-item within group | in progress |

## Eval-IoU snapshot

| run                                | BenchCAD val      | DeepCAD test      | Fusion360 test       |
|------------------------------------|--------------------|--------------------|------------------------|
| curriculum_qwen3vl_2b @ 20k (final)| **0.597 / 12k**    | 0.489 / 9k        | **0.565 / 20k**       |
| big_bench_shell_50k v1 @ 4k (final)| 0.356             | 0.384             | 0.405                  |
| big_bench_shell_50k_phase2 v2      | TBD               | TBD               | TBD                    |

(numbers from `*/eval_metrics.csv`. v2 in progress — eval rows append every 1k steps.)

## Key finding (resolved by v2)

v1 of `big_bench_shell_50k` showed a clear pattern: Fusion360 IoU regressed
vs curriculum by 6–17 IoU points at steps 1k–3k. Diagnosis: 78.6% of every
step's draws were bench-style geometry (benchcad + cad_iso_106 +
benchcad_simple = 11/14), and Fusion360's geometry distribution doesn't
match bench-shell. The curriculum's phase-2 (90% recode-v1.5) gave it
richer cadrille-style geometry exposure → better DC/FU generalization.

→ v2 was launched 2026-04-27 09:08 UTC with a fresh start and a 50% combined
text2cad_bench + recode_bench mix (equal per-item within each group). See
`big_bench_shell_50k_phase2_20260427/data_mix.md` for the per-source
weight derivation.

## How to extract metrics from a new run's log

```bash
uv run python experiments_log/extract_metrics.py \
    logs/<your_run>.log \
    experiments_log/<run_name>/eval_metrics.csv
```

## How to upload to HuggingFace

`experiments_log/` is structured to upload as a HF dataset (or as a
snapshot on a model card). Each `eval_metrics.csv` is a flat table
ready for pandas / sql / scientific plotting. The cross-run
comparison plot lives at `iou_curves.png` (regenerate via
`scripts/analysis/plot_experiments_log.py`).
