# `cad_bench_722` — multi-baseline evaluation

**Dataset:** [`BenchCAD/cad_bench_722`](https://huggingface.co/datasets/BenchCAD/cad_bench_722) — 720 rows, single `train` split, the *diversified / substituted-parts* track of the BenchCAD benchmark.

**Hardware:** RTX 4080 SUPER (16 GB).

**Branch:** `eval/cad-bench-722`

---

## 1. Headline (greedy, single attempt)

| model                            | input                        |   exec | mean IoU |   mean CD |
|----------------------------------|------------------------------|--------|----------|-----------|
| Cadrille-rl (filapro)            | pc + composite_png           |  66.9% |   0.0538 |  0.411574 |
| CADEvolve-rl1 (kulibinai)        | 8-view 476×952 axis-coloured |  86.5% |   0.3672 |  0.034397 |
| Qwen2.5-VL-3B (zero-shot)        | composite_png 268×268        |   2.2% |   0.1460 |  0.365229 |
| Cadrille-Q3VL-v3 (50k clean)     | composite_png 268×268 (Qwen3-VL) |  92.1% |   0.6529 |  0.025873 |

### Per-difficulty (exec / mean IoU)

| model                            | easy           | medium         | hard           |
|----------------------------------|----------------|----------------|----------------|
| Cadrille-rl (filapro)            | 66.5% / 0.054  | 71.0% / 0.048  | 63.5% / 0.060  |
| CADEvolve-rl1 (kulibinai)        | 83.0% / 0.391  | 91.6% / 0.364  | 84.9% / 0.348  |
| Qwen2.5-VL-3B (zero-shot)        | 2.6% / 0.101   | 2.5% / 0.220   | 1.6% / 0.102   |
| Cadrille-Q3VL-v3 (50k clean)     | 93.0% / 0.705  | 94.1% / 0.663  | 89.3% / 0.593  |

## 2. IoU-24 rotation rescue

`Δ = mean(iou_24 − iou)` over paired cases. `pct_rot_win` = fraction of cases where a non-identity rotation beat the identity, i.e. correct shape but oriented wrong.

| model                            | n_paired | mean iou | mean iou_24 |       Δ | pct_rot_win |
|----------------------------------|----------|----------|-------------|---------|-------------|
| Cadrille-rl (filapro)            |      661 |   0.0393 |      0.0680 | +0.0288 |       84.1% |
| CADEvolve-rl1 (kulibinai)        |      688 |   0.3325 |      0.3387 | +0.0062 |       50.0% |
| Qwen2.5-VL-3B (zero-shot)        |       17 |   0.1374 |      0.2101 | +0.0727 |       88.2% |
| Cadrille-Q3VL-v3 (50k clean)     |      686 |   0.6310 |      0.6546 | +0.0236 |       40.4% |

## 3. Distribution-level metrics

Computed against the full 720 GT image distribution. FID / KID lower = better; CLIP R-Precision higher = better.

| model                            | n_pred |      FID |       KID |    R@1 |    R@5 |   R@10 |
|----------------------------------|--------|----------|-----------|--------|--------|--------|
| Cadrille-rl (filapro)            |    482 |   191.87 |   0.07610 |  0.000 |  0.017 |  0.042 |
| CADEvolve-rl1 (kulibinai)        |    600 |   165.80 |   0.08654 |  0.023 |  0.125 |  0.197 |
| Qwen2.5-VL-3B (zero-shot)        |     16 |   499.10 |   0.52383 |  0.000 |  0.000 |  0.062 |
| Cadrille-Q3VL-v3 (50k clean)     |    600 |   118.26 |   0.03679 |  0.082 |  0.255 |  0.382 |

## 4a. Out-of-distribution: Deepcad (300 samples, seed=42)

| model                            |    n |   exec | mean IoU |   mean CD |
|----------------------------------|------|--------|----------|-----------|
| Cadrille-rl (filapro)            |  300 |  59.7% |   0.1396 |  0.317448 |
| CADEvolve-rl1 (kulibinai)        |  300 |  74.7% |   0.1433 |  0.482483 |
| Qwen2.5-VL-3B (zero-shot)        |  300 |  21.7% |   0.1706 |  0.212922 |
| Cadrille-Q3VL-v3 (50k clean)     |  300 |  95.7% |   0.7802 |  0.010317 |

## 4b. Out-of-distribution: Fusion360 (300 samples, seed=42)

| model                            |    n |   exec | mean IoU |   mean CD |
|----------------------------------|------|--------|----------|-----------|
| Cadrille-rl (filapro)            |  300 |  60.0% |   0.1472 |  0.283255 |
| CADEvolve-rl1 (kulibinai)        |  300 |  70.7% |   0.0946 |  0.585575 |
| Qwen2.5-VL-3B (zero-shot)        |  300 |  25.3% |   0.1388 |  0.301565 |
| Cadrille-Q3VL-v3 (50k clean)     |  300 |  90.7% |   0.6843 |  0.017836 |

---

## Artifacts

```
eval_outputs/cad_bench_722/
  cadrille_rl/      metadata.jsonl  metadata_24.jsonl  720 × <stem>.py
  cadevolve_rl1/    metadata.jsonl  metadata_24.jsonl  720 × <stem>.py
  qwen25vl_3b_zs/   metadata.jsonl  metadata_24.jsonl  720 × <stem>.py
  cadrille_qwen3vl_v3/  metadata.jsonl  metadata_24.jsonl  720 × <stem>.py
  summary.json                  — model-level IoU/CD
  summary_iou_24.json           — IoU-24 rescue summary
  distribution_metrics.json     — FID / KID / CLIP R-Precision
  metrics_per_case_full.json    — per-case Fs / DINO / LPIPS / SSIM / PSNR
  iou_vs_iou24/{report.md, scatter.png, histogram.png, rotation_dist.png}
  full_case_grids/cases_NNNN-NNNN.png × 15  — visual grid, 4 model columns
  RESULTS.md                    — this file
eval_outputs/deepcad_n300/<model>/metadata.jsonl   — OOD sample
eval_outputs/fusion360_n300/<model>/metadata.jsonl — OOD sample
```

## How to reproduce

```bash
set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a

# 1. cad_bench_722 (greedy, all 4 models — already wired in eval/bench.py)
bash scripts/eval_cad_bench_722.sh

# 2. IoU-24 rotation rescore on the resulting metadata
bash scripts/run_rescore_iou_24.sh

# 3. Extended per-case metrics (F-score / DINO / LPIPS / SSIM)
uv run python research/3d_similarity/compute_full_metrics.py

# 4. Distribution-level (FID / KID / CLIP R-Precision)
uv run python research/3d_similarity/score_distribution.py

# 5. IoU vs IoU-24 analysis (figures + report.md)
uv run python research/3d_similarity/analyze_iou_vs_iou24.py

# 6. Full 720-case visual grid (15 PNG pages)
uv run python research/3d_similarity/build_full_grid.py

# 7. OOD: DeepCAD + Fusion360 sampled n=300 each
bash scripts/run_stl_eval_all.sh

# 8. Rebuild this markdown
uv run python scripts/analysis/build_summary_md.py
```
