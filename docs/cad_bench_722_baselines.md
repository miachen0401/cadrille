# `BenchCAD/cad_bench_722` baselines — official Cadrille, official CADEvolve, zero-shot Qwen2.5-VL

**Dataset:** [`BenchCAD/cad_bench_722`](https://huggingface.co/datasets/BenchCAD/cad_bench_722) — 720 rows, single `train` split, the *diversified / substituted-parts* track of the BenchCAD benchmark (`is_substituted=True` for half the rows; OOD by construction). Each row carries `composite_png` (single 268×268 multi-view collage), `gt_code` (CadQuery), `family / difficulty / base_plane / feature_tags / ops_used`.

**Run date:** 2026-04-28
**Hardware:** RTX 4080 SUPER (16 GB)
**Branch:** `eval/cad-bench-722` · commits `c56747c` (eval adapters), `6374203` (`compute_iou_24`), `b0e2be1` (rescore driver)

---

## Headline (greedy, single attempt per sample)

| model                                 | input format                  | exec    | mean IoU | mean CD   |
|---                                    |---                            |---:     |---:      |---:       |
| **CADEvolve-rl1** (official)          | 8-view 476×952 axis-coloured  | **86.5%** | **0.367** | **0.034** |
| **Cadrille-rl** (official, filapro)   | point cloud + composite_png   | 66.9%   | 0.054    | 0.412     |
| **Qwen2.5-VL-3B-Instruct** (zero-shot)| composite_png 268×268 only    | 2.2%    | 0.146*   | 0.365     |

*Qwen IoU is over the 16 successful samples only — the model has never seen CadQuery and crashes on 98% of inputs.

### Per-difficulty (exec / mean IoU)

```
                    easy           medium         hard
cadevolve_rl1   83% / 0.392    92% / 0.364    85% / 0.348
cadrille_rl     67% / 0.054    71% / 0.048    63% / 0.060
qwen2.5-vl-3b    3% / 0.101     3% / 0.221     2% / 0.102
```

CADEvolve generalises smoothly across difficulty; Cadrille's IoU is uniformly low across `easy / medium / hard`, which is itself a clue (see §3).

---

## Rotation-invariant rescore (IoU-24)

For each prediction, we tried all 24 axis-aligned rotations of `pred_mesh` and kept the maximum volumetric IoU. Implementation: `common.metrics.compute_iou_24` plus the iou-24 mode of the existing CadQuery subprocess worker, with early-stop at IoU ≥ 0.95. Re-scored on the same 720 samples per model:

| model           | n paired | mean iou | mean iou_24 | Δ (mean) | rotation-win rate |
|---              |---:      |---:      |---:         |---:      |---:               |
| cadevolve_rl1   | 688      | 0.273    | **0.339**   | +0.066   | 344/688 = 50.0%   |
| cadrille_rl     | 661      | 0.039    | **0.068**   | +0.029   | **556/661 = 84.1%** |
| qwen25vl_3b_zs  | 17       | 0.137    | 0.210       | +0.073   | 15/17 = 88.2%     |

(`mean iou` here averages over `success ∪ zero_iou` records — the same denominator as `mean iou_24` — which is why it differs from the headline-table `mean IoU` that excludes zero-IoU records.)

### What this tells us
- **Cadrille-rl: orientation drift dominates.** 84% of Cadrille's well-formed predictions are correct shape with the wrong orientation. The naive IoU 0.039 → 0.068 (+73% relative) is large compared to CADEvolve's much smaller relative gain (+24% relative). On this diversified test track, Cadrille is mostly losing geometry to choice of `base_plane` / axis convention, not to incorrect part topology.
- **CADEvolve-rl1: shape and orientation usually align.** 50% rotation-win rate means half the time identity already wins, the other half a 90° rotation snaps a slightly-mis-oriented model into place for a small score boost. The official 8-view input with axis-encoded green channel is doing the heavy lifting here.
- **Qwen2.5-VL-3B zero-shot: not a useful baseline number from 17 successes.** Useful as a "VLM has no idea what CadQuery is" floor.

---

## Methodology

1. **Single-attempt eval.** Greedy decoding, `do_sample=False`, `max_new_tokens=768` (1024 for Qwen). One prediction per sample.
2. **Cadrille input** = `composite_png` (the dataset's GT render), point-cloud branch fed via the existing `Cadrille.collate` path with `n_points=256`.
3. **CADEvolve input** = `gt_code` exec → STL → 8-view 476×952 collage rendered by `experiments/cadevolve/render.py` (matplotlib-coloured by the per-view depth axis). Falls back to `composite_png` only if the 8-view render fails.
4. **Qwen2.5-VL-3B input** = `composite_png` 268×268 + a strict CadQuery-only prompt (no example geometry to copy). Output stripped of markdown fences before scoring.
5. **Scoring** for both naive IoU and Chamfer Distance: bounding box → `[-1, 1]^3` normalisation, trimesh boolean intersection for IoU (`common.metrics.compute_iou`), 8192-sample bidirectional L2 chamfer (`compute_cd`).
6. **IoU-24:** same scoring pipeline, but pred_mesh is rotated through all 24 cube symmetries (signed axis-permutations with det=+1) and the maximum is kept. Early-stop at IoU ≥ 0.95.

---

## Artifacts

```
eval_outputs/cad_bench_722/
  cadevolve_rl1/         metadata.jsonl  metadata_24.jsonl  720 × <stem>.py
  cadrille_rl/           metadata.jsonl  metadata_24.jsonl  720 × <stem>.py
  qwen25vl_3b_zs/        metadata.jsonl  metadata_24.jsonl  720 × <stem>.py
  summary.json
  summary_iou_24.json
```

`metadata.jsonl` schema (one line per sample): `stem, family, difficulty, base_plane, split, feature_tags, feature_count, code_len, error_type, iou, cd`. `metadata_24.jsonl` adds `iou_24`, `rot_idx` (0 = identity, 1–23 = the 24 rotations), and `iou_recheck`.

## How to reproduce

```bash
# Three baselines + Discord summary
set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
nohup bash scripts/eval_cad_bench_722.sh > logs/eval_cad_bench_722.log 2>&1 &

# Rotation-invariant rescore on the resulting metadata
nohup bash scripts/run_rescore_iou_24.sh > logs/rescore_iou_24.log 2>&1 &
```

## Caveats

- The exec-failure mass (Cadrille 33%, Qwen 98%) is excluded from mean-IoU figures — these are conditional-on-exec scores. A complete picture would multiply by exec rate.
- `cad_bench_722` is the diversified split — heavy substitution and synthesised stems put it well outside Cadrille's training distribution. Headline numbers are not directly comparable to numbers reported on the original BenchCAD benchmark.
- Qwen2.5-VL-3B is purely zero-shot. A single-shot or few-shot prompt with one or two CadQuery examples would likely push exec rate well above 2%, but that is not the comparison we ran.
