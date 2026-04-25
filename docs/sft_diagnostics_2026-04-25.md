# SFT diagnostics — checkpoint-4000 (sft-s10k-bc2-r20k1-t2c1)

Snapshot 2026-04-25. Documents a mid-training analysis on the 2:1:1 mix run
(benchcad : recode20k : text2cad = 2 : 1 : 1, bs=15 acc=2 eff=30, lr=2e-4
cosine, 4000 of 10000 steps trained, killed early on user request after
online IoU plateau).

Inputs:
- ckpt: `checkpoints/sft-s10k-lr2e-4-b6a5-img-0424-2001/checkpoint-4000`
- diversity raw: `eval_outputs/diversity_t3_ckpt4k_*/raw.jsonl` (30 items
  from data/benchcad/val, seed 42, K=8 samples at temps 0/0.5/1.0)
- bench sweep partial (n=50/split, 6 temps, 16 samples) for IoU numbers

## Headline metric (online during training)

| step | BenchCAD val IoU | DeepCAD test IoU | Fusion360 test IoU |
|---:|---:|---:|---:|
| 0 | 0.000 (Fail 100%) | 0.000 | 0.000 |
| 1000 | 0.132 | 0.144 | 0.195 |
| 2000 | 0.114 | 0.186 | 0.351 |
| 3000 | 0.107 | 0.271 | 0.347 |
| **4000** | **0.126** | **0.256** | **0.404** |
| 5000 | 0.120 | 0.270 | 0.388 |

eval_loss(benchcad val): 1.253 → 0.346 (step 1k) → 0.252 (step 2k) → 0.270
(step 5k). **Eval-loss min at step 2k, then climbing — overfit signal.**
Fusion360 IoU peaks at step 4000.

## T4 sweep (50 items × 6 temps × 16 samples) — partial

### benchcad/val
| t | exec | iou_first | max_iou@16 |
|---:|---:|---:|---:|
| 0.00 | 90% | 0.050 | 0.050 |
| 0.50 | 95% | 0.079 | 0.154 |
| 1.00 | 91% | 0.058 | **0.169** |
| 1.25 | 80% | 0.068 | 0.152 |

### deepcad/test
| t | exec | iou_first | max_iou@16 |
|---:|---:|---:|---:|
| 0.00 | 94% | 0.074 | 0.074 |
| 0.50 | 95% | 0.093 | 0.210 |
| 0.75 | 95% | 0.083 | **0.226** |
| 1.00 | 93% | 0.078 | 0.225 |

### fusion360/test
| t | exec | iou_first | max_iou@16 |
|---:|---:|---:|---:|
| 0.00 | 90% | 0.073 | 0.073 |
| 0.50 | 96% | 0.127 | 0.232 |
| 0.75 | 95% | 0.101 | **0.272** |

**Pattern across all three test sets:**
- exec_rate is healthy at 90-95% for t ≤ 1.0; collapses at t=1.25
- max_iou@16 ≈ 2-3× iou_first → sampling diversity helps recover IoU
- best temperature around 0.75 across datasets

## Op-presence diagnostic (the punchline)

Frequency of each CadQuery op in items containing it (regex, no exec).

### Datasets (full GT)

| op | benchcad (20,143) | recode20k (20,000) | text2cad (3,000 / 171k) | pred t=1.0 (30) |
|---|---:|---:|---:|---:|
| `extrude` | 41 | 91 | 100 | **100** |
| `sketch` | 0 | 91 | 100 | **100** |
| `segment` | 0 | 76 | 77 | **83** |
| `union` | 37 | 71 | 30 | **60** |
| `workplane` | **58** | 47 | 0 | **57** |
| `transformed` | **44** | 0 | 0 | **0** ❌ |
| `circle` | 32 | 38 | 43 | **87** |
| `box` | **43** | 23 | 0 | **37** |
| `hole` | **42** | 0 | 0 | **0** ❌ |
| `moveTo` | 25 | 39 | 0 | **37** |
| `chamfer` | **30** | 0 | 0 | **0** ❌ |
| `rect` | 23 | 29 | 0 | **57** |
| `cut` | **28** | 0 | 14 | **0** ❌ |
| `revolve` | **17** | 0 | 0 | **0** ❌ |
| `polyline` | **16** | 0 | 0 | **0** ❌ |
| `arc` | **11** | 0 | 0 | **0** ❌ |
| `fillet` | **8** | 0 | 0 | **0** ❌ |

## Learnings

### 1. Three corpora, two distinct CadQuery dialects
- **benchcad** uses the op-level API (`.box/.cylinder/.hole/.cut/.chamfer/
  .revolve/.transformed`) — maps closely to BREP CAD operations.
- **recode20k** + **text2cad** are sketch-based (`.sketch().segment().extrude()`
  composed with `.workplane/.union/.moveTo`) — closer to declarative 2D-then-
  extrude.
- The 2:1:1 mix nominally weights benchcad heaviest, but token-level
  cross-entropy is dominated by the sketch dialect because:
  - recode20k + text2cad have very repetitive short templates → low
    perplexity → small gradient → easy convergence
  - benchcad's diverse op-level code has higher token-level entropy →
    harder to fit → effectively under-weighted in optimisation

### 2. The model collapsed to the easy dialect
- `extrude/sketch/segment/circle/rect/union` are over-produced (pred 60–100%
  vs GT 30–80%).
- `hole/cut/chamfer/revolve/transformed/polyline/arc/fillet` are zero in
  pred at all temperatures (greedy and sampled) — even though benchcad has
  them at 8–44% rates.
- This collapse is **not** caused by data scarcity — `transformed` and
  `hole` are present in 44% / 42% of benchcad items; the model has seen
  them tens of thousands of times across 4000 × 30 = 120,000 effective
  exposures.

### 3. Sampling helps but doesn't fix the gap
- max_iou@16 is 2–3× iou_first across all three eval datasets, so the
  model *can* sometimes produce a better solution — diversity is not zero.
- BUT, max_iou@16 still caps at ~0.17 (BenchCAD) / 0.22 (DeepCAD) / 0.27
  (Fusion360). The IoU ceiling reflects the missing op set: any sample
  that *needs* a hole or a chamfer is geometrically wrong even when
  syntactically valid.

### 4. Online IoU eval is the right signal
- `eval_loss` minimum at step 2k diverged from `online IoU` peak at step
  4k for Fusion360. Loss-only would have early-stopped too aggressively.
- 30 items per dataset × 3 datasets at every eval_steps tick is cheap
  enough (~50 s per pass) to keep on continuously.

## Next-run hypotheses (ranked)

1. **benchcad-heavy mix** — push the dialect we want.
   `sft_mix_weights: benchcad=4, recode20k=1, text2cad=1` (benchcad 67%).
   Continue from `checkpoint-4000` with a fresh cosine cycle. **Do this
   first** — minimal risk and tests the hypothesis directly.

2. **Longer training** with the current mix (10k → 16k steps). If the
   problem is *just* under-exposure of rare ops, more steps + a fresh lr
   peak (cosine restart) might suffice. Cheaper to combine with #1.

3. **Op-level loss reweighting** — give cross-entropy on tokens of rare
   ops (hole/chamfer/cut/revolve) a higher weight. Big change; only worth
   it if #1+#2 don't move the needle.

4. **Sketch-style normalisation** — rewrite recode20k and text2cad GT into
   the benchcad op-level dialect before training. Eliminates the dialect
   conflict at source. Most upside, most engineering. Defer.

## Artefacts

- `eval_outputs/diversity_t3_ckpt4k_<ts>/` — full diversity run
  - `summary.md` — 30-item GT slice vs pred ops
  - `benchcad_op_freq.md` — full benchcad GT (20k items) vs pred
  - `dataset_op_dist.md` — three GT corpora + pred side-by-side
  - `raw.jsonl` — every generated code preserved for re-analysis
- `eval_outputs/sweep_t3_ckpt4k_*/` — partial T4 sweep (killed at fusion360
  t=1.0)
- HF backup: `Hula0401/cadrille-sft-bc2-r20k1-t2c1/checkpoint-4000` (private)
