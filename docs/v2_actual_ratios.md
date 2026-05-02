# §7 v2 — pool size vs sampling proportion (two independent axes)

The 5 v2 configs control TWO orthogonal aspects of training data exposure.
This doc clarifies which is which, where each lives in code, and what
each ablation actually tests.

## Axis 1 — POOL size (which rows are loaded)

How many UNIQUE rows make it into the training dataset, broken down per
source. Set by `total_train_dp` in the yaml.

- yaml field: `total_train_dp: 500000`
- code: `train/sft/train.py` lines ~614–704 — saturate-and-redistribute
  block, runs AFTER `Dataset` constructors but BEFORE `ConcatDataset`
  assembly. Edits `ds.annotations` in-place to truncate to the per-source
  target.
- per-source rng seed = `sha256(f'{base_seed}:{src_name}')[:4]` (uint32) —
  stable across processes and configs, so smaller targets are PREFIXES of
  larger targets for the same source.
- effect: equalises total unique-row count across configs. Drifts
  per-source ratios when source supply < demand (saturation).

## Axis 2 — SAMPLING proportion (per-step batch composition)

What FRACTION of each batch comes from each source. Set by
`sft_mix_weights` in the yaml.

- yaml field: `sft_mix_weights: {benchcad: 34, cad_iso_106: 366, ...}`
- code: `train/sft/train.py` lines ~159 (`LengthGroupedWeightedSampler`),
  ~278 (`_expand_mix_to_sample_weights`), ~316 (`WeightedSamplerTrainer`).
  Builds a `WeightedRandomSampler` whose per-sample weight = `mix_w / n`
  so total source mass = `mix_w` regardless of `n`.
- effect: per-step batch composition is fixed at the source weights.
  Pool size shrinkage does NOT change source sampling probability — it
  only changes how many UNIQUE rows the model can see during training.

### Verified independence

```python
mix = {'recode': 490, 'iso': 366, 'bench': 34, 't2c_img': 55, 't2c_text': 55}

# Full pool (no total_train_dp)
sizes_full  = {'recode': 472244, 'iso': 82849, 'bench': 7511, 't2c_img': 53315, 't2c_text': 53315}

# total_train_dp=500k rebalanced pool
sizes_rebal = {'recode': 334539, 'iso': 82849, 'bench': 7511, 't2c_img': 37550, 't2c_text': 37550}

# source-level sampling probability (after _expand_mix_to_sample_weights)
full pool   sampling probs:  recode=0.490 iso=0.366 bench=0.034 t2c_img=0.055 t2c_text=0.055
rebal pool  sampling probs:  recode=0.490 iso=0.366 bench=0.034 t2c_img=0.055 t2c_text=0.055
```

Both pools yield IDENTICAL source-level sampling probabilities.

## Per-config summary (POOL size)

After `total_train_dp: 500000` rebalance:

| Config | total | benchcad | iso | simple | easy | t2c_img | t2c_text | recode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_v2`     | 500,000 | — | — | — | — | 46,000 | 46,000 | 408,000 |
| `iid_enhanced_v2` | 500,000 | — | — | — | 79,737 | 38,524 | 38,524 | 343,215 |
| `ood_v2`          | 499,999 | 7,511 | 82,849 | — | — | 37,550 | 37,550 | 334,539 |
| `ood_enhanced_v2` | 500,000 | 11,443 | 122,483 | — | — | 33,557 | 33,557 | 298,960 |
| `iid_v2`          | 499,999 | 11,443 | 122,483 | 60,876 | — | 27,976 | 27,976 | 249,245 |

Saturated sources (capped at availability) → deficit redistributed to
non-saturated sources proportional to weight. Per-source rows are
DETERMINISTIC and NESTED across configs — e.g. ood_v2's 334,539 recode
rows are a strict prefix of baseline_v2's 408,000 recode rows.

## Per-config summary (SAMPLING proportions)

The `sft_mix_weights` from each yaml — **unchanged by `total_train_dp`**:

| Config | benchcad | iso | simple | easy | t2c_img | t2c_text | recode | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_v2`     | — | — | — | — | 92 | 92 | 816 | 1000 |
| `iid_enhanced_v2` | — | — | — | 400 | 55 | 55 | 490 | 1000 |
| `ood_v2`          | 34 | 366 | — | — | 55 | 55 | 490 | 1000 |
| `ood_enhanced_v2` | 33 | 367 | — | — | 55 | 55 | 490 | 1000 |
| `iid_v2`          | 23 | 250 | 125 | — | 55 | 55 | 490 | 998 |

These are the per-step source weights (sum to 1000 = 100%). Each batch
draws ~weight/1000 fraction of its rows from that source. So at every
training step, ood_v2 is 60% HQ + 40% bench-stack at the BATCH level —
even though its POOL is 82% HQ / 18% bench-stack rows.

## What the ablation actually controls

Both axes matter; v2 separates them:

| What | Controlled by | Same across all 5? |
|---|---|---|
| Total unique rows (pool size) | `total_train_dp: 500000` | YES (≈ 500k each) |
| Total training compute | `max_steps: 50000` × `batch × accum × G` | YES |
| Per-step batch composition (source mix) | `sft_mix_weights` | NO (different per config) |
| Which families are train-vs-test (holdout) | `holdout_families` + `holdout_families_v2` | NO (varies per config) |

The §7 v2 figure isolates: **at fixed compute and fixed pool size, what
does the per-step source mix and family-holdout selection do to OOD
ess_pass / IoU?**

## Effective epochs per source (combined view)

Approximate rows seen during 50k training steps (batch 8, accum 4 →
effective batch 32):

```
total samples viewed = 50,000 × 32 = 1,600,000
samples per source   = 1,600,000 × (mix_weight / total_weight)
unique rows per src  = (from POOL table above)
effective epochs     = samples_viewed / unique_rows
```

For ood_v2 specifically:
| Source | unique rows | sampling prob | samples viewed | epochs |
|---|---:|---:|---:|---:|
| benchcad | 7,511 | 3.4% | 54,400 | **7.2** |
| cad_iso_106 | 82,849 | 36.6% | 585,600 | **7.1** |
| text2cad_img | 37,550 | 5.5% | 88,000 | **2.3** |
| text2cad_text | 37,550 | 5.5% | 88,000 | **2.3** |
| recode_bench | 334,539 | 49.0% | 784,000 | **2.3** |

Note: bench-stack rows (high sampling prob, small pool) get repeated 7×
while HQ rows get 2.3×. This is the design — sampling weight × small pool
→ effective oversampling of mechanical content.

For baseline_v2 (HQ only):
| Source | unique rows | sampling prob | samples viewed | epochs |
|---|---:|---:|---:|---:|
| text2cad_img | 46,000 | 9.2% | 147,200 | **3.2** |
| text2cad_text | 46,000 | 9.2% | 147,200 | **3.2** |
| recode_bench | 408,000 | 81.6% | 1,305,600 | **3.2** |

All ~3.2 epochs — pool/sampling matched within HQ.

## Code references

| What | File:line | Notes |
|---|---|---|
| `total_train_dp` arg parsing | `train/sft/train.py:1049` | `cfg.get('total_train_dp', None)` |
| Pool subsample (saturate-redistribute) | `train/sft/train.py:614–704` | runs once per training launch |
| `sft_mix_weights` arg parsing | `train/sft/train.py:1010` | `cfg.get('sft_mix_weights', None)` |
| Per-sample weight expansion | `train/sft/train.py:278` | `_expand_mix_to_sample_weights` |
| Sampler class | `train/sft/train.py:159` | `LengthGroupedWeightedSampler` |
| Trainer that consumes sampler | `train/sft/train.py:306` | `WeightedSamplerTrainer` |

## TL;DR

- **POOL size** = how many distinct rows the model can ever see (set by
  `total_train_dp` + per-source availability). Equalised at 500k across
  v2 configs.
- **SAMPLING proportion** = which source each batch row is drawn from
  (set by `sft_mix_weights`). Stays at the yaml value regardless of
  pool size.

The two interact as `effective_epochs = (sampling_prob × total_samples) /
unique_rows`. Different configs have different effective epochs per
source — this is intentional (the high-weight + small-pool design
oversamples bench-stack rows for the OOD-relevant signal).
