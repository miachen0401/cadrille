# §7 v2 — training mix per config (Plan A: natural pool, no cap)

After several iterations, settled on the simplest design:

- **No `total_train_dp` cap** — each config uses its full natural pool.
- **`sft_mix_weights` controls per-batch sampling** (uniform within source).
- **bench+iso ≥ 40%** in every config that includes them (the user-set floor).
- **Same compute** across configs: 50k step × batch 8 × accum 4 = 1.6M
  samples per run.

## Mix per config (per-batch sampling proportions)

`sft_mix_weights` from each yaml — sums to 1000 (= 100% per batch):

| Config | bench | iso | simple | easy | t2c_img | t2c_text | recode | bench+iso |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_v2`     | — | — | — | — | 92 | 92 | 816 | **0%** |
| `iid_enhanced_v2` | — | — | — | 400 | 55 | 55 | 490 | **0%** |
| `ood_v2`          | 33 | 367 | — | — | 55 | 55 | 490 | **40%** |
| `ood_enhanced_v2` | 33 | 367 | — | — | 55 | 55 | 490 | **40%** |
| `iid_v2`          | 33 | 367 | 125 | — | 43 | 43 | 389 | **40%** |

Configs without bench+iso (baseline_v2 / iid_enhanced_v2) are HQ-only or
HQ + benchcad-easy — those are testing data axes that don't include
mechanical content, so the bench+iso floor doesn't apply.

## Natural pool size per config (unique training rows)

Each config loads sources at full size (no subsample):

| Config | benchcad | iso | simple | easy | text2cad×2 | recode | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_v2`     | — | — | — | — | 53,315×2 | 472,244 | 578,874 |
| `iid_enhanced_v2` | — | — | — | 79,737 | 53,315×2 | 472,244 | 658,611 |
| `ood_v2`          | 7,511 | 82,849 | — | — | 53,315×2 | 472,244 | 669,234 |
| `ood_enhanced_v2` | 11,443 | 122,483 | — | — | 53,315×2 | 472,244 | 712,800 |
| `iid_v2`          | 11,443 | 122,483 | 60,876 | — | 53,315×2 | 472,244 | 773,676 |

Different totals are intentional — controlled axis is **per-batch source
mix**, not unique-row count. Comparing at matched compute (1.6M samples
seen by every config) is the standard ablation framing.

## Effective epochs per source (matched compute = 1.6M samples)

`epochs = sampling_prob × 1.6M / unique_rows`. Small pools (bench, iso,
simple, easy) get oversampled because their high mix weight × few unique
rows.

### ood_v2 example

| Source | unique | mix prob | samples seen | epochs |
|---|---:|---:|---:|---:|
| benchcad | 7,511 | 3.3% | 52,800 | **7.0** |
| cad_iso_106 | 82,849 | 36.7% | 587,200 | **7.1** |
| text2cad_img | 53,315 | 5.5% | 88,000 | **1.7** |
| text2cad_text | 53,315 | 5.5% | 88,000 | **1.7** |
| recode_bench | 472,244 | 49.0% | 784,000 | **1.7** |

bench-stack rows oversampled ~4× vs HQ — by design (small mechanical
pool gets seen many times so the rare-op signal lands).

### baseline_v2 example

| Source | unique | mix prob | samples seen | epochs |
|---|---:|---:|---:|---:|
| text2cad_img | 53,315 | 9.2% | 147,200 | **2.8** |
| text2cad_text | 53,315 | 9.2% | 147,200 | **2.8** |
| recode_bench | 472,244 | 81.6% | 1,305,600 | **2.8** |

All ~2.8 epochs — pure HQ, no oversample asymmetry.

## What v2 controls vs lets vary

| Axis | Controlled? | Mechanism |
|---|---|---|
| Total compute | YES (same) | `max_steps × batch × accum × G` identical |
| Per-batch source mix | YES (per-config) | `sft_mix_weights` |
| bench+iso ≥ 40% | YES (≥ floor) | hand-set in yamls |
| Family holdout | YES (per-config) | `holdout_families` (v1 mech) + `holdout_families_v2` (bench-simple) |
| Total unique rows | NO (varies 579k–774k) | natural pool |
| Effective epochs per source | NO (varies) | derived from mix × pool |

## Why Plan A vs the earlier total_train_dp + saturate-redistribute

The original concern was "data volume confound" — different mixes giving
different total unique rows might confound the ablation. We tried two
fixes:

- **B (cap-only)**: `target_per_source = min(weight × budget / total_w,
  available)`. Some configs underflow → totals drift up to ±20%. Ratio
  preserved.
- **C (saturate-redistribute)**: cap, then redistribute deficit. Total
  hit exactly, but ratio drifts up to 24pp from nominal.

Both added complexity for a confound that's already controlled by
**matched compute** (every config sees 1.6M samples regardless of pool).
A is the cleanest framing — defends against the volume-confound critique
by pointing at compute parity.

`total_train_dp` and `sft_pool_rows` support remain in train.py for
future fine control, but the v2 yamls leave them unset (back to natural
pool).

## Code references

| What | File:line |
|---|---|
| sampler weight expansion | `train/sft/train.py:278` `_expand_mix_to_sample_weights` |
| sampler class | `train/sft/train.py:159` `LengthGroupedWeightedSampler` |
| trainer with sampler | `train/sft/train.py:306` `WeightedSamplerTrainer` |
| (off by default) pool subsample | `train/sft/train.py:614+` |
