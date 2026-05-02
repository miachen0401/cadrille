# §7 v2 — actual training mix per config (after `total_train_dp: 500000` rebalance)

The 5 v2 configs all set `total_train_dp: 500000`, which the train.py
saturate-and-redistribute logic enforces by capping each source at its
available row count and shifting the deficit to non-saturated sources
(proportional to their nominal weight).

**Effect**: each config trains on ≈ 500k unique rows (off by ≤ 1 from
rounding), but the per-source row split drifts from the nominal 60/40
HQ:bench-stack ratio when bench-stack sources can't fill their share.

## Table

| Config | total | HQ rows | bench-stack rows | actual ratio | nominal | bench-stack detail |
|---|---:|---:|---:|---:|---:|---|
| `baseline_v2` | 500,000 | 500,000 | 0 | **100/0** | 100/0 | (no bench) |
| `iid_enhanced_v2` | 500,000 | 420,263 | 79,737 | **84/16** | 60/40 | easy 80k saturated → HQ +220k |
| `ood_v2` | 499,999 | 409,639 | 90,360 | **82/18** | 60/40 | bench 7.5k + iso 83k saturated → HQ +110k |
| `ood_enhanced_v2` | 500,000 | 366,074 | 133,926 | **73/27** | 60/40 | bench 11k + iso 122k saturated → HQ +66k |
| `iid_v2` | 499,999 | 305,197 | 194,802 | **61/39** | 60/40 | bench 11k + iso 122k + simple 61k → near-nominal |

## Per-source rows (deterministic across runs)

Per-source rng seed = `sha256(f'42:{src_name}')[:4]` (uint32) — stable
across processes, so the SAME source picks identical rows in any v2
config that loads it (smaller target = prefix of larger).

|  | benchcad | cad_iso_106 | benchcad_simple | benchcad_easy | text2cad_img | text2cad_text | recode_bench |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_v2` | — | — | — | — | 46,000 | 46,000 | 408,000 |
| `iid_enhanced_v2` | — | — | — | 79,737 | 38,524 | 38,524 | 343,215 |
| `ood_v2` | 7,511 | 82,849 | — | — | 37,550 | 37,550 | 334,539 |
| `ood_enhanced_v2` | 11,443 | 122,483 | — | — | 33,557 | 33,557 | 298,960 |
| `iid_v2` | 11,443 | 122,483 | 60,876 | — | 27,976 | 27,976 | 249,245 |

## Why ratio drifts

Saturate-redistribute picks "total volume controlled" over "ratio
controlled". The trade-off is documented because either choice is
defensible:

- **Volume-controlled** (current): every config trains on exactly the
  same number of unique rows → comparable compute / data-volume axis.
  Ratio drifts up to 24pp from nominal (ood_enhanced_v2 73/27 vs
  60/40).
- **Ratio-controlled** (alternative): every config has exactly 60/40
  HQ:bench-stack mix when bench is non-empty → comparable proportional
  exposure. Volume drifts up to 22% (ood_v2 390k vs 500k).

## Implication for §7 narrative

When ratio drift is large (e.g., ood_v2 82/18 vs nominal 60/40), the
"bench-stack effect" being measured is *diluted* — model sees ~half as
much bench content as the nominal mix would suggest. A reviewer might
ask: "is the lower OOD ess_pass on ood_v2 because bench-stack didn't
help, or because there wasn't enough of it?"

Mitigations to defend the comparison:

1. **iid_v2 is the strongest signal**: 61/39 actual is close to nominal,
   so it's the cleanest "saw bench-simple op patterns" reference.
2. **Per-source row identity**: ood_v2 ⊆ ood_enhanced_v2 ⊆ iid_v2 for
   recode_bench and text2cad_*; the deltas come from bench/iso/simple
   inclusion, not from "different HQ rows happened to be picked".
3. **Step-matched comparison**: at any fixed step, the model has seen
   the same total batches × tokens; ratio drift only affects
   per-source frequency.

If reviewers push back, switch to ratio-controlled mode by setting
`total_train_dp: null` in the v2 yamls (back to current behavior pre-fix).

## Verification

```bash
uv run python -c "
def saturate_redistribute(weights, available, budget):
    saturated = set(); targets = {s: 0 for s in weights}
    for _ in range(len(weights) + 2):
        non_sat = [s for s in weights if s not in saturated and weights[s] > 0]
        if not non_sat: break
        non_w = sum(weights[s] for s in non_sat)
        remaining = budget - sum(targets[s] for s in saturated)
        if remaining <= 0: break
        any_new = False
        for s in non_sat:
            proposed = int(round(remaining * weights[s] / non_w))
            if proposed >= available[s]:
                targets[s] = available[s]; saturated.add(s); any_new = True
            else:
                targets[s] = proposed
        if not any_new: break
    return targets
"
```

train.py prints the actual numbers at startup:
```
[total_train_dp] target=500,000  achieved=499,999  (2/9 sources saturated)
  benchcad                  7,511 kept  (saturated at available)
  recode_bench              398,701 → 334,539  (weight 490, src_seed=2766633513)
  cad_iso_106               82,849 kept  (saturated at available)
  text2cad_bench_img        52,500 → 37,550  (weight 55, src_seed=2569111977)
  text2cad_bench_text       52,500 → 37,550  (weight 55, src_seed=2358973235)
```
