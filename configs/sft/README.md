# SFT Configs — Paper §7 v4-holdout study

Six configs total. Each row: which run, where the checkpoint lives.

| Config | Run name | Ckpt path | HF repo | Status |
|---|---|---|---|---|
| `big_bench_shell_50k_v3.yaml` | sft-s50k-lr2e-4-b8a4-img-0428-1320 | `/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/` | (local only) | ✅ 50k done (baseline) |
| `big_bench_shell_50k_v4_holdout.yaml` | sft-s50k-lr2e-4-b8a4-img-0430-0828 | `/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/` | `Hula0401/cadrille-qwen3vl-2b-v4-holdout-50k` | 🟡 23k/50k in progress |
| `big_bench_shell_50k_v4_baseline.yaml` | (TBD) | (TBD) | `Hula0401/cadrille-qwen3vl-2b-v4-baseline-50k` | ❌ pending — control: same recipe, no holdout |
| `big_bench_shell_50k_v4_holdout_noeasy.yaml` | (TBD) | (TBD) | `Hula0401/cadrille-qwen3vl-2b-v4-holdout-noeasy-50k` | ❌ pending — same holdout, no benchcad-easy |
| `big_bench_shell_50k_v4_hq_only.yaml` | (TBD) | (TBD) | `Hula0401/cadrille-qwen3vl-2b-v4-hq-only-50k` | ❌ pending — text2cad + recode_bench only |
| `holdout_families.yaml` | (n/a) | (n/a) | (n/a) | ✅ canonical holdout list — single source for all offline scripts via `common/holdout.py` |

## Recipe diffs

| Config | Mix | Holdout | benchcad-easy | bench-stack |
|---|---|---|---|---|
| v3 | 36% HQ / 64% bench | none (all 106 fams) | ✗ | ✓ |
| v4-holdout | 60% HQ / 40% bench | 10 fams | ✓ (80k) | ✓ |
| v4-baseline | 60% HQ / 40% bench | none | ✓ | ✓ |
| v4-holdout-noeasy | 60% HQ / 40% bench | 10 fams | ✗ | ✓ |
| v4-hq-only | 100% HQ | 10 fams (filter still applied) | ✗ | ✗ |

## Naming convention

`big_bench_shell_50k_*`: 50k-step SFT on BenchCAD-shell-style data (cadquery code rewritten to BenchCAD's conventional shell format via AST rewrite). The "shell" suffix distinguishes from raw cad-recode style.

## Holdout family list

Single source: `holdout_families.yaml`. Currently 10 families:
`tapered_boss, taper_pin, venturi_tube, bucket, dome_cap, nozzle, enclosure, waffle_plate, bolt, duct_elbow`

Loaded by:
- offline scripts via `common/holdout.py`
- training runtime via `train/sft/online_eval.py::set_holdout_families()` (called from train.py with cfg.holdout_families)

Each SFT yaml's `holdout_families:` field MUST stay in sync with this file.

## Deleted (Apr 30)

21 obsolete configs removed (a100/h100 hardware variants, mix_* experiments,
curriculum_*, qwen3vl_2b_*, smoke.yaml). All produced 0 paper-relevant
checkpoints OR were superseded by v3/v4 series.
