# SFT Configs — Paper §7 v4-holdout study

Four runs cover the §7.a/§7.b 4-line plots. Naming scheme is **role-based**:
the config name tells you what role the run plays in the §7 figure.

| Config | Role | Holdout | benchcad-easy | bench-stack | Mix (HQ/bench) |
|---|---|---|---|---|---|
| `baseline.yaml`     | Floor — no BenchCAD data       | n/a (no bench) | ✗ | ✗ | 100/0  |
| `ood.yaml`          | Held-out fams, plain bench     | 10 fams        | ✗ | ✓ | 60/40  |
| `ood_enhance.yaml`  | Held-out fams + easy supplement| 10 fams        | ✓ (80k) | ✓ | 60/40  |
| `iid.yaml`          | Full bench, no holdout (upper) | none           | ✓ | ✓ | 60/40  |

`holdout_families.yaml` is the canonical list of held-out families
(loaded by `common/holdout.py` for offline scripts and by `train/sft/online_eval.py`
for the OOD bucket). Each SFT yaml's `holdout_families:` field MUST stay in sync
with this file.

## Run name + checkpoint mapping

| Config | Run name (wandb / ckpt dir) | Ckpt path | HF repo | Status |
|---|---|---|---|---|
| `ood_enhance.yaml`  | sft-s50k-lr2e-4-b8a4-img-0430-0828 | `/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/` | `Hula0401/cadrille-qwen3vl-2b-v4-holdout-50k` | 🟡 in progress (will stop at next save) |
| `baseline.yaml`     | (TBD)                              | (TBD)                                                       | `Hula0401/cadrille-qwen3vl-2b-baseline-50k`        | ❌ pending — chain run #1 |
| `ood.yaml`          | (TBD)                              | (TBD)                                                       | `Hula0401/cadrille-qwen3vl-2b-ood-50k`             | ❌ pending — chain run #2 |
| `iid.yaml`          | (TBD)                              | (TBD)                                                       | `Hula0401/cadrille-qwen3vl-2b-iid-50k`             | ❌ pending — chain run #3 |

Legacy v3 (50k done, no HF upload, ckpt at
`/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/`) is kept on disk as
the IID upper-bound for offline cad_bench_722 eval. The v3 yaml itself was
removed from `configs/sft/`; reference it through the saved
`run_config.yaml` inside that ckpt dir if you need to re-launch.

## Holdout family list

Single source: `holdout_families.yaml`. Currently 10 families:
`tapered_boss, taper_pin, venturi_tube, bucket, dome_cap, nozzle, enclosure, waffle_plate, bolt, duct_elbow`

Selection criterion: v3 baseline `essential_pass ≥ 0.80` on its own training
set (i.e., families demonstrably learnable when seen). Held out from train so
that v4-* runs can probe family-level generalization.

## "shell" — naming history

The deleted `big_bench_shell_50k_*.yaml` files were the v3/v4 series. "shell"
referred to BenchCAD's conventional shell-style cadquery layout (sketch →
direct Workplane → `.transformed` → `.cut(mode='s')`), produced by AST-rewrite
of raw cad-recode-style code via `data_prep/rewrite_recode_to_benchcad_v2.py`.
Both the rewritten cad-recode-bench and text2cad-bench datasets share this
style — hence the naming on the legacy configs.

The new role-based names drop the `shell` prefix: all four runs use the same
shell-style data, so it carries no information.
