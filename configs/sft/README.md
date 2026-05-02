# SFT Configs — Paper §7 study (v1 mech-OOD + v2 op-pattern-OOD)

## §7 v2 — bench-simple op-pattern OOD (5-line ablation, current)

| Config | mech-OOD (10 v1 fams) holdout | bench-simple OOD (10 v2 fams) holdout | benchcad-easy in train? | Mix HQ:bench | total_train_dp |
|---|---|---|---|---|---|
| `baseline_v2.yaml` | n/a (no bench)        | n/a (no bench)              | ✗ | 100/0 | 500k |
| `iid_easy_v2.yaml` | n/a (no bench)        | n/a (no bench)              | ✓ | 60/40 | 500k |
| `ood_v2.yaml`      | **YES** (held out)    | **YES** (held out from train, eval bucket) | ✗ | 60/40 | 500k |
| `ood_enhanced_v2.yaml` | NO (in train)     | **YES** (held out from train, eval bucket) | ✗ | 60/40 | 500k |
| `iid_v2.yaml`      | NO (in train)         | **YES** (held out from train; 44 IID fams in train, eval bucket on 10 OOD) | ✗ | 60/40 | 500k |

`holdout_families_v2.yaml` is the canonical 10 op-pattern fams:
`simple_revolve, simple_loft, simple_polygon_cut, simple_extrude_hole,
simple_box_hole_fillet, simple_cyl_cut, simple_revolve_cut,
simple_extrude_hole_fillet, simple_loft_cut, simple_taper_extrude_chamfer`.

`total_train_dp: 500000` — caps unique training rows to 500k across all
sources, subsampling each by mix-weight ratio (deterministic seed=42).
Equalises pool size across the 5 lines so the only confound is content
not data volume. Set to `null` to disable (use full source sizes).

Eval bucket: `bench-simple OOD` = 50 stratified samples (10 fams × 5)
loaded from `data/benchcad-simple/val.pkl` via online_eval. ess spec
auto-derived from family name (e.g. `simple_revolve_cut → [revolve, cut]`).

## §7 v1 — mech-OOD legacy (kept for §7 main figure)

Four roles in the §7 figures, but only **three** trained runs needed —
v3 already plays the iid role (it saw all 106 families during training).

| Config | Role in §7 figures | Holdout | benchcad-easy | bench-stack | Mix (HQ/bench) | Status |
|---|---|---|---|---|---|---|
| _legacy v3 ckpt_    | (1) **iid** ceiling — v3 saw all families | n/a | ✗ | ✓ | 36/64 | ✅ done (50k) |
| `ood.yaml`          | (2) ood — held-out fams, plain bench       | 10 fams | ✗ | ✓ | 60/40 | 🟡 chain run 2 |
| `ood_enhance.yaml`  | (3) ood_enhance — holdout + easy supplement| 10 fams | ✓ (80k) | ✓ | 60/40 | ✅ stopped @ 24k |
| `baseline.yaml`     | (4) baseline — HQ only (floor)             | n/a (no bench) | ✗ | ✗ | 100/0 | ✅ stopped @ 11k |
| ~~`iid.yaml`~~      | (skipped — v3 covers this role)            | none | ✓ | ✓ | 60/40 | ⛔ not run |

`holdout_families.yaml` is the canonical list of held-out families
(loaded by `common/holdout.py` for offline scripts and by `train/sft/online_eval.py`
for the OOD bucket). Each SFT yaml's `holdout_families:` field MUST stay in sync
with this file.

## Run name + checkpoint mapping

| Config | Run name (wandb / ckpt dir) | Ckpt path | HF repo | Status |
|---|---|---|---|---|
| _legacy v3_         | sft-s50k-lr2e-4-b8a4-img-0428-1320 | `/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/` | (local only, 50k done) | ✅ iid line in §7 figs |
| `ood_enhance.yaml`  | sft-s50k-lr2e-4-b8a4-img-0430-0828 | `/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828/` | `Hula0401/cadrille-qwen3vl-2b-v4-holdout-50k` | ✅ stopped @ 24k |
| `baseline.yaml`     | sft-s50k-lr2e-4-b8a4-img-0501-0629 | `/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0501-0629/` | `Hula0401/cadrille-qwen3vl-2b-baseline-50k`   | ✅ stopped @ 11k (OOD ess_pass=0 plateau established) |
| `ood.yaml`          | sft-s50k-lr2e-4-b8a4-img-0501-1753 | `/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0501-1753/` | `Hula0401/cadrille-qwen3vl-2b-ood-50k`        | 🟡 in progress |
| ~~`iid.yaml`~~      | —                                  | —                                                            | —                                              | ⛔ not run (v3 plays this role) |

`iid.yaml` is kept on disk as a documented option but the §7 chain skips it:
v3 trained on all 106 BenchCAD families, so on the held-out families it IS
the iid line (those families are in-distribution for v3). Re-launch via
`launch_run configs/sft/iid.yaml iid` in `scripts/launch_chain_runs.sh`
only if a recipe-matched iid is needed (controls v3's 36/64 mix vs v4's
60/40 mix).

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
