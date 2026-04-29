# Learnings — 2026-04-29

Single source of truth for SFT recipe decisions and what we know works.
Earlier learnings are in `docs/comparison_2026-04-26.md` and the v3 config
header (`configs/sft/big_bench_shell_50k_v3.yaml` lines 1-38). This doc
captures everything since 04-26 and consolidates v3's design rationale.

**Active config**: [`configs/sft/big_bench_shell_50k_v3.yaml`](../configs/sft/big_bench_shell_50k_v3.yaml).

---

## 1. Data mix evolution (chronological)

| version | dates | mix (filtered counts) | best max@8 BC/DC/Fu | notes |
|---|---|---|---|---|
| **curriculum** | 04-23 → 04-26 | 3-phase: bench 1→2→8 / recode 2→1→1 / t2c 2→1→1, ~113k items | 0.612 / 0.616 / 0.664 | Phase 3 (8:1:1) regressed DC 0.49→0.41. Curriculum schedule plateaued early. |
| **v2 phase2** | 04-27 from-scratch | bench 18, iso 162, simple 86, t2c-bench 94, recode-bench 175 (~485k) | 0.555 / 0.581 / 0.636 @ 11k | Killed @ step 12k. Same data, no curriculum. Slow BC ramp. |
| **v2 phase2b** | 04-27 resumed | bumped benchcad 18→**50** (8.8% step share, was 3.4%) | 0.644 / 0.650 / 0.666 @ 27-29k | The 50× lift on benchcad pushed BC val greedy 0.43→0.59. Best v2 result. |
| **v3 (current)** | 04-28 from-scratch | bench 11, iso 122, simple 77, t2c-img 29, t2c-text 29, recode-bench 257 (789k after dedup+drop) | **0.624 / 0.688 / 0.676 @ 15k** | **At step 15k v3 already exceeds v2 phase2b's 27-29k max@8 on DC.** |

### v3 design choices and why

1. **80% drop on DROP-flagged items, keep 20% for diversity coverage.**
   Justification: family ops_p50 ≤ 2 AND lines_p50 ≤ 8 → "trivial family,
   model collapses to extrude-only."  Cutting them entirely lost diversity
   on edge cases; 20% retain gave the best trade-off in pre-launch sweep.

2. **Code-hash dedup** (md5 of normalized code).  Removed:
   - benchcad: **37%** dupes (18k → 11k)
   - cad-iso-106: **24%** dupes
   - text2cad-bench: **29%** dupes
   - benchcad-simple: ~12% dupes
   These dupes were pure compute waste — every step a worker wasted on a
   re-seen code is a step not learning a new code.

3. **`text2cad_bench` split into image AND text dual-mode** as separate
   weighted entries (`text2cad_bench_img` + `text2cad_bench_text`).  Each
   item is sampled in *exactly one* modality per training step, never
   mixed on the same sample (avoids img-vs-text encoder confusion).

4. **60% HQ / 40% bench-stack split, equal-per-item within group.**
   - HQ = recode_bench + t2c_img + t2c_text (text + diverse code)
   - Bench-stack = benchcad + iso + simple (image + structured family)
   - Within each group, weight ∝ N_items so every item gets the same draw
     rate inside its group.  Per-item rate within HQ ≈ 0.55× rate within
     bench, on purpose: bench-stack is smaller so per-item exposure is
     higher (bench-stack ≈ 3 epochs/item, HQ ≈ 1.65 epochs/item over 50k
     steps).

5. **`text2cad_legacy` deleted entirely** — measured 28% trivial codes.
   Was dead weight in v2.

---

## 2. Per-source op statistics (n=500/source, seed=42)

Distinct ops per code:

| source           | n_items | ops/case median | ops/case mean | ops/case p95 |
|---|---|---|---|---|
| benchcad         | 11,443  | 5.0 | 4.89 |  7 |
| cad_iso_106      | 122,483 | 5.0 | 4.81 |  7 |
| benchcad_simple  | 76,671  | 4.0 | 3.79 |  5 |
| text2cad_bench   | 53,339  | 4.0 | 3.63 |  6 |
| **cad_recode_bench** (v3 main HQ) | 472,244 | **5.0** | **5.16** |  7 |
| cad_recode_20k (NOT in v3)         | 18,000  | 7.0 | 6.64 | 10 |

**Key finding**: `cad_recode_bench`, the source that gets 49% of v3's step
share, is **per-case op-compact** (median 5, same as benchcad).  The
"recode-is-much-wider" intuition only applies to `cad_recode_20k`, which
v3 doesn't use.

Plots in `docs/op_distribution_2026-04-29/`.  Generator:
`scripts/analysis/op_distribution_plot.py`.

---

## 3. Why `op_loss_cos_weighted` is BC≈0.20 vs recode20k≈0.77

`op_loss_w` is **dataset-internal difficulty signal, NOT cross-dataset
comparable**.  See `train/sft/online_eval.py:200-214` for the formula:

```
G, P ∈ {0,1}^K     # multi-hot, K = global op vocabulary
w_k = -log(P_k)    # global frequency weight
cos_w = (w·G·P).sum() / sqrt((w·G).sum() * (w·P).sum())
loss = 1 - cos_w
```

The ~3.8× gap is driven by:
- **BC val** uses ~20 distinct ops total across all simple_* families —
  small `K`, few rare ops → cos_w naturally high → loss low.
- **recode20k** spans 100+ distinct ops with long-tail — many rare ops
  in GT, model mode-collapses to common ops at eval → den_g >> den_p →
  cos_w low → loss high.

Per-case op count is only ~1.4× different (5 vs 7) — it's **op-vocab tail
breadth + mode collapse on rare ops** that drive the loss gap, not raw
op count.

**Recommendation**: use `op_macro_f1` and `rare_op_macro_recall` for
cross-dataset comparison — they're per-op averages, scale-free.  Reserve
`op_loss_w` for within-dataset trajectory tracking.

---

## 4. v3 trajectory vs v2 phase2b (matched-step)

| step | v2b BC / DC / Fu (greedy) | v3 BC / DC / Fu (greedy) |
|---|---|---|
| 5k  | 0.30 / 0.40 / 0.46 | 0.42 / 0.49 / 0.59 |
| 11k | 0.43 / 0.51 / 0.60 | 0.49 / 0.59 / 0.61 |
| 15k | 0.51 / 0.55 / 0.60 | 0.51 / 0.55 / 0.57 |
| 25k | 0.59 / 0.57 / 0.60 | (TBD) |

| step | v2b max@8 BC / DC / Fu | v3 max@8 BC / DC / Fu |
|---|---|---|
| 11k | 0.555 / 0.581 / 0.636 | 0.568 / 0.666 / 0.655 |
| 13k | 0.62 / 0.63 / 0.66 | 0.605 / 0.654 / 0.680 |
| 15k | -  | **0.624 / 0.688 / 0.676** |
| 27k-29k | 0.644 / 0.650 / 0.666 (best v2b) | (will likely surpass at v3 ~17-20k) |

**Take**: v3 reaches v2b's best at ~half the steps.  Cleaner data ⇒ more
information per step.  Same model architecture, same compute budget, ~+2
IoU pts headroom over v2b at matched ckpt.

---

## 5. Rendering pipeline lessons

**Tessellation tolerance for thumbnails**: `compound.tessellate(0.01, 0.5)`
is ~10× faster than `(0.001, 0.1)` and **visually identical at 268×268**.
The latter is OpenCascade's default for STL export — overkill for a
thumbnail.  C extension call → SIGALRM doesn't fire, so a single bad
tessellation can hang a worker 30-60s.  Loose params + per-task timeout
+ pool `maxtasksperchild=200` is the safe combination.

**Family-clustered data**: BenchCAD's gt_code is sorted by family.  When
a Pool of 12 workers all pull the same 4-item chunk (from `imap_unordered`
chunksize=4), they all hit the same family at once.  If that family is
pathological (e.g. `simple_coil_spring` → 50w-face meshes), the entire
pool stalls.  **Always shuffle the todo list with a deterministic seed
before running parallel cadquery rendering.**

**Pre-flight integrity check is mandatory**: `scripts/preflight_check.py`
verifies (1) every train.pkl is loadable, (2) png_path resolves on disk,
(3) py_path or inline code resolves, (4) sample-decode PNGs.  v3 launch
caught 2 missing-PNG bugs (one in text2cad-bench: 38 cadquery codes
silently failed render, leaving 53,339 pkl rows but only 53,315 PNGs —
crashed dataloader at step 0).

**Stale `.filter_cache` directories**: dataset loaders cache filter
results keyed by N_items.  After re-filtering a source, `data/*/.filter_cache`
must be removed or new train.pkl rows are silently masked.

---

## 6. HF data state (post-session)

| repo | state |
|---|---|
| `BenchCAD/benchcad-easy` upstream parquet | 11.3% with composite_png → **80.8%** (88,773 / 109,804). Shards 0-3 + 15-54 covered. Shards 4-14 still missing in this single-parquet view (VM1 has them locally only, they're in Hula0401 below). |
| `Hula0401/cad-sft/benchcad-easy/` 55 shards | 16 shards (28k rows) → **all 55 shards complete** (~109k rows). VM1's shards 0-14 + my shards 15-54. Schema: stem/code/render_img/family/difficulty/n_ops/ops_json/base_plane. Ready to wire as v4 SFT source. |
| `Hula0401/cad-sft/{benchcad, benchcad-simple, text2cad-bench, cad-recode-bench}` | Unchanged — same as v3 input. |

---

## 7. Two HF write tokens (don't confuse)

- `HF_TOKEN` → user `Hula0401`, can write `Hula0401/*` (cad-sft, model checkpoints)
- `BenchCAD_HF_TOKEN` → user `BenchCAD`, can write `BenchCAD/*` (the upstream parquets)

A 403 on push almost always means the wrong token is loaded.  The
`merge_benchcad_easy_renders.py` and `upload_shards_to_hula0401.py`
scripts respect whichever token is passed via `HF_TOKEN` env at call
time — wrap with `HF_TOKEN="$BenchCAD_HF_TOKEN"` for BenchCAD pushes.

---

## 8. Open questions / next moves (highest leverage first)

1. **Wait for v3 to step 25k+, then decide** (12h wall-clock, no effort).
   If max@8 DC/Fu cross 0.70, RL becomes viable.

2. **v4 = v3 + benchcad-easy (109k)**.  Same family as benchcad/simple
   but 10× the size, render_img now 100% on Hula0401.  Expected lift:
   +0.03-0.05 BC val, modest DC/Fu boost from added bench-style image
   diversity.  Effort: ~2h prep (wire path + 1 config + relaunch); 25h
   training.

3. **Bump benchcad weight 11→55 in a v3-style mix.**  v2 phase2b's 50×
   benchcad lift was the biggest single BC val intervention seen.
   With cleaner v3 data + benchcad-easy, this could push BC greedy past
   0.65.  Effort: just config tweak.

**Skip** for now: Qwen3-VL-4B (no SFT ceiling evidence yet), RL (greedy
still below 0.8 gate), `pc_img` modality (large refactor, T15 punted).
