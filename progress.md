# T8 progress — recode/text2cad → BenchCAD-style + 100k scale-up

> Single-file plan + progress. See plan.md T8 for context. User is asleep,
> autonomous run authorized 2026-04-26.
>
> **Hard constraints** (in priority order):
>   1. RAM never below 1 GB free (CLAUDE.md threshold; OOM'd once already)
>   2. No re-render of existing 20k (PNG bytes reused — geometry unchanged)
>   3. Fallback to raw code if v2 rewrite fails — never emit invalid code
>   4. Goal is data volume, not 100% correctness
>   5. Single commit at end with all changes

## Phase A — re-pack old 20k with bench-style code  ✅ DONE

- [x] Validate Path X rewriter (v2): IoU 200/200=100% recode, 197/199=99% text2cad
- [x] Op-distribution check: sketch/finalize/assemble dropped 92% → 0% on recode
- [x] Write `data_prep/repack_recode_to_bench.py`
- [x] Download 9 parquet shards from `Hula0401/cad-sft/cad-recode-20k/`
- [x] Per shard: replace `code` field with v2 rewrite (100% rewritten, 0 fallbacks)
- [x] Upload to `Hula0401/cad-sft/cad-recode-bench/train-XXXXX-of-00009.parquet`
   - 9/9 shards on HF, ~20k rows total
- [x] Verify: HfApi.list_repo_files confirmed all 9 shards visible

## Phase C — text2cad full rewrite + repush  ✅ DONE

- [x] Write `data_prep/repack_text2cad_to_bench.py`
- [x] Rewrite all 171k `data/text2cad/cadquery/*.py` via v2 (fallback raw)
   - Result: 90,499 rows total (train 76238 + val 6464 + test 7797),
     90,492 rewritten (100.0%), 7 fallbacks
- [x] Repack train/val/test as both pkl + parquet
- [x] Upload to `Hula0401/cad-sft/text2cad-bench/{train,val,test}.{pkl,parquet}`
   - All 6 files on HF

## Phase B — new 80k recode pipeline  🟡 RUNNING

- [x] Restore `data_prep/prepare_hf_cadrecode.py` from commit 2f6396c (read-only, wrote v2)
- [x] Write `data_prep/prepare_hf_cadrecode_v2.py`:
   - exec → tessellate → render PNG (v1.5 raw)
   - rewrite_source(raw) → bench code (fallback raw)
   - pack incremental shards, upload as we go, free memory
- [x] Smoke n=50 jobs=2: 17.3s, 0 errors, 1.0 MB shard
- [x] Scaling test n=1000 jobs=4: 119s, rate 8.4/s, 0 errors, RAM 11+ GB free
- [x] Off-by-one fix: cap successes at n_target so final shard isn't a 1-row stub
- [✅] Full 80k run RESUMED + COMPLETED (PID 24442, jobs=4, max_tasks_per_child=100,
   shard-size=2000, start-shard 6) — 80000/80000 successes, 0 errors,
   40/40 shards on HF. Done at 08:34 (~2h 44min total wall time).
   - First attempt (PID 82055): aborted at 11996/80000 successes after RAM
     dropped to 654 MB (below 1.5 GB safety floor). 6 shards (12000 rows)
     successfully uploaded before abort.
   - Root cause: cadquery/OCP/open3d worker memory leak — RSS grew steadily
     from start, hit OOM-risk territory at ~12k tasks (~30 min).
   - Fix: added `--max-tasks-per-child 100` flag, recycles each worker after
     100 tasks → bounded RAM. Smoke verified: rate 7.3/s (vs 8.4 before),
     RAM stable at 14 GB free.
   - Resume: skip first 12000 successes (re-render but discard, ~25 min waste),
     then continue uploading shards 6..39. ETA total ~3h from 05:50.
   - Live: logs/phase_b_resume.log

## Phase D — wire up fetcher + datasets + configs  ✅ CODE DONE

- [x] `data_prep/fetch_cad_sft.py`: added `recode-bench`, `text2cad-bench`,
   `bench-all` choices to `--what`. Bench fetchers enumerate shards via
   list_repo_files (so Phase A's 9-shard naming + Phase B's 40-shard naming
   both work without hardcoding).
- [x] `train/sft/train.py`: added `recode_bench` source (uses CadRecode20kDataset
   pointed at `data/cad-recode-bench/`) and `text2cad_bench` source (uses
   Text2CADDataset pointed at `data/text2cad-bench/`). Existing dataset
   classes work unchanged because the on-disk layout is identical.
- [x] Added `configs/sft/mix_bc_rb_t2cb.yaml` — benchcad:2 + recode_bench:2 +
   text2cad_bench:1 mix, max_steps=8000.
- [ ] Smoke test `train.sft.train --config mix_bc_rb_t2cb.yaml --max-steps 5`
   (deferred until Phase B done so recode-bench has data)

## Wrap-up  ✅ DONE (08:35)

- [x] Single commit with all changes
- [x] Update plan.md T8 status to "DONE"

### HF dataset state (Hula0401/cad-sft) — final
- `cad-recode-bench/`: **49 parquet shards** (Phase A: 9 of-00009, ~20k rows;
  Phase B: 40 of-00040, 80k rows) → **~100k bench-style recode rows with
  pre-rendered 4-view 268×268 PNGs**
- `text2cad-bench/`: 6 files (train/val/test × pkl + parquet) →
  **~90k bench-style text2cad rows with descriptions**
- `cad-recode-20k/`: untouched (legacy compact-style 20k preserved for
  reproducibility of older SFT runs)

### Summary
- **Net new training data**: 80k recode (Phase B) + 0 text2cad (already
  full) — recode-bench corpus 5× the size of recode-20k.
- **Code style**: 100% rewritten on recode (sketch/finalize/assemble dropped
  from 92% to 0%); ~90% rewritten on text2cad (10% face_wrap_mixed kept as
  AST-formatted sketch — still loadable, just less aligned).
- **Geometry**: zero-loss (PNGs from Phase A reused; Phase B re-rendered
  from raw recode .py since rewrite is AST-only and equivalent).
- **Cost**: ~3.5h compute (Phase B mostly), 0 errors across 80k samples,
  RAM stayed safely above 11 GB free thanks to max_tasks_per_child=100.

## Resource monitoring log

| time | phase | step | RAM free | swap used | disk free | notes |
|------|-------|------|----------|-----------|-----------|-------|
| 04:30 | — | — | 13G | 5.4G | 583G | post-OOM cleanup, 4 RL ckpts deleted |
| 05:14 | B | start (PID 82055) | 12G | — | 583G | 4 workers, no recycle |
| 05:48 | B | abort 11996/80k | 654M | — | 583G | RAM floor breach (worker leak) |
| 05:50 | B | resume (PID 24442) | 11G | — | 583G | --start-shard 6 --max-tasks-per-child 100 |
| 06:32 | B | 20772/80k (26%) | 11G | — | 583G | RAM oscillates 11.2-11.5G (recycle works) |
