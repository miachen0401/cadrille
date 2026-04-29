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

## Phase F — Import BenchCAD/cad_simple_ops_100k as image-conditioned bench data  🟡 RUNNING

User asked for more bench-style data. Found `BenchCAD/cad_simple_ops_100k`:
99k samples already in BenchCAD shell style (no rewrite needed!) with `step_bytes`
embedded for fast geometry (~30ms STEP→mesh vs ~1s exec).

- [x] Inspect `BenchCAD/cad_simple_ops_100k`: 12 parquet shards, ~98880 rows,
   schema (stem, code, step_bytes, family, difficulty, n_ops, ops_json,
   base_plane). Code includes a 526-char OCP HashCode shim header which we
   strip before training.
- [x] Verify: STEP→mesh works (0.03s), exec FAILS without show_object stub
   (we use STEP path).
- [x] Write `data_prep/import_benchcad_simple.py`: 1) strip shim, 2) STEP→mesh
   via OCP STEPControl_Reader, 3) render via common.meshio.render_img,
   4) pack stem/code/render_img/family/difficulty/n_ops/ops_json/base_plane.
- [x] Smoke n=20 jobs=2: 10.7s, rate 1.88/s (startup-dominated)
- [x] Scale n=200 jobs=4: 24s, rate 8.32/s, 0 errors, RAM stable
- [x] Wire fetcher (`fetch_cad_sft.py --what benchcad-simple` or `bench-all`)
- [x] Wire `benchcad_simple` source in train.sft.train (uses CadRecode20kDataset
   pointed at `data/benchcad-simple/` since on-disk layout is identical)
- [x] New config `configs/sft/mix_all_bench.yaml` —
   benchcad:2 + benchcad_simple:2 + recode_bench:2 + text2cad_bench:1
- [✅] First run 78k (39/50) + smart resume v3 added 10,175 → **88,175 / 45 shards**
   - **First attempt** (16:21–19:53, 3.5h): 78,000 rows / 39 shards / 0 errors.
     Throughput collapsed to 0.6/s on upstream shards 9-11 because parametric-
     surface families (sweep_helix, sweep_spline, twist_*) tessellate to
     50k-500k triangles — render takes 5-30s vs 0.15s baseline.
   - **Smart resume v3** (21:13–21:36, 23.5 min): 10,175 new / 6 shards /
     0 errors / rate 7.21/s. Three new flags in import_benchcad_simple.py:
     - `--start-upstream-shard 9 --skip-rows-in-first-upstream 3840`: don't
       re-render shards 0-8 (already on HF) or first 3840 of shard 9 (done)
     - `--skip-family-substr helix spline twist`: cheap pre-filter at the
       main process. Drops 89% of shard 9 remaining (3905/4400), 24% of
       shard 10 (1940/8240), ~58% of shard 11 — BEFORE any STEP load
     - `--per-task-timeout-sec 30`: SIGALRM safety net (didn't fire — 0 errors)
   - **Final**: 88,175 rows / 45 shards on HF
     `Hula0401/cad-sft/benchcad-simple-100k/train-{00000..00044}-of-00050.parquet`
     ("-of-00050" suffix nominal; fetcher uses list_repo_files so all 45 load.)
   - **Excluded by design**: ~9k samples from helix/spline/twist families
     (extreme tessellation cost vs marginal training value). To include them
     later: drop --skip-family-substr and budget 8+ hours.

- [✅ PARTIAL] v4 attempt 21:49–23:44 (1h55m): user wanted to also include
   the 10.7k helix/spline/twist samples skipped by v3. Added two flags:
   - `--only-family-substr` (inverse of skip): keep ONLY rows matching
   - tightened `--max-tasks-per-child=50` (vs v3's 100) to bound RAM more
   Run produced 2,000 more rows (1 shard) + 1,653 in-buffer when RAM hit
   the safety zone:
   - SIGALRM 30s timeout doesn't preempt OCP C-extension code → some
     tessellate calls ran 11+ minutes blocking workers
   - Worker RSS climbed to 4.5 GB on heavy meshes; total system RAM hit
     2.7 GB available with 1 GB headroom above the 1.5 GB floor
   - Throughput dropped from 0.7/s → 0.05/s as workers got stuck
   - Killed manually before auto-abort would have triggered
   - **Net v4 gain: +2,000 rows uploaded (1 shard, idx 45-of-50)**
   - Lesson: the SIGALRM timeout works for Python-level stalls but not for
     OCP/OpenCASCADE tessellate, which spends seconds-to-minutes in C++
     without checking Python signals. To process these complex families
     would need an OS-level watchdog process kill — out of scope for now.

## Phase F final HF state

```
benchcad-simple-100k/  46 parquet shards = 90,175 rows
  ├─ v1 (first attempt):  shards 0..38 of-00050  →  78,000 rows
  ├─ v3 (family blacklist): shards 39..44 of-00050 → 10,175 rows
  └─ v4 (only-family):     shard 45 of-00050     →  2,000 rows
                                                  ───────────
                                                  ~90k bench-style image+code samples
```
~9k helix/spline/twist samples remain unprocessed (would need OS-level watchdog).

## Phase B' — scale recode-bench to 140k  ✅ DONE 2026-04-27

User requested another +20k–40k recode-bench. Took the upper end (+40k):

- Script: `data_prep/prepare_hf_cadrecode_v2.py` (unchanged from Phase B)
- Command: `--offset 100000 --n 40000 --workers 4 --shard-size 2000 --max-tasks-per-child 100`
- Sampling slice [int(100000×1.3) : int(140000×1.3)] = **[130000, 182000]** of the
  seed=42 shuffled cad-recode-v1.5/train candidate list — zero overlap with
  Phase A (took [0:26000]) or Phase B ([26000:130000]).
- Started 01:42, completed 03:35. Wall time 113.7 min, rate 5.86/s
  (recent steady ~8/s; cumulative pulled down by ~5 min worker cold-start).
- Result: **40,000 successes / 0 errors** / 20 shards on HF
- Output: `Hula0401/cad-sft/cad-recode-bench/train-{00000..00019}-of-00020.parquet`
  (distinct from Phase A's of-00009 and Phase B's of-00040 naming; fetcher
  uses list_repo_files so all 69 shards load without naming-convention concern.)

### cad-recode-bench corpus after Phase B'
```
Phase A:   9 shards of-00009  →  20,000 rows  (re-pack of original 20k)
Phase B:  40 shards of-00040  →  80,000 rows  (offset=20000)
Phase B': 20 shards of-00020  →  40,000 rows  (offset=100000)
                              ────────────────
                              140,000 rows total bench-style recode + 4-view PNG
```

### Total bench data on HF (Hula0401/cad-sft) after T8 + Phase F + Phase B'
```
benchcad             ~20k
cad-recode-bench    ~140k  ← was 100k
text2cad-bench       ~90k
benchcad-simple-100k ~90k
                     ─────
                    ~340k bench-style training samples
```

## Phase Scale-Up — recode-bench to ~520k via 8 chained 50k batches  ✅ DONE 2026-04-27

User requested another scale-up to ~500k+ in 50k chunks. Total +380k samples
added across 8 chained batches (Batch C through J), cumulative recode-bench
went from 140k to ~520k.

### Bug found + fixed during the run

**Bug 1: chain mode broken by success-skip logic**
- Symptom: Batch D ran for 2h with 0 shards uploaded
- Cause: `n_skip_for_resume = args.start_shard * args.shard_size` was
  designed for crash-recovery resume mode but kicked in for non-zero
  --start-shard chained runs too. With --start-shard 25, it discarded the
  first 50000 successes — exactly equal to the batch's --n target — so
  successes_buf was always empty and zero shards were written.
- Fix: added `--total-shards-override` flag; when present, the script
  treats --start-shard as a filename offset only and does NOT skip any
  successes (chain mode).

**Bug 2: silent HF upload hang**
- Symptom: shard 41/200 in Batch D wrote locally but `api.upload_file`
  hung forever (3 ESTAB connections to HF, 0% CPU, no exception). Same
  hang pattern seen previously in Phase A.
- Fix: wrapped upload in a daemon thread with 5min wall-clock timeout +
  3 retries. Hung threads abandoned; orphans hold network sockets but
  don't block the chain. Saw 4 hangs across the 6 E-J batches, all
  auto-recovered without intervention.

**Bug 3: Discord webhook 403**
- Symptom: wrapper notify() got HTTP 403 Forbidden
- Cause: Python `urllib` default User-Agent (`Python-urllib/3.11`) is
  on Discord's blocklist
- Fix: explicit `User-Agent: cadrille-batch-runner/1.0` header in the
  wrapper notify() function

### Batch breakdown

| batch | offset | start-shard | result |
|-------|-------:|------------:|--------|
| C | 140000 | 0 | ✅ 50k / 0 errors / 1h54m |
| D | 190000 | 25 | ⚠️ partial 30k (15 of 25 shards) before upload hang; recovery skipped |
| E | 240000 | 50 | ✅ 50k / 0 errors / 1h47m |
| F | 290000 | 75 | ✅ 50k / 0 errors / 1h42m |
| G | 340000 | 100 | ✅ 50k / 0 errors / 1h42m |
| H | 390000 | 125 | ✅ 50k / 0 errors / 1h52m / 2 hangs auto-recovered |
| I | 440000 | 150 | ✅ 50k / 0 errors / 1h42m / 1 hang auto-recovered |
| J | 490000 | 175 | ✅ 50k / 0 errors / 1h42m |

Total runtime: ~12h (with retries + restart from D bug).
Aggregate hangs across all batches: 6, all auto-recovered by 5min thread
timeout + retry. **0 actual data losses; D's missing 20k was a deliberate
skip after the fix.**

### cad-recode-bench corpus final state
```
Phase A:    9 shards of-00009 →  20k    [0:26000]
Phase B:   40 shards of-00040 →  80k    [26000:130000]
Phase B':  20 shards of-00020 →  40k    [130000:182000]
Batch C:   25 shards of-00200 →  50k    [182000:247000]
Batch D:   15 shards of-00200 →  30k    [247000:312000] (partial)
Batch E:   25 shards of-00200 →  50k    [312000:377000]
Batch F:   25 shards of-00200 →  50k    [377000:442000]
Batch G:   25 shards of-00200 →  50k    [442000:507000]
Batch H:   25 shards of-00200 →  50k    [507000:572000]
Batch I:   25 shards of-00200 →  50k    [572000:637000]
Batch J:   25 shards of-00200 →  50k    [637000:702000]
                              ─────────
                              ~520k bench-style image+code samples
```
259 parquet shards total on `Hula0401/cad-sft/cad-recode-bench/`. Slice
[0:702000] of seed=42 shuffled cad-recode-v1.5/train (981,865 candidates).

### Total bench data on HF (Hula0401/cad-sft) — final
```
benchcad             ~20k
cad-recode-bench    ~520k  ← was 140k
text2cad-bench       ~90k
benchcad-simple-100k ~90k
                     ─────
                    ~720k bench-style training samples
```

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
| 08:34 | B | DONE 80k/80k | 14G | 0 | 583G | Phase B complete, 0 errors |
| 16:21 | F | start (PID 33980) | 10G | — | 582G | benchcad-simple-100k import |
| 17:25 | F | 26k/99k (26%) | 10G | — | 582G | rate 6.9/s |
| 18:26 | F | 53k/99k (54%) | 9G | — | 582G | rate 7.1/s |
| 19:08 | F | 72k/99k (73%) | 9G | — | 582G | rate 7.3/s |
| 19:27 | F | 77k/99k (78%) | 4G | — | 582G | upstream shard 10 starts, complex shapes slow |
| 19:53 | F | STOP 78k/99k | 8G | — | 580G | killed at shard 39, rate had dropped to 0.7/s |
