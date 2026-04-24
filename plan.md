# Cadrille SFT + RL — Environment & Smoke Plan

> **Repo simplification**: `docs/repo_simplification.md` — 7-step reorg completed
> 2026-04-24 on branch `revision` (commits 764c96b..cd0e2f2). Canonical layout
> now in CLAUDE.md reflects the new `common/`, `train/`, `eval/`, `data_prep/`,
> `bench/`, `experiments/` structure.
>
> Session source: `notes/sft_setup_2026-04-24.md` (prior container session).

## Host state (2026-04-24 brave-hubble)

- A100 80GB, 118 GB RAM, 48 GB free on `/`
- Ubuntu, user `ubuntu` with passwordless sudo, NOT in container
- `/home/ubuntu/cadrille` — repo, tip `a13a104`
- `.env` at `/home/ubuntu/cadrille/.env` has HF_TOKEN, WANDB_API_KEY, GITHUB_PAT_TOKEN, BenchCAD_HF_TOKEN
- apt deps for Open3D headless build **already installed** (libosmesa6-dev, libgl1-mesa-dev, libglu1-mesa-dev, libglew-dev)
- `uv 0.11.7` installed at `~/.local/bin/uv`

## Execution order (one-by-one; parallel where marked ⇢)

### Phase 1 — venv rebuild  ✅ DONE

- [x] **1.1** Delete stale `.venv` (shebangs pointed to `/workspace/.venv` from container copy)
- [x] **1.2** `uv sync --no-install-project` — torch 2.5.1+cu124, transformers 4.50.3, etc.
- [x] **1.3** `uv pip install setuptools`
- [x] **1.4** pytorch3d 0.7.8 (git 06a76ef, nvcc built for sm_80)
- [x] **1.5** cadquery 2.5.0.dev0 (git e99a15d)
- [x] **1.6** flash-attn 2.7.2.post1
- [x] **1.7** GPU sanity: A100 80GB ✓
- [x] **1.8** `scripts/setup.sh` patched with `uv pip install setuptools` before git installs

Log: `logs/git_installs_20260424_053833.log`

### Phase 2 — Open3D headless (source build)  ✅ DONE

- [x] **2.1** Clone `isl-org/Open3D.git` → `/tmp/Open3D-build`, checkout `8e434558a`
- [x] **2.2** cmake 3.28.3 + make -j4 with `-DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF`
- [x] **2.3** Install wheel `open3d_cpu-0.18.0+8e43455-...whl` via `uv pip install` (bypassing `make install-pip-package` which dumped to ~/.local — needs venv pip on PATH)
- [x] **2.4** Verified: headless=True, `Visualizer().create_window(visible=False)`=True, rendered box 256×256 mean=0.906

⚠️ Side effect: `open3d-cpu` pins `numpy<2` — venv numpy was downgraded from 2.2.6 → 1.26.4. pyproject says `numpy>=2.0`. Monitor for breakage; may need to force-upgrade numpy after install if other packages balk.

Follow-up: `scripts/setup.sh` should gain a Phase-2 block (apt deps + cmake 3.28 download + Open3D source build + wheel install via uv pip). Deferred until SFT smoke passes.

### Phase 3 — smoke verification tests  ✅ DONE

All 5 scripts pass:

- [x] **3.1** `check_torch.py` — torch 2.5.1+cu124, A100 80GB sm_80, bf16, pytorch3d _C, flash-attn, transformers 4.50.3
- [x] **3.2** `check_open3d.py` — open3d 0.18.0+8e43455, HEADLESS=True, box rendered (256, 256, 3) mean=0.906
- [x] **3.3** `check_cadquery.py` — cadquery 2.5.0.dev0, tessellate 24v/12f, STL 684 bytes
- [x] **3.4** `check_dataset.py` — sample `0.py` → 400v/364f → rendered mean=0.979
- [x] **3.5** `check_model.py` — Qwen/Qwen2-VL-2B-Instruct on cuda:0 bf16 + flash_attention_2, forward logits (1, 103, 151936)

### Phase 4 — SFT dataset setup (route β: conservative, keep file-based loaders)

Target: `configs/sft/mix_1_2_2.yaml` — `sft_mix_weights: recode:2 text2cad:1 benchcad:2`.

**Decision**: keep existing `CadRecodeDataset` / `Text2CADDataset` as-is (file-based). Write a new `BenchCadDataset` that mirrors that API. text2cad weight → 0 for first mix smoke (no local text2cad data yet).

**Data sources discovered (2026-04-24 session):**
- `BenchCAD/cad_sft_training` (private, needs `BenchCAD_HF_TOKEN`) — parquet packaging of recode + text2cad. Used by route α only. **Route β ignores this.**
- `BenchCAD/cad_bench` (public) — 20143 rows × `{stem, family, difficulty, base_plane, feature_tags, feature_count, ops_used, gt_code, composite_png, qa_pairs, iso_tags}`. Used as the **benchcad** training source.
- `filapro/cad-recode-v1.5` (public) — per-file `.py`; already have 1673/~3700 train + 8/982 val locally.
- text2cad — upstream location unknown; **defer**.

**val set**: train.py only evals from `cad-recode-v1.5/val.pkl` (line 223-237). mix has no val contribution.

#### 4.A — recode train.pkl + val.pkl (from already-downloaded `.py`)
- [ ] **4.A.1** Download the full val split (982 files; small, no rate-limit issue)
- [ ] **4.A.2** Run `data/cadrecode2mesh.py --path data/cad-recode-v1.5 --workers 8` — `.py` → `.stl` + `train.pkl` / `val.pkl`. Partial train (1673 of ~1M) is fine for smoke.
- [ ] **4.A.3** Sanity: `pickle.load(open('data/cad-recode-v1.5/train.pkl','rb'))` shows a list of `{py_path, mesh_path}` dicts; sample `.stl` exists.

#### 4.B — benchcad materialization + loader
- [ ] **4.B.1** Write `tools/fetch_benchcad.py` — downloads `BenchCAD/cad_bench/data/test-00000-of-00001.parquet`, splits into `train/` + `val/` 90/10 by stem hash, materializes:
    - `data/benchcad/{split}/{stem}.py` — gt_code
    - `data/benchcad/{split}/{stem}_render.png` — composite_png bytes
    - `data/benchcad/{split}/{stem}.stl` — exec(gt_code) → tessellate → export (may skip failures; log count)
    - `data/benchcad/{split}.pkl` — list of `{uid: stem, py_path, mesh_path, png_path, description: family+ops_used}`
- [ ] **4.B.2** Add `BenchCadDataset` class in `dataset.py`. API: takes `(root_dir, split, mode, img_size, n_points, normalize_std_*, max_code_len)`. For img mode: load pre-rendered PNG directly (no Open3D render needed — major speedup). For pc mode: load STL, sample point cloud. Emits `{description, file_name, answer, video OR point_cloud}`.
- [ ] **4.B.3** Wire into `train.py` between line 185 (`text2cad`) and line 194 (ConcatDataset). Source key = `'benchcad'`, matching `sft_mix_weights` key.
- [ ] **4.B.4** Smoke: instantiate BenchCadDataset, fetch item 0 in pc / img / pc_img modes; confirm return shape matches what `Cadrille.forward` expects.

#### 4.C — text2cad (defer)
- Skip for this pass. In mix_1_2_2 smoke, text2cad weight will fire a `[sft_mix_weights] WARNING: ['text2cad'] not loaded` line (by design — train.py line 201-205).

### Phase 5 — SFT mix smoke  ✅ DONE (benchcad-only)

- [x] **5.1** `configs/sft/a100_mix_smoke.yaml` (benchcad-only, recode:0 benchcad:1 text2cad:0) passed end-to-end.
    - train_loss 1.22 → 0.55 over 100 steps
    - eval_loss fires at 0/50/100 (baseline 2.45 → 5.45 expected — model is learning training distribution, eval set is pc_img over different samples)
    - train_runtime 231s, samples/s 3.46
    - ckpt-final 4.2 GB saved to `checkpoints/sft-s100-lr2e-4-b4a2-pc_img-0424-0658/checkpoint-final/`
    - wandb: https://wandb.ai/hula-the-cat/cadrille-sft/runs/lvwzb165 (no entity hardcode; apikey bound to `hula-the-cat` org)

**Fixes landed during smoke:**
- `accelerate==0.34.2` → `1.3.0` (transformers 4.50.3 needs 1.1+ for `data_seed` kwarg).

Next up for actual training run:
- [ ] **5.2** 3-source mix: after recode 全量 download + text2cad 本地化, set `sft_mix_weights: {recode:2, text2cad:1, benchcad:2}` and run full `configs/sft/mix_1_2_2.yaml` (12k steps).

### Phase 6 — RL smoke (only if Phase 5 passes)

- [ ] **6.1** `bash scripts/setup.sh --data` — downloads `maksimko123/cadrille` SFT ckpt + test meshes + `data/mined/combined_hard.pkl`. (Packed as zips to avoid HF rate limit — same pattern as the smooth 87c91d3 path.)
- [ ] **6.2** `uv run python rl/train.py --config configs/rl/smoke.yaml`

## Reference — the previous "smooth" RL launch (commit 87c91d3)

Three commands after Dockerfile.official base was built:
```bash
bash scripts/setup.sh           # uv sync + 3 git packages
bash scripts/setup.sh --data    # SFT ckpt + test meshes + hard examples (zips, not per-file)
uv run python rl/train.py --config configs/rl/h100.yaml
```
Smoothness came from: (a) Dockerfile already had Open3D source-built, (b) RL started from public `maksimko123/cadrille` ckpt (no SFT training), (c) zip downloads bypassed HF rate limit.

Our target is tighter: SFT (with BenchCAD mix) BEFORE RL, so `setup.sh --data` won't cover the BenchCAD corpus — needs its own fetcher.

## Open questions (deferred until smoke passes)

1. Do we need to set `bf16: true` in `configs/sft/smoke.yaml` for A100, or override via CLI? (notes say smoke default is `false` to be safe on 4080/WSL2)
2. Should `scripts/setup.sh` gain a `--sft-data` flag that calls `data/cadrecode2mesh.py` after downloading? (see notes)
3. Slim-down of master: delete `experiments/cadevolve/`, `eval/` standalone, `viz/`, old logs — defer per `docs/repo_simplification.md`.

## Reference files

- `CLAUDE.md` — env rules (uv, wandb, no PyPI open3d, train_modality=img non-negotiable)
- `notes/sft_setup_2026-04-24.md` — previous session memo with container context
- `docs/repo_simplification.md` — parallel reorg effort, doesn't block this plan
- `install_open3d_apt.sh` — apt-install script for the Open3D headless build deps (already run)
