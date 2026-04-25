# Claude Code Instructions

## Environment

- Always use **uv** for Python package management. Never use `pip install` or `conda install` directly.
- Run scripts with `uv run python ...` or activate the uv environment. The base conda env at `/opt/conda` is read-only.
- Install packages with `uv pip install` or declare them in `pyproject.toml` and run `uv sync`.
- 永远在回复和思考的时候说中文或者英语 不要说日语和韩语 

## Workflow

- Always read `plan.md` before starting any task to understand the current goals and architecture. And update the detailed implementation to-do list and task queries to `plan.md` for tracibility.
- Track all detailed progress in `progress.md` — update it as tasks are started, completed, or blocked.

## Resource Monitoring & Job Safety

**Before starting ANY job (training, conversion, eval, large script):**
1. Check resources: `free -h`, `df -h /workspace`, `nvidia-smi`, CPU via `top -bn1 | head -5`
2. Never start a job if any resource is near capacity:
   - RAM: leave ≥ 1 GB free (system has 15 GB total)
   - Disk: leave ≥ 100 GB free on /workspace (1 TB total)
   - GPU VRAM: leave ≥ 1 GB free (4080 SUPER has 16 GB)
   - CPU: do not pin all 12 cores; leave ≥ 2 cores free
3. **Always ask the user before starting** any long-running or resource-intensive job:
   - RL training runs (hours–days)
   - Dataset conversions (>10k files)
   - Full eval sweeps (>100 samples)
   - Any `nohup` / background job
4. Never restart a killed job without user confirmation.
5. Monitor running jobs every few minutes and report status proactively rather than waiting for the user to ask.

## RL Training Modality — Non-Negotiable Rules

- **`train_modality` must always be `img`** in all RL configs (4080, a100, h100, smoke). Never switch to `pc` mode to work around rendering issues — fix the rendering instead.
- **`val_modalities` must always include `img`** (e.g. `pc,img`). Never drop img from validation.
- Rendering uses the custom conda open3d (`0.18.0+8e434558a`) via `Visualizer(visible=False)` — no EGL or Xvfb needed. The PyPI open3d requires EGL and **must not be installed** (`pip install open3d` breaks rendering).
- Run `data_prep/prerender_dataset.py` to pre-render STLs to `{stem}_render.png` PNGs; `common.meshio.render_img` loads the PNG if it exists, otherwise falls back to on-the-fly rendering.

## Training Visualization

- Always use **Weights & Biases (wandb)** for training visualization. Never use TensorBoard or log-only approaches.
- Pass `report_to="wandb"` in `TrainingArguments` (for HuggingFace Trainer).
- For custom training loops (e.g. `train/rl/train.py`), call `wandb.log(...)` each step with at minimum: loss, reward/advantage metrics, learning rate, and step count.
- Initialize with `wandb.init(project="cadrille", ...)` at the start of each training script.

## Repository Structure

Keep the repo clean and structured — think of it as a Meta Research codebase.

**Canonical layout:**
```
common/              # shared across train, eval, scripts
  model.py           # Cadrille (Qwen2-VL-2B + FourierPointEncoder) + collate
  datasets.py        # CadRecode / Text2CAD / BenchCad / CadRecode20k loaders
  meshio.py          # render_img, MeshDataset
  metrics.py         # compute_metrics, compute_reward, worker pools

train/
  sft/
    train.py         # SFT entry  (python -m train.sft)
    online_eval.py   # IoU + Failures TrainerCallback
    hf_uploader.py   # background ckpt push to HF model repo
  rl/
    train.py         # RL entry  (python -m train.rl.train)
    algorithms/      # CPPO, DPO
    dataset.py       # RLDataset, CurriculumRLDataset, DPODataset
    eval.py eval_passk.py  (eval_passk re-exported via eval/passk.py)
    config.py mine.py filter_scores.py

eval/
  pipeline.py runner.py config.py render.py report.py
  passk.py           # pass@k runner + CLI (python -m eval.passk)
  bench.py bench_visualize.py   # BenchCAD benchmark eval
  others/            # paper-original: evaluate.py + test.py (do not modify)

data_prep/           # one-time dataset preparation
bench/               # training throughput benchmarks
experiments/
  cadevolve/         # off-main CAD-evolve experiment
  repair_lora/       # repair-LoRA mini-experiment
  data_prep_cadlib/  # DeepCAD/Fusion360 mesh gen (needs cadlib, not in pyproject)

configs/             # YAML configs (one per GPU tier: a100, h100, 4080, smoke)
scripts/             # python utilities (no shell wrappers — invoke `python -m …` directly)
  check_env/         # post-install env verification (torch, open3d, model, …)
  analysis/          # one-off research analyses (plot_kl_quadrants, analyze_*,
                     #   mining_analysis, render_*_grid, failure_analysis, …)
                     #   parse_cq.py is the local helper library in here
tests/               # test_refactor_safety + test_iou + test_pipeline + test_cppo_step
data/                # datasets (gitignored large files)
checkpoints/         # model checkpoints (gitignored)
```

**Rules:**
- **Always add new dataset paths to `.gitignore` before downloading data.** Datasets are never committed.
- No debug scripts in the repo root or `train/rl/`. One-off scripts → delete after use.
- Putting a new script somewhere? `data_prep/` if it prepares data once, `bench/` if it times training, `experiments/` if it is an off-main-path investigation, `tools/` if it analyzes trained models. Full argparse docstring required; add to `tools/README.md` if going there.
- No scratch notebooks committed.
- `plan.md` tracks current work.
- Prefer flat module structure: add to an existing file before creating a new one.
