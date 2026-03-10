# Claude Code Instructions

## Environment

- Always use **uv** for Python package management. Never use `pip install` or `conda install` directly.
- Run scripts with `uv run python ...` or activate the uv environment. The base conda env at `/opt/conda` is read-only.
- Install packages with `uv pip install` or declare them in `pyproject.toml` and run `uv sync`.

## Workflow

- Always read `plan.md` before starting any task to understand the current goals and architecture. And update the detailed implementation to-do list and task queries to `plan.md` for tracibility.
- Track all detailed progress in `progress.md` — update it as tasks are started, completed, or blocked.

## Resource Monitoring & Job Safety

**Before starting ANY job (training, conversion, eval, large script):**
1. Check resources: `free -h`, `df -h /workspace`, `nvidia-smi`, CPU via `top -bn1 | head -5`
2. Never start a job if any resource is near capacity:
   - RAM: leave ≥ 3 GB free (system has 15 GB total)
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
- The EGL/CUDA conflict on 4080 is solved by pre-rendering PNG cache before model loading (`pre_render_img_cache` in `rl/train.py`). Do not revert this.

## Training Visualization

- Always use **Weights & Biases (wandb)** for training visualization. Never use TensorBoard or log-only approaches.
- Pass `report_to="wandb"` in `TrainingArguments` (for HuggingFace Trainer).
- For custom training loops (e.g. `rl_train.py`), call `wandb.log(...)` each step with at minimum: loss, reward/advantage metrics, learning rate, and step count.
- Initialize with `wandb.init(project="cadrille", ...)` at the start of each training script.

## Repository Structure

Keep the repo clean and structured — think of it as a Meta Research codebase.

**Canonical layout:**
```
cadrille.py          # model definition (single file)
train.py             # SFT entry point
evaluate.py          # paper's reference metric script (do not modify)
test.py              # paper's reference inference script (do not modify)
dataset.py           # shared dataset utilities
rl/                  # RL fine-tuning
  train.py           # RL training entry point
  algorithms/        # CPPO and other RL algorithms
  dataset.py         # RL-specific data loading
  reward.py          # reward computation
  eval.py            # training-time eval
  eval_passk.py      # pass@k eval
configs/             # YAML configs (one per GPU tier: a100, h100, 4080, smoke)
tools/               # reusable CLI scripts (each with --help docstring)
data/                # datasets (gitignored large files)
checkpoints/         # model checkpoints (gitignored)
docs/                # design documents and paper notes
```

**Rules:**
- **Always add new dataset paths to `.gitignore` before downloading data.** Datasets are never committed.
- No debug scripts in the repo root or rl/. One-off scripts → delete after use.
- If a script is worth keeping, put it in `tools/` with a full argparse docstring and add it to `tools/README.md`.
- No scratch notebooks committed (use `colab.ipynb` only for the public demo).
- `plan.md` and `progress.md` track current work; `eval_results.md` tracks paper vs. ours metrics.
- Prefer flat module structure: add to an existing file before creating a new one.
