# Claude Code Instructions

## Workflow

- Always read `plan.md` before starting any task to understand the current goals and architecture. And update the detailed implementation in plan for tracibility.
- Track all detailed progress in `progress.md` — update it as tasks are started, completed, or blocked.

## Training Visualization

- Always use **Weights & Biases (wandb)** for training visualization. Never use TensorBoard or log-only approaches.
- Pass `report_to="wandb"` in `TrainingArguments` (for HuggingFace Trainer).
- For custom training loops (e.g. `rl_train.py`), call `wandb.log(...)` each step with at minimum: loss, reward/advantage metrics, learning rate, and step count.
- Initialize with `wandb.init(project="cadrille", ...)` at the start of each training script.
