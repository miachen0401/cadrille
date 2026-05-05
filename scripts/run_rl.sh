#!/usr/bin/env bash
# One-click RL launcher.
#
# Usage:
#   scripts/run_rl.sh <short_name>
#     where <short_name> matches a configs/rl/rl_<short>.yaml file (e.g. ess_iou_v1, pure_iou).
#
# What it does:
#   1. Pre-flight: verify config exists, hard_examples_pkl exists, ckpt path exists, GPU has spare VRAM.
#   2. Reads checkpoint_path + hard_examples_pkl from the YAML so any drift is caught early.
#   3. Picks a non-conflicting tmux session name (rl_<short>) and log file (logs/rl_<short>_<ts>.log).
#   4. Launches training inside detached tmux session with `tee` so we get both terminal + log file.
#   5. Polls the log for the wandb run URL and prints it (so you can immediately open it).
#
# Notes:
#   * Single-A100 holds ONE bs=128 RL run at a time. Script aborts if VRAM < 70 GB free.
#   * Use `tmux attach -t rl_<short>` to watch live, Ctrl-b d to detach.
#   * Use `tmux kill-session -t rl_<short>` to stop.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $(basename "$0") <short_name>"
    echo "Example: $(basename "$0") pure_iou"
    echo
    echo "Available configs:"
    ls configs/rl/rl_*.yaml 2>/dev/null | sed 's|configs/rl/rl_|  |;s|\.yaml||'
    exit 1
fi

SHORT="$1"
CONFIG="configs/rl/rl_${SHORT}.yaml"
SESSION="rl_${SHORT}"
LOG="logs/rl_${SHORT}_$(date +%Y%m%d_%H%M%S).log"

cd /home/ubuntu/cadrille

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"; exit 1
fi

# ── 1. Read pool + ckpt from YAML (single source of truth) ───────────────────
POOL=$(awk -F': ' '/^hard_examples_pkl:/ {print $2; exit}' "$CONFIG" | tr -d '"')
CKPT=$(awk -F': ' '/^checkpoint_path:/  {print $2; exit}' "$CONFIG" | tr -d '"')
if [ -z "$POOL" ] || [ -z "$CKPT" ]; then
    echo "ERROR: could not parse hard_examples_pkl or checkpoint_path from $CONFIG"; exit 1
fi
[ -f "$POOL" ] || { echo "ERROR: pool not found: $POOL"; exit 1; }
[ -d "$CKPT" ] || { echo "ERROR: ckpt dir not found: $CKPT"; exit 1; }
COMPGEN=$(compgen -G "$CKPT/model*.safetensors" || true)
[ -n "$COMPGEN" ] || { echo "ERROR: no weight file in $CKPT"; exit 1; }

# ── 2. GPU + RAM pre-flight ──────────────────────────────────────────────────
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$GPU_FREE" -lt 70000 ]; then
    echo "ERROR: only ${GPU_FREE} MiB GPU free — bs=128 needs >=70 GB. Stop the running job first:"
    pgrep -af "python.*train\.rl" 2>/dev/null | grep -v "$0" | head
    exit 1
fi
RAM_AVAIL=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
[ "$RAM_AVAIL" -ge 30 ] || { echo "ERROR: only ${RAM_AVAIL} GB RAM avail (need >=30)"; exit 1; }

# ── 3. Tmux session conflict check ──────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION"
    echo "  or kill it: tmux kill-session -t $SESSION"
    exit 1
fi

mkdir -p logs

# ── 4. Print the plan + launch ───────────────────────────────────────────────
RUN_NAME=$(awk -F': ' '/^run_name:/ {print $2; exit}' "$CONFIG" | tr -d '"')
echo "=== launching ==="
echo "  config:    $CONFIG"
echo "  ckpt:      $CKPT"
echo "  pool:      $POOL"
echo "  run_name:  $RUN_NAME"
echo "  tmux:      $SESSION"
echo "  log file:  $LOG"
echo "  GPU free:  ${GPU_FREE} MiB | RAM avail: ${RAM_AVAIL} GB"
echo

tmux new-session -d -s "$SESSION" \
    "uv run python -m train.rl.train --config $CONFIG 2>&1 | tee $LOG"

echo "tmux session started. Waiting for wandb URL..."

# ── 5. Poll for wandb URL ────────────────────────────────────────────────────
WANDB_URL=""
for _ in $(seq 1 60); do
    if [ -f "$LOG" ]; then
        WANDB_URL=$(grep -oE 'https://wandb\.ai/[^ ]*runs/[a-z0-9]+' "$LOG" | head -1 || true)
        if [ -n "$WANDB_URL" ]; then break; fi
    fi
    sleep 2
done

if [ -n "$WANDB_URL" ]; then
    echo "wandb run: $WANDB_URL"
else
    echo "(wandb URL not seen in 2 min — check the log: tail -f $LOG)"
fi

echo
echo "=== controls ==="
echo "  watch live :   tmux attach -t $SESSION    (Ctrl-b d to detach)"
echo "  tail log :     tail -f $LOG"
echo "  stop :         tmux kill-session -t $SESSION"
