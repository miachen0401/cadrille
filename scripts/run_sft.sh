#!/usr/bin/env bash
# 1-click SFT launcher for multi-GPU (torchrun) or single-GPU (python).
#
# Usage:
#   bash scripts/run_sft.sh                          # fresh run, auto config
#   bash scripts/run_sft.sh --resume                 # resume latest checkpoint in run dir
#   bash scripts/run_sft.sh --config configs/sft/h100.yaml
#   bash scripts/run_sft.sh --run-name my-run-v2
#   N_GPUS=1 bash scripts/run_sft.sh                 # force single-GPU
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---------- GPU count ----------
N_GPUS="${N_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}"
echo "GPUs detected: $N_GPUS"

# ---------- Default config by GPU count ----------
if [[ -z "${CONFIG:-}" ]]; then
    if [[ "$N_GPUS" -ge 8 ]]; then
        CONFIG="configs/sft/h100.yaml"
    else
        CONFIG="configs/sft/full.yaml"
    fi
fi
echo "Config: $CONFIG"

# ---------- Resume detection ----------
# Pass --resume to auto-detect and continue from the latest checkpoint.
# The run output dir is derived from the config the same way train.py does it.
RESUME_FLAG=""
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--resume" ]]; then
        RESUME_FLAG="--resume-from-checkpoint latest"
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# ---------- Launch ----------
if [[ "$N_GPUS" -gt 1 ]]; then
    echo "Launching with torchrun (${N_GPUS} GPUs)..."
    torchrun \
        --nproc_per_node="$N_GPUS" \
        --master_port="${MASTER_PORT:-29500}" \
        train.py --config "$CONFIG" $RESUME_FLAG "${EXTRA_ARGS[@]}"
else
    echo "Launching single-GPU..."
    python train.py --config "$CONFIG" $RESUME_FLAG "${EXTRA_ARGS[@]}"
fi
