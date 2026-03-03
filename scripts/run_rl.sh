#!/usr/bin/env bash
# 1-click RL (Dr. CPPO) launcher.
#
# RL uses a single model process for training. Generation of G=16 rollouts
# is the bottleneck; on 8 H100s we exploit this by sharding the dataset
# across GPUs (each GPU trains independently on its shard, then checkpoints
# are averaged — "model soup" / ensemble, or just use the best-reward ckpt).
#
# For true DDP RL (shared gradients), set DISTRIBUTED=1 — requires
# the model to fit within one GPU's memory (2B model = fine on H100).
#
# Usage:
#   bash scripts/run_rl.sh                           # single process, auto config
#   bash scripts/run_rl.sh --checkpoint-path ./checkpoints/cadrille-sft/checkpoint-final
#   DISTRIBUTED=1 bash scripts/run_rl.sh             # 8 independent shards
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

N_GPUS="${N_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}"
CONFIG="${CONFIG:-configs/rl-s50k-lr3e-5-G16-cppo.yaml}"
DISTRIBUTED="${DISTRIBUTED:-0}"

echo "GPUs detected : $N_GPUS"
echo "Config        : $CONFIG"

if [[ "$DISTRIBUTED" == "1" && "$N_GPUS" -gt 1 ]]; then
    echo "Launching $N_GPUS independent RL shards (one per GPU)..."
    for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
        SHARD_NAME="shard${GPU_ID}"
        echo "  Starting shard $GPU_ID → --run-name suffix -${SHARD_NAME}"
        CUDA_VISIBLE_DEVICES="$GPU_ID" python rl_train.py \
            --config "$CONFIG" \
            --run-name "$(date +%m%d-%H%M)-rl-${SHARD_NAME}" \
            "$@" &
    done
    echo "All $N_GPUS shards launched. Wait with: wait"
    wait
else
    echo "Launching single RL process..."
    python rl_train.py --config "$CONFIG" "$@"
fi
