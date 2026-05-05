#!/bin/bash
# Wrapper for systemd auto-restart: resume from latest RL checkpoint if any,
# else fall back to the SFT init checkpoint declared in the YAML.
#
# Usage (from systemd-run):
#   scripts/rl_resume_wrapper.sh <config_yaml> <run_output_dir>
#
# Why: train.rl.train's checkpoint_path is fixed in the YAML (SFT init).
# On crash, systemd Restart=on-failure would re-launch from step 0, losing
# all RL progress. This wrapper detects the highest-numbered checkpoint-*
# in <run_output_dir> and passes it via --checkpoint-path.
#
# train.rl.config's start_step regex picks up the trailing integer in
# `checkpoint-N`, so resume_step is correct.
set -euo pipefail

CFG="${1:?config yaml required}"
OUT_DIR="${2:?run output dir required}"

cd /home/ubuntu/cadrille
set -a; source .env; set +a

LATEST=""
if [ -d "$OUT_DIR" ]; then
    # Pick the highest-numbered checkpoint-* dir that has model weights.
    for ckpt in $(ls -1d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -V -r); do
        if compgen -G "$ckpt/model*.safetensors" > /dev/null \
                || compgen -G "$ckpt/pytorch_model*.bin" > /dev/null; then
            LATEST="$ckpt"
            break
        fi
    done
fi

if [ -n "$LATEST" ]; then
    echo "[resume_wrapper] resuming from $LATEST" | tee -a logs/rl_resume.log
    exec /home/ubuntu/.local/bin/uv run python -m train.rl.train \
        --config "$CFG" --checkpoint-path "$LATEST"
else
    echo "[resume_wrapper] no RL checkpoint found, using SFT init from $CFG" \
        | tee -a logs/rl_resume.log
    exec /home/ubuntu/.local/bin/uv run python -m train.rl.train --config "$CFG"
fi
