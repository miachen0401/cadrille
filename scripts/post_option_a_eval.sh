#!/usr/bin/env bash
# Post-session eval — fires near the end of the autonomous 8h budget to
# produce numbers for the summary doc. Eval whatever Option A ckpt is
# currently the most recent (might not be 20000 if budget ran out).
#
# Run as: bash scripts/post_option_a_eval.sh
# Optionally: add a deadline check so we don't run if Option A is still
# 30+ min from done.
set -euo pipefail

set -a; source /home/ubuntu/cadrille/.env; set +a

EVAL_OUT="eval_outputs/option_a_final"
LOG="/tmp/option_a_post_eval.log"
RUN_DIRS_PARENT="/ephemeral/checkpoints"

# Find the Option A run dir (newest sft-* dir from this session)
RUN_DIR=$(ls -dt "$RUN_DIRS_PARENT"/sft-* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ] || [[ "$RUN_DIR" == *curriculum* ]] || [[ "$RUN_DIR" == *0425* ]]; then
  echo "[post-eval] no Option A run dir found, exiting"
  exit 0
fi
echo "[post-eval] Option A dir: $RUN_DIR"

# Most recent ckpt
CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT" ]; then
  echo "[post-eval] no ckpt found in $RUN_DIR yet, exiting"
  exit 0
fi
STEP=$(basename "$CKPT" | grep -oP '\d+')
echo "[post-eval] eval ckpt-$STEP @ $CKPT"

uv run python -u -m eval.bench_sweep \
    --ckpt "$CKPT" \
    --base-model Qwen/Qwen3-VL-2B-Instruct \
    --backbone qwen3_vl \
    --datasets benchcad,deepcad,fusion360 \
    --temps 0 \
    --n-samples 1 \
    --limit 30 \
    --img-size 268 \
    --batch-size 4 \
    --out "$EVAL_OUT/step_$STEP" \
    --label "option_a_step_$STEP" 2>&1 | tee "$LOG"

echo "[post-eval] done; results at $EVAL_OUT/step_$STEP"
