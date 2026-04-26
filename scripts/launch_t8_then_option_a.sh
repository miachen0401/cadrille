#!/usr/bin/env bash
# Pre-staged orchestration: when curriculum finishes (step 20000) →
# T8 eval on final ckpt → launch Option A (qwen3vl_2b_recode_30k_clean).
# Run as: bash scripts/launch_t8_then_option_a.sh
set -euo pipefail

CURR_DIR="/ephemeral/checkpoints/sft-s20k-lr2e-4-b8a4-img-0425-1929"
EVAL_OUT="eval_outputs/t8_curriculum_final"
LOG_T8="/tmp/t8_eval.log"
LOG_OA="/tmp/option_a_train.log"

set -a; source /home/ubuntu/cadrille/.env; set +a

# 1. Wait for curriculum to actually finish if not done yet
while pgrep -f "train.sft.*curriculum" > /dev/null; do
  echo "[wait] curriculum still running ..."
  sleep 30
done
echo "[wait] curriculum process gone"

# Pick final ckpt (prefer 20000)
FINAL_CKPT="$CURR_DIR/checkpoint-20000"
if [ ! -d "$FINAL_CKPT" ]; then
  FINAL_CKPT=$(ls -d "$CURR_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
fi

# Best historical ckpts (pulled from HF since save_total_limit=4 evicted them)
BENCHCAD_PEAK="/ephemeral/checkpoints/curriculum_best_from_hf/checkpoint-11000"
DEEPCAD_PEAK="/ephemeral/checkpoints/curriculum_best_from_hf/checkpoint-15000"

# 2. Run T8 eval on the 3 ckpts: BenchCAD peak (11k), DeepCAD peak (15k), final (20k)
for CKPT_DIR in "$BENCHCAD_PEAK" "$DEEPCAD_PEAK" "$FINAL_CKPT"; do
  if [ ! -d "$CKPT_DIR" ]; then
    echo "[T8] skipping missing ckpt: $CKPT_DIR"
    continue
  fi
  STEP=$(basename "$CKPT_DIR" | grep -oP '\d+')
  echo "[T8] eval ckpt-$STEP @ $CKPT_DIR"
  # T8 fast greedy-only sweep — directional check, not exhaustive.
  # online_eval already gave us max_iou@8; we just want greedy IoU on each ckpt
  # to confirm where the trajectory peaks.
  if uv run python -u -m eval.bench_sweep \
      --ckpt "$CKPT_DIR" \
      --base-model Qwen/Qwen3-VL-2B-Instruct \
      --backbone qwen3_vl \
      --datasets benchcad,deepcad,fusion360 \
      --temps 0 \
      --n-samples 1 \
      --limit 30 \
      --img-size 268 \
      --batch-size 4 \
      --out "$EVAL_OUT/step_$STEP" \
      --label "curriculum_step_$STEP" 2>&1 | tee -a "$LOG_T8"; then
    echo "[T8] ckpt-$STEP done OK"
  else
    echo "[T8] ckpt-$STEP FAILED — continuing with next ckpt"
  fi
done
echo "[T8] all evals done — results at $EVAL_OUT/"

# 3. Wait for render to finish + rebuild train.pkl on the freshly-rendered corpus
echo "[render] waiting for fetch_cadrecode_full to finish ..."
while pgrep -f "fetch_cadrecode_full.*phase render" > /dev/null; do
  sleep 30
done
echo "[render] done; rebuilding pkl manifest"
uv run python -m data_prep.fetch_cadrecode_full --phase pkl \
    --out /ephemeral/data/cad-recode-v1.5 2>&1 | tail -5

# Sanity: confirm new corpus is loadable + has > 50k items
N_TRAIN=$(uv run python -c "
import pickle
with open('/ephemeral/data/cad-recode-v1.5/train.pkl','rb') as f:
    print(len(pickle.load(f)))
" 2>&1 | tail -1)
echo "[render] cad-recode-v1.5/train.pkl rows=$N_TRAIN"
if [ "$N_TRAIN" -lt 50000 ]; then
  echo "WARNING: only $N_TRAIN rendered, expected ~100k. Continuing anyway."
fi

# 4. Launch Option A run in background
echo "[Option A] launching qwen3vl_2b_recode_30k_clean.yaml ..."
nohup uv run python -u -m train.sft \
    --config configs/sft/qwen3vl_2b_recode_30k_clean.yaml \
    > "$LOG_OA" 2>&1 &
OA_PID=$!
echo "[Option A] PID=$OA_PID  log=$LOG_OA"
echo "$OA_PID" > /tmp/option_a.pid
