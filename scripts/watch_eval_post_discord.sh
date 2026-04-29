#!/usr/bin/env bash
# Watch the predictions/ dir for new step-NNNNNN.jsonl files, and fire
# scripts/analysis/eval_to_discord.py for each new step. Idempotent via
# {step}.posted marker. Designed to run alongside training.
#
# Usage:
#   nohup bash scripts/watch_eval_post_discord.sh \
#       /home/ubuntu/cadrille/logs/big_bench_shell_50k_20260427_061353.log \
#       /ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0427-0613 \
#       > logs/watch_post_discord.log 2>&1 &

set -euo pipefail
LOG_PATH="${1:?usage: $0 <training-log> <ckpt-output-dir> [poll-secs]}"
OUT_DIR="${2:?usage: $0 <training-log> <ckpt-output-dir> [poll-secs]}"
POLL="${3:-60}"

PRED_DIR="$OUT_DIR/predictions"
WORKERS="${WORKERS:-4}"

# Load Discord webhook (handles non-interactive bashrc early-return)
eval "$(grep '^export DISCORD_WEBHOOK_URL' ~/.bashrc)"
[ -n "${DISCORD_WEBHOOK_URL:-}" ] || { echo "DISCORD_WEBHOOK_URL not set"; exit 2; }

cd /home/ubuntu/cadrille

echo "[watch] log=$LOG_PATH"
echo "[watch] out_dir=$OUT_DIR"
echo "[watch] poll=${POLL}s  workers=$WORKERS"

while true; do
    if [ -d "$PRED_DIR" ]; then
        for f in "$PRED_DIR"/step-*.jsonl; do
            [ -e "$f" ] || continue
            base=$(basename "$f" .jsonl)             # step-001000
            step_str="${base#step-}"                  # 001000
            step=$((10#$step_str))                    # 1000
            [ "$step" -eq 0 ] && continue             # skip eval_on_start
            marker="$PRED_DIR/${base}.posted"
            [ -e "$marker" ] && continue
            # Check that the eval block is actually present in the log
            if ! grep -q "step=${step} running IoU eval" "$LOG_PATH" 2>/dev/null; then
                continue
            fi
            # Wait until all 5 buckets are logged for this step (otherwise IoU
            # for the last bucket might be missing). Heuristic: count "[img/" +
            # "[text/" lines after the marker.
            block=$(awk -v m="step=${step} running IoU eval" '
                $0 ~ m {found=1}
                found {print}
            ' "$LOG_PATH")
            # Threshold: 4 buckets (text2cad legacy deleted; only BC val + recode20k train +
            # DC test + FU test now reliably show. text2cad train counts but if 0 items it's skipped).
            n_done=$(echo "$block" | grep -cE '\[(img|text)/(BenchCAD val|recode20k train|text2cad train|DeepCAD test|Fusion360 test)\]' || true)
            if [ "$n_done" -lt 4 ]; then
                echo "[watch] step=$step only $n_done/4 buckets logged, waiting"
                continue
            fi
            # max_iou@8 cycle: fires at step 1k, 3k, 5k, 7k, ... (odd thousand).
            # When applicable, wait for all 3 max_iou@8 bucket lines too so the
            # Discord post includes both greedy + max@8.
            half_step=$((step / 1000))
            if [ $((half_step % 2)) -eq 1 ]; then
                n_max=$(echo "$block" | grep -cE '\] max_iou@8 \(t=[0-9.]+\)=' || true)
                if [ "$n_max" -lt 3 ]; then
                    echo "[watch] step=$step max_iou@8 only $n_max/3 buckets logged, waiting"
                    continue
                fi
            fi

            echo "[watch] $(date -u +%H:%M:%S) firing eval_to_discord for step=$step"
            if uv run python -m scripts.analysis.eval_to_discord \
                    --step "$step" \
                    --log "$LOG_PATH" \
                    --output-dir "$OUT_DIR" \
                    --workers "$WORKERS"; then
                touch "$marker"
                echo "[watch] step=$step posted, marker=$marker"
            else
                echo "[watch] step=$step FAILED, will retry next poll"
            fi
        done
    fi
    sleep "$POLL"
done
