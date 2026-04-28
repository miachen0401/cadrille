#!/usr/bin/env bash
# Wait for v2 to hit step 30000 + eval done + ckpt-30000 saved, then swap to v3.
#
# Triggers:
#   - /ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0427-0908/checkpoint-30000/model.safetensors exists
#   - log has "step=30000 running IoU eval" + 5 bucket result lines after it
#
# Then:
#   1. SIGINT all training processes (graceful shutdown)
#   2. Kill old watcher
#   3. Launch v3 training (configs/sft/big_bench_shell_50k_v3.yaml)
#   4. Start new Discord watcher
#   5. Post Discord notification

set -uo pipefail

V2_LOG=/home/ubuntu/cadrille/logs/big_bench_shell_50k_phase2b_20260427_184015.log
V2_OUT=/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0427-0908
TARGET_CKPT="$V2_OUT/checkpoint-30000"
SCRIPT_LOG=/home/ubuntu/cadrille/logs/swap_v2_to_v3.log

exec >> "$SCRIPT_LOG" 2>&1
echo "=== swap_v2_to_v3 started at $(date -u) ==="

# Source .env for HF_TOKEN, WANDB_API_KEY, DISCORD_WEBHOOK_URL
set -a; source /home/ubuntu/cadrille/.env; set +a

# 1. Wait for ckpt-30000 model.safetensors to exist
echo "Waiting for $TARGET_CKPT/model.safetensors ..."
while [ ! -f "$TARGET_CKPT/model.safetensors" ]; do
    sleep 60
done
echo "[$(date -u +%H:%M:%S)] ckpt-30000 model.safetensors exists"

# 2. Wait for eval block at step 30000 to have all 5 bucket lines
echo "Waiting for step=30000 eval block to be complete..."
while true; do
    n_buckets=$(awk '/step=30000 running IoU eval/{found=1} found{print}' "$V2_LOG" 2>/dev/null \
        | grep -cE '\[(img|text)/(BenchCAD val|recode20k train|text2cad train|DeepCAD test|Fusion360 test)\]' || echo 0)
    if [ "$n_buckets" -ge 5 ]; then
        echo "[$(date -u +%H:%M:%S)] step=30000 eval complete ($n_buckets buckets logged)"
        break
    fi
    sleep 60
done

# Wait an extra 30 sec to let HF upload thread queue (best-effort)
sleep 30

# 3. Discord pre-stop notification
curl -sX POST "$DISCORD_WEBHOOK_URL" -H 'Content-Type: application/json' \
    -d '{"content":"🛑 **swap_v2_to_v3**: v2 reached step 30 000, stopping now and launching v3 (cleaned data, 5-source mix)"}'

# 4. SIGINT v2 training
echo "[$(date -u +%H:%M:%S)] SIGINT v2 training"
pkill -INT -f 'train.sft.*phase2.yaml' || true
sleep 30
# Verify dead
if pgrep -f 'train.sft.*phase2.yaml' > /dev/null; then
    echo "v2 still alive after 30s, sending SIGTERM"
    pkill -TERM -f 'train.sft.*phase2.yaml' || true
    sleep 15
fi

# 5. Kill old watcher
pkill -f 'watch_eval_post_discord' || true
sleep 2

# 6. Launch v3
TS=$(date -u +%Y%m%d_%H%M%S)
V3_LOG=/home/ubuntu/cadrille/logs/v3_clean_${TS}.log
echo "[$(date -u +%H:%M:%S)] launching v3 → $V3_LOG"
cd /home/ubuntu/cadrille
nohup bash -c '
    set -a; source /home/ubuntu/cadrille/.env; set +a
    cd /home/ubuntu/cadrille
    uv run python -u -m train.sft --config configs/sft/big_bench_shell_50k_v3.yaml
' > "$V3_LOG" 2>&1 &
disown

# Wait for v3 to spin up + extract output_dir from log
sleep 45
V3_OUT=$(grep -m1 -oP 'Output\s+:\s+\K[^ ]+' "$V3_LOG" || echo "")
if [ -z "$V3_OUT" ]; then
    echo "WARN: could not parse v3 output_dir from $V3_LOG; watcher will fail"
else
    echo "[$(date -u +%H:%M:%S)] v3 output_dir = $V3_OUT"
fi

# 7. Start new watcher
if [ -n "$V3_OUT" ]; then
    nohup bash /home/ubuntu/cadrille/scripts/watch_eval_post_discord.sh \
        "$V3_LOG" "$V3_OUT" \
        > /home/ubuntu/cadrille/logs/watch_post_discord_v3.log 2>&1 &
    disown
    echo "[$(date -u +%H:%M:%S)] watcher started for v3"
fi

# 8. Discord post-launch notification
curl -sX POST "$DISCORD_WEBHOOK_URL" -H 'Content-Type: application/json' \
    -d "{\"content\":\"🚀 **v3 launched** — clean data (736 k items, 5 sources), 60% HQ mix. Log: \`$V3_LOG\`. Output: \`$V3_OUT\`. Discord auto-reports each eval as before.\"}"

echo "=== swap_v2_to_v3 done at $(date -u) ==="
