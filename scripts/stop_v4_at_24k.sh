#!/usr/bin/env bash
# Watchdog: wait for v4 ood_enhance to write checkpoint-24000, then SIGTERM.
#
# Usage:
#   nohup bash scripts/stop_v4_at_24k.sh > logs/stop_v4_at_24k.log 2>&1 &

set -uo pipefail
cd /home/ubuntu/cadrille

V4_PID="${V4_PID:-1554507}"
CKPT_ROOT="${CKPT_ROOT:-/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828}"
TARGET_STEP="${TARGET_STEP:-24000}"
TARGET_DIR="$CKPT_ROOT/checkpoint-$TARGET_STEP"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*"; }

log "watching for $TARGET_DIR/model.safetensors (PID=$V4_PID)"

# 1. Wait for the checkpoint dir to exist with a model file
while [[ ! -f "$TARGET_DIR/model.safetensors" ]]; do
    if ! kill -0 "$V4_PID" 2>/dev/null; then
        log "v4 PID exited before reaching step $TARGET_STEP — nothing to do"
        exit 0
    fi
    sleep 30
done
log "$TARGET_DIR/model.safetensors exists"

# 2. Wait until the file size is stable for 30 s (save complete)
prev_size=0
stable=0
while (( stable < 2 )); do
    size=$(stat -c%s "$TARGET_DIR/model.safetensors" 2>/dev/null || echo 0)
    if (( size == prev_size && size > 0 )); then
        ((stable++))
    else
        stable=0
    fi
    prev_size=$size
    sleep 15
done
log "checkpoint-$TARGET_STEP stable at $prev_size bytes"

# 3. SIGTERM v4 — Trainer will save trainer state cleanly on SIGTERM
log "sending SIGTERM to v4 PID $V4_PID"
kill -TERM "$V4_PID" 2>/dev/null || true
sleep 30
if kill -0 "$V4_PID" 2>/dev/null; then
    log "still alive after 30s SIGTERM, sending SIGKILL"
    kill -KILL "$V4_PID" 2>/dev/null || true
fi
log "v4 stopped"
