#!/usr/bin/env bash
# Sequential launcher for §7 4-line plot SFT runs.
#
# Watches the current SFT training PID (default: ood_enhance run). When it
# exits, launches in order:
#   1. baseline 50k     (HQ only, no benchcad)
#   2. ood 50k          (holdout, no benchcad-easy)
#   3. iid 50k          (no holdout, full data)
#
# Each run ~24h. Total chain ~3 days.
#
# Usage:
#   WAIT_PID=<pid> nohup bash scripts/launch_chain_runs.sh > logs/launch_chain.log 2>&1 &

set -euo pipefail
cd /home/ubuntu/cadrille

WAIT_PID="${WAIT_PID:-1554507}"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*"
}

wait_for_pid_exit() {
    local pid=$1
    log "waiting for PID $pid to exit ..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "PID $pid exited"
}

launch_run() {
    local config=$1
    local label=$2
    local ts=$(date +%Y%m%d_%H%M%S)
    local logfile="logs/${label}_${ts}.log"
    log "launching ${label} with config ${config}"
    log "log: $logfile"
    set -a; source .env; set +a
    nohup uv run python -m train.sft --config "$config" > "$logfile" 2>&1 &
    local pid=$!
    log "${label} PID: $pid"
    sleep 10
    if ! kill -0 "$pid" 2>/dev/null; then
        log "ERROR: ${label} died within 10s, see $logfile"
        return 1
    fi
    log "${label} alive at PID $pid, monitoring ..."
    wait_for_pid_exit "$pid"
    log "${label} finished"
}

log "=== chain launcher start ==="
log "current SFT PID: $WAIT_PID"
wait_for_pid_exit "$WAIT_PID"

# Step 0: v3 IID upper-bound eval on cad_bench_722 (~30min, GPU free)
V3_CKPT="${V3_CKPT:-/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0428-1320/checkpoint-46000}"
V3_OUT="eval_outputs/v3_cad_bench_722"
v3_log="logs/v3_eval_722_$(date +%Y%m%d_%H%M%S).log"
log "running v3 cad_bench_722 eval (ckpt=$V3_CKPT) → $v3_log"
set -a; source .env; set +a
if uv run python scripts/eval_v3_cad_bench_722.py \
        --ckpt "$V3_CKPT" --out "$V3_OUT" \
        --batch-size 4 --score-workers 8 \
        > "$v3_log" 2>&1; then
    log "v3 cad_bench_722 eval done → $V3_OUT/summary.json"
else
    log "WARN: v3 cad_bench_722 eval failed (rc=$?), continuing chain anyway"
fi

# Step 0b: retro 50-OOD eval at saved v4 ood_enhance ckpts
V4_CKPT_ROOT="${V4_CKPT_ROOT:-/ephemeral/checkpoints/sft-s50k-lr2e-4-b8a4-img-0430-0828}"
V4_OUT="eval_outputs/v4_ood_retro"
v4_log="logs/v4_ood_retro_$(date +%Y%m%d_%H%M%S).log"
log "running v4 OOD retro eval over saved ckpts → $v4_log"
if uv run python scripts/eval_v4_ood_retro.py \
        --ckpts "$V4_CKPT_ROOT" --out "$V4_OUT" \
        --batch-size 4 --score-workers 8 \
        > "$v4_log" 2>&1; then
    log "v4 OOD retro eval done → $V4_OUT/summary.csv"
else
    log "WARN: v4 OOD retro eval failed (rc=$?), continuing chain anyway"
fi

launch_run configs/sft/baseline.yaml baseline
launch_run configs/sft/ood.yaml ood
launch_run configs/sft/iid.yaml iid

log "=== chain complete ==="
