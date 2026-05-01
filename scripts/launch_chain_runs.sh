#!/usr/bin/env bash
# Sequential launcher for §7 4-line plot runs.
#
# Watches the current v4-holdout training PID. When it exits, launches:
#   1. v4-hq-only 50k        (line 4: no-bench)
#   2. v4-holdout-noeasy 50k (line 2: OOD plain)
#   3. v4-baseline 50k       (control)
#
# Each step ~24h. Total chain ~3 days.
#
# Usage:
#   nohup bash scripts/launch_chain_runs.sh > logs/launch_chain.log 2>&1 &

set -euo pipefail
cd /home/ubuntu/cadrille

WAIT_PID="${WAIT_PID:-1554507}"  # current v4-holdout PID
WORKERS=4

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
log "current v4-holdout PID: $WAIT_PID"
wait_for_pid_exit "$WAIT_PID"

# Run 1: v4-hq-only (no bench-stack)
launch_run \
    configs/sft/big_bench_shell_50k_v4_hq_only.yaml \
    v4_hq_only

# Run 2: v4-holdout-noeasy
launch_run \
    configs/sft/big_bench_shell_50k_v4_holdout_noeasy.yaml \
    v4_holdout_noeasy

# Run 3: v4-baseline (control: same recipe, no holdout)
launch_run \
    configs/sft/big_bench_shell_50k_v4_baseline.yaml \
    v4_baseline

log "=== chain complete ==="
