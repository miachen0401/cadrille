#!/usr/bin/env bash
# §7 v2 sequential SFT chain — 5 lines (baseline → iid_easy → ood → ood_enhanced → iid).
#
# Each run: 50k steps × ~1.7s = ~24h on A100. Total chain ≈ 5 days.
#
# Pre-flight (idempotent — safe to re-run, runs fast if pkls already exist):
#   uv run python -m data_prep.generate_simple_op_specs   # writes common/essential_ops_simple.yaml
#   uv run python -m data_prep.build_holdout_v2           # writes data/benchcad-simple/train_v2_holdout.pkl
#
# Usage:
#   nohup bash scripts/launch_chain_v2.sh > logs/launch_chain_v2.log 2>&1 &
# To resume mid-chain (skip already-done runs):
#   START_FROM=ood_enhanced_v2 nohup bash scripts/launch_chain_v2.sh > ... &

set -uo pipefail
cd /home/ubuntu/cadrille
set -a
[[ -f .env ]] && source .env
set +a
# DISCORD_WEBHOOK_URL: extract from .bashrc (lives behind the non-interactive
# guard) without echoing the value.
if [[ -z "${DISCORD_WEBHOOK_URL:-}" ]]; then
    _line=$(grep -E "^export DISCORD_WEBHOOK_URL=" /home/ubuntu/.bashrc 2>/dev/null | head -1 || true)
    [[ -n "$_line" ]] && eval "$_line"
    unset _line
fi

START_FROM="${START_FROM:-ood_enhanced_v2}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*"; }

# Pre-flight: ensure auto-generated assets are in place.
log "=== pre-flight ==="
if [[ ! -f common/essential_ops_simple.yaml ]]; then
    log "generating bench-simple ess specs ..."
    uv run python -m data_prep.generate_simple_op_specs > /dev/null
fi
if [[ ! -f data/benchcad-simple/train_v2_holdout.pkl ]]; then
    log "building bench-simple v2 holdout pkl ..."
    uv run python -m data_prep.build_holdout_v2 > /dev/null
fi
log "pre-flight ok"

# Chain order (matches paper §7.v2 figure line ordering).
# Use a single space between label and config — multi-space breaks the
# `${entry#* }` parameter expansion below.
RUNS=(
    "ood_enhanced_v2 configs/sft/ood_enhanced_v2.yaml"
    "ood_v2 configs/sft/ood_v2.yaml"
    "iid_enhanced_v2 configs/sft/iid_enhanced_v2.yaml"
    "iid_v2 configs/sft/iid_v2.yaml"
    "baseline_v2 configs/sft/baseline_v2.yaml"
)

skip=true
[[ "$START_FROM" == "baseline_v2" ]] && skip=false

launch_run() {
    local label="$1" config="$2"
    local ts=$(date +%Y%m%d_%H%M%S)
    local logfile="logs/${label}_${ts}.log"
    log "launching ${label} with ${config} → ${logfile}"
    nohup uv run python -m train.sft --config "$config" > "$logfile" 2>&1 &
    local pid=$!
    log "${label} PID: $pid"
    sleep 10
    if ! kill -0 "$pid" 2>/dev/null; then
        log "ERROR: ${label} died within 10s, see $logfile"
        return 1
    fi
    log "${label} alive at PID $pid, waiting for it to exit ..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "${label} exited"
}

log "=== chain start (start_from=$START_FROM) ==="
for entry in "${RUNS[@]}"; do
    read -r label cfg <<<"$entry"
    if $skip; then
        if [[ "$label" == "$START_FROM" ]]; then
            skip=false
        else
            log "skipping $label (before $START_FROM)"
            continue
        fi
    fi
    launch_run "$label" "$cfg" || log "WARN: $label failed (rc=$?), continuing chain"
done

log "=== chain v2 complete ==="
