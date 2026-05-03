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
cd /home/ubuntu/cadrille || { echo "FATAL: cd /home/ubuntu/cadrille failed" >&2; exit 1; }
mkdir -p logs
set -a
[[ -f .env ]] && source .env
set +a
# DISCORD_WEBHOOK_URL: parse from .bashrc (lives behind the non-interactive
# guard) by string extraction — never `eval` arbitrary shell text.
if [[ -z "${DISCORD_WEBHOOK_URL:-}" ]]; then
    _line=$(grep -E "^export DISCORD_WEBHOOK_URL=" /home/ubuntu/.bashrc 2>/dev/null | head -1 || true)
    if [[ -n "$_line" ]]; then
        _val=${_line#*=}
        _val=${_val#\"}; _val=${_val%\"}
        _val=${_val#\'}; _val=${_val%\'}
        export DISCORD_WEBHOOK_URL="$_val"
        unset _val
    fi
    unset _line
fi

START_FROM="${START_FROM:-ood_enhanced_v2}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*"; }

# Resource preflight — gate the chain on RAM/disk/GPU/CPU per CLAUDE.md.
# Each run is ~24h × 5 = ~5 days; aborting before launch is much cheaper
# than discovering a low-resource crash mid-chain.
assert_resources() {
    local ram_gb disk_gb gpu_free_mib cpu_idle
    ram_gb=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
    disk_gb=$(df -BG --output=avail /ephemeral 2>/dev/null | tail -1 | tr -d ' G' || echo 0)
    gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    cpu_idle=$(top -bn1 | awk '/Cpu\(s\)/ {print $8}' | head -1)
    log "resources: RAM=${ram_gb}GB free, disk=${disk_gb}GB free on /ephemeral, GPU=${gpu_free_mib}MiB free, CPU idle=${cpu_idle}%"
    awk -v ram="$ram_gb" 'BEGIN{exit !(ram >= 1.0)}' || { log "FATAL: <1GB RAM free"; return 1; }
    [[ "$disk_gb" -ge 100 ]] || { log "FATAL: <100GB free on /ephemeral (${disk_gb}GB)"; return 1; }
    [[ "$gpu_free_mib" -ge 1024 ]] || { log "FATAL: <1GB GPU free (${gpu_free_mib}MiB)"; return 1; }
    return 0
}
log "=== resource preflight ==="
if ! assert_resources; then
    log "abort — resource preflight failed"
    exit 1
fi

# Pre-flight: ensure auto-generated assets are in place.
# `set -e` is off so we explicitly `|| exit` to fail-fast on prep errors.
log "=== pre-flight ==="
if [[ ! -f common/essential_ops_simple.yaml ]]; then
    log "generating bench-simple ess specs ..."
    uv run python -m data_prep.generate_simple_op_specs > /dev/null \
        || { log "FATAL: generate_simple_op_specs failed"; exit 2; }
fi
if [[ ! -f data/benchcad-simple/train_v2_holdout.pkl ]]; then
    log "building bench-simple v2 holdout pkl ..."
    uv run python -m data_prep.build_holdout_v2 > /dev/null \
        || { log "FATAL: build_holdout_v2 failed"; exit 2; }
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
        # Surface the actual exit status — `wait` after the process is gone
        # returns the captured rc.
        wait "$pid" 2>/dev/null
        return $?
    fi
    log "${label} alive at PID $pid, waiting for it to exit ..."
    # Use `wait` (not poll-via-kill) so we capture the child's exit status.
    # `wait` blocks until the bg job finishes and returns its rc.
    wait "$pid"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "ERROR: ${label} exited with rc=$rc — see $logfile"
        return $rc
    fi
    log "${label} exited cleanly (rc=0)"
    return 0
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
