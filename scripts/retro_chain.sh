#!/usr/bin/env bash
# Sequential retro 50-OOD eval chain — v3 (23 ckpts) → ood_enhance missing
# 8 ckpts (2k–16k). Run after the local 4-ckpt ood_enhance retro finishes.
#
# Usage:
#   nohup bash scripts/retro_chain.sh > logs/retro_chain.log 2>&1 &
#
# Watches: WAIT_PID env var (default 0 = no wait).

set -uo pipefail
cd /home/ubuntu/cadrille
# Source .env normally (set -a marks all assignments for export, then unset).
# Avoids the previous bug where `export $(grep ...)` fed grep's "file:KEY=val"
# output to export verbatim (leaking values into stderr/log).
set -a
[[ -f .env ]] && source .env
set +a
# DISCORD_WEBHOOK_URL lives in ~/.bashrc behind the non-interactive guard,
# so just extract it via parameter expansion (does not echo the value).
if [[ -z "${DISCORD_WEBHOOK_URL:-}" ]]; then
    _line=$(grep -E "^export DISCORD_WEBHOOK_URL=" /home/ubuntu/.bashrc | head -1 || true)
    if [[ -n "$_line" ]]; then
        eval "$_line"
    fi
    unset _line
fi

WAIT_PID="${WAIT_PID:-0}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S)] $*"; }

if [[ "$WAIT_PID" -gt 0 ]]; then
    log "waiting for PID $WAIT_PID to exit ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    log "PID $WAIT_PID exited"
fi

# Step 1: v3 — 23 ckpts (2k-46k)
log "=== v3 retro (23 ckpts, ~2h) ==="
v3_log="logs/retro_v3_$(date -u +%Y%m%d_%H%M%S).log"
if uv run python scripts/eval_retro_hf.py \
        --repo Hula0401/cadrille-qwen3vl-2b-v3-clean-50k \
        --steps all \
        --out eval_outputs/v3_ood_retro_hf \
        --batch-size 4 --score-workers 8 \
        > "$v3_log" 2>&1; then
    log "v3 retro done → eval_outputs/v3_ood_retro_hf/summary.csv"
else
    log "WARN: v3 retro failed (rc=$?), continuing"
fi

# Step 2: ood_enhance — 8 missing ckpts (local already has 18k/20k/22k/24k)
log "=== ood_enhance retro (missing 8 ckpts: 2k-16k, ~40min) ==="
oe_log="logs/retro_ood_enhance_$(date -u +%Y%m%d_%H%M%S).log"
if uv run python scripts/eval_retro_hf.py \
        --repo Hula0401/cadrille-qwen3vl-2b-v4-holdout-50k \
        --steps 2000,4000,6000,8000,10000,12000,14000,16000 \
        --out eval_outputs/v4_ood_retro \
        --batch-size 4 --score-workers 8 \
        > "$oe_log" 2>&1; then
    log "ood_enhance retro done"
else
    log "WARN: ood_enhance retro failed (rc=$?)"
fi

log "=== retro chain complete ==="
