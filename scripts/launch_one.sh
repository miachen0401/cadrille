#!/usr/bin/env bash
# Single-config launcher for multi-HPC parallel §7 v2 study.
#
# Each HPC claims one config and runs it to convergence (50k step ≈ 24h).
# Training auto-pushes ckpts + predictions/ to HF (per the config's
# hf_upload_repo). Analysis happens elsewhere by pulling from HF.
#
# Usage:
#   nohup bash scripts/launch_one.sh configs/sft/baseline_v2.yaml > logs/baseline_v2.log 2>&1 &
#
# Pre-flight (idempotent):
#   bash scripts/setup.sh --data
#   uv run python -m data_prep.generate_simple_op_specs
#   uv run python -m data_prep.build_holdout_v2

set -uo pipefail
cd /home/ubuntu/cadrille

CFG="${1:?usage: $0 <config.yaml>}"
[[ -f "$CFG" ]] || { echo "config not found: $CFG"; exit 1; }

set -a
[[ -f .env ]] && source .env
set +a

# DISCORD_WEBHOOK_URL — extract from .bashrc (lives behind non-interactive guard).
if [[ -z "${DISCORD_WEBHOOK_URL:-}" ]]; then
    _line=$(grep -E "^export DISCORD_WEBHOOK_URL=" /home/ubuntu/.bashrc 2>/dev/null | head -1 || true)
    [[ -n "$_line" ]] && eval "$_line"
    unset _line
fi

# Pre-flight (skip if already done)
if [[ ! -f common/essential_ops_simple.yaml ]]; then
    echo "[pre-flight] generating bench-simple ess specs ..."
    uv run python -m data_prep.generate_simple_op_specs > /dev/null
fi
if [[ ! -f data/benchcad-simple/train_v2_holdout.pkl ]]; then
    echo "[pre-flight] building bench-simple v2 holdout pkl ..."
    uv run python -m data_prep.build_holdout_v2 > /dev/null
fi

LABEL=$(basename "$CFG" .yaml)
TS=$(date -u +%Y%m%d_%H%M%S)
LOG="logs/${LABEL}_${TS}.log"

echo "[launch_one] config=$CFG label=$LABEL log=$LOG host=$(hostname)"
exec uv run python -m train.sft --config "$CFG"
