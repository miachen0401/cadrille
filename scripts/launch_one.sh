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
cd /home/ubuntu/cadrille || { echo "FATAL: cd /home/ubuntu/cadrille failed" >&2; exit 1; }
mkdir -p logs

CFG="${1:?usage: $0 <config.yaml>}"
[[ -f "$CFG" ]] || { echo "config not found: $CFG"; exit 1; }

set -a
[[ -f .env ]] && source .env
set +a

# DISCORD_WEBHOOK_URL — parse from .bashrc by string extraction (never `eval`
# arbitrary shell text; .bashrc lives behind the non-interactive guard).
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

# Pre-flight (skip if already done) — explicit `|| exit` so a failed
# generator stops the launcher instead of training on missing artifacts.
if [[ ! -f common/essential_ops_simple.yaml ]]; then
    echo "[pre-flight] generating bench-simple ess specs ..."
    uv run python -m data_prep.generate_simple_op_specs > /dev/null \
        || { echo "FATAL: generate_simple_op_specs failed" >&2; exit 2; }
fi
if [[ ! -f data/benchcad-simple/train_v2_holdout.pkl ]]; then
    echo "[pre-flight] building bench-simple v2 holdout pkl ..."
    uv run python -m data_prep.build_holdout_v2 > /dev/null \
        || { echo "FATAL: build_holdout_v2 failed" >&2; exit 2; }
fi

LABEL=$(basename "$CFG" .yaml)
TS=$(date -u +%Y%m%d_%H%M%S)
LOG="logs/${LABEL}_${TS}.log"

echo "[launch_one] config=$CFG label=$LABEL log=$LOG host=$(hostname)"
exec uv run python -m train.sft --config "$CFG"
