#!/usr/bin/env bash
# Periodic monitor for the BenchCAD/benchcad-easy → Hula0401/cad-sft upload.
# Every $INTERVAL seconds (default 1800 = 30 min):
#   - Asks HF how many `benchcad-easy/*.parquet` shards are now live
#   - Tails the latest batch log for current rate / ETA / error count
#   - Pings $DISCORD_WEBHOOK_URL with a one-block status
# Exits cleanly once all 55 shards are present on HF.
#
# Usage:
#   set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
#   nohup bash data_prep/monitor_benchcad_easy.sh > logs/benchcad_easy_monitor.log 2>&1 &

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

if [[ -f ~/.bashrc ]]; then
    eval "$(grep '^export DISCORD' ~/.bashrc 2>/dev/null || true)"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set"; exit 1; }

INTERVAL=${MONITOR_INTERVAL:-1800}   # seconds; default 30 min
TOTAL_SHARDS=55
PREV_DONE=-1
T_START=$(date +%s)

notify() {
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        DISCORD_MSG="$*" python3 -c "
import os, json, urllib.request
url = os.environ['DISCORD_WEBHOOK_URL']
msg = os.environ['DISCORD_MSG']
req = urllib.request.Request(
    url, data=json.dumps({'content': msg}).encode(),
    headers={'Content-Type':'application/json',
             'User-Agent':'cadrille-benchcad-easy-monitor/1.0'})
try: urllib.request.urlopen(req, timeout=5).read()
except Exception as e: print(f'discord ping failed: {e}', flush=True)
" || true
    fi
}

# Number of shards currently on HF.
# Uses `uv run python` because huggingface_hub lives in the project's uv venv,
# not the system python3. Falls back to "?" so the caller still pings.
hf_shard_count() {
    HF_TOKEN="$HF_TOKEN" uv run --quiet python -c "
from huggingface_hub import HfApi
import os
api = HfApi()
files = api.list_repo_files('Hula0401/cad-sft', repo_type='dataset',
                            token=os.environ['HF_TOKEN'])
print(sum(1 for f in files
          if f.startswith('benchcad-easy/') and f.endswith('.parquet')))
" 2>/dev/null || echo "?"
}

# Pull latest progress line from the most recently-touched batch log
batch_status() {
    LATEST=$(ls -t logs/benchcad_easy_batch_*.log 2>/dev/null | head -1)
    if [[ -z "$LATEST" ]]; then
        echo "no batch log yet"
        return
    fi
    LABEL=$(basename "$LATEST" .log | sed 's/benchcad_easy_batch_//')
    LAST_PROG=$(grep -E "^\s*\[[0-9]+/[0-9]+\]" "$LATEST" | tail -1 | sed -E 's/^\s+//')
    LAST_SHARD=$(grep -E "^\s*shard [0-9]+/[0-9]+" "$LATEST" | tail -1 | sed -E 's/^\s+//')
    if [[ -z "$LAST_PROG" && -z "$LAST_SHARD" ]]; then
        echo "batch=${LABEL} (warming up)"
    else
        echo "batch=${LABEL} | ${LAST_PROG:-no progress yet} | ${LAST_SHARD:-no shard yet}"
    fi
}

# Check if the batch driver process is still alive
driver_alive() {
    pgrep -f "run_benchcad_easy_batches.sh" > /dev/null
}

notify "👁️  benchcad-easy monitor up — pinging every $((INTERVAL/60)) min until all ${TOTAL_SHARDS} shards land"

while true; do
    DONE=$(hf_shard_count)
    NOW=$(date +%s)
    ELAPSED_MIN=$(( (NOW - T_START) / 60 ))
    STAT=$(batch_status)

    if [[ "$DONE" == "?" ]]; then
        notify "⚠️ benchcad-easy monitor (t+${ELAPSED_MIN}min): HF list call failed; will retry next tick. ${STAT}"
    else
        DRIVER_STATE="alive ✓"
        driver_alive || DRIVER_STATE="DEAD ❌"
        DELTA=$((PREV_DONE >= 0 ? DONE - PREV_DONE : 0))
        notify "📈 benchcad-easy update (t+${ELAPSED_MIN}min) — **${DONE}/${TOTAL_SHARDS}** shards on HF (Δ=+${DELTA} since last tick) | driver: ${DRIVER_STATE} | ${STAT}"
        PREV_DONE=$DONE
        if [[ "$DONE" -ge "$TOTAL_SHARDS" ]]; then
            notify "🎉 benchcad-easy monitor: all ${TOTAL_SHARDS} shards present on HF after ${ELAPSED_MIN}min — exiting."
            exit 0
        fi
    fi

    sleep "$INTERVAL"
done
