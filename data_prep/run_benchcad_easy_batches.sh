#!/usr/bin/env bash
# Drive BenchCAD/benchcad-easy → Hula0401/cad-sft/benchcad-easy upload as
# 5 chained batches × 10 shards each (last batch is 9). Each batch:
#   - calls data_prep/import_benchcad_easy.py with explicit --start-shard /
#     --end-shard
#   - lands its 10 (×2000-row) parquet shards on HF before the next batch
#     starts, so a kill at any point loses at most one in-flight shard
#     (~2 minutes of render work)
#   - pings Discord with a one-line summary on start / finish
#
# The importer auto-detects already-uploaded shards on HF so explicit
# --start-shard becomes documentation rather than mandatory state. Re-running
# this script is safe: each batch will skip work that's already on HF.
#
# Usage:
#   set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
#   nohup bash data_prep/run_benchcad_easy_batches.sh > logs/benchcad_easy_batches.log 2>&1 &
#
# Resume after crash: rerun the script. Auto-resume picks up where it left.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

if [[ -f ~/.bashrc ]]; then
    eval "$(grep '^export DISCORD' ~/.bashrc 2>/dev/null || true)"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set"; exit 1; }

WORKERS=6
SHARD_SIZE=2000
TASK_TIMEOUT=60   # seconds per render task (SIGALRM)

# label  end-shard-cap (exclusive)   approx_rows-this-batch
# Each batch lets the importer auto-detect start_shard from HF (subject to
# the global VM-side cap below), so if a previous batch aborted early
# (e.g. RAM floor) the next batch picks up the missing shards instead of
# skipping them.
#
# This driver runs on VM1 and is capped at shard 32 (end=33).
# Shards 33..54 are owned by VM2 — must NOT be auto-detected here, so we
# must NEVER set an end-shard ≥ 33 when a parallel VM is active.
BATCHES=(
    "A   24"   # shards 15..23   (9 shards if starting fresh from 15)
    "B   33"   # shards 24..32   (9 shards)
)

notify() {
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        DISCORD_MSG="$*" python3 -c "
import os, json, urllib.request
url = os.environ['DISCORD_WEBHOOK_URL']
msg = os.environ['DISCORD_MSG']
req = urllib.request.Request(
    url, data=json.dumps({'content': msg}).encode(),
    headers={'Content-Type':'application/json',
             'User-Agent':'cadrille-benchcad-easy-batches/1.0'})
try: urllib.request.urlopen(req, timeout=5).read()
except Exception as e: print(f'discord ping failed: {e}', flush=True)
" || true
    fi
}

T0=$(date +%s)
notify "🚀 benchcad-easy batched upload start (5 batches, workers=${WORKERS}, ${SHARD_SIZE} rows/shard)"

for batch in "${BATCHES[@]}"; do
    read label end <<< "$batch"
    LOG="logs/benchcad_easy_batch_${label}.log"

    # Auto-detect start_shard from HF inside the importer (no --start-shard
    # passed). End-shard caps this batch. If everything below `end` is already
    # done, the importer prints "rows-to-process: 0" and exits fast.
    notify "▶️ Batch ${label}: process up through shard $((end-1)) (auto-detect start)"
    BSTART=$(date +%s)

    if uv run python -m data_prep.import_benchcad_easy \
        --end-shard "$end" \
        --workers "$WORKERS" \
        --shard-size "$SHARD_SIZE" \
        --per-task-timeout-sec "$TASK_TIMEOUT" \
        > "$LOG" 2>&1; then
        DUR=$(( ($(date +%s) - BSTART) / 60 ))
        N_SHARDS=$(grep -c "uploaded in" "$LOG" || echo "0")
        N_ERR=$(grep -oE "render_errors=[0-9]+" "$LOG" | tail -1 || echo "render_errors=?")
        notify "✅ Batch ${label} DONE in ${DUR}min — ${N_SHARDS} shards, ${N_ERR}"
    else
        notify "❌ Batch ${label} FAILED (end-shard=${end}) — see ${LOG}; aborting chain"
        exit 1
    fi
done

TOTAL_MIN=$(( ($(date +%s) - T0) / 60 ))
notify "🎉 benchcad-easy ALL batches DONE in ${TOTAL_MIN}min — https://huggingface.co/datasets/Hula0401/cad-sft/tree/main/benchcad-easy"
