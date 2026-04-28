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

# label  start  end (exclusive)   approx_rows
BATCHES=(
    "A   6   16"   # shards 06..15  → 20k rows
    "B  16   26"   # shards 16..25  → 20k rows
    "C  26   36"   # shards 26..35  → 20k rows
    "D  36   46"   # shards 36..45  → 20k rows
    "E  46   55"   # shards 46..54  → ~18k + tail (final batch)
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
    read label start end <<< "$batch"
    LOG="logs/benchcad_easy_batch_${label}.log"

    # Auto-resume protection: if all shards in this batch are already on HF,
    # the importer will print "rows-to-process: 0" and exit fast.
    notify "▶️ Batch ${label}: shards ${start}..$((end-1))"
    BSTART=$(date +%s)

    if uv run python -m data_prep.import_benchcad_easy \
        --start-shard "$start" \
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
        notify "❌ Batch ${label} FAILED at shards ${start}..$((end-1)) — see ${LOG}; aborting chain"
        exit 1
    fi
done

TOTAL_MIN=$(( ($(date +%s) - T0) / 60 ))
notify "🎉 benchcad-easy ALL batches DONE in ${TOTAL_MIN}min — https://huggingface.co/datasets/Hula0401/cad-sft/tree/main/benchcad-easy"
