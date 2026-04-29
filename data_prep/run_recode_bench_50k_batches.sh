#!/usr/bin/env bash
# Run 8 sequential 50k batches of recode-bench using
# data_prep/prepare_hf_cadrecode_v2.py. Each batch:
#   - takes a non-overlapping slice of the seed=42 shuffled cad-recode-v1.5
#     candidate list (--offset varies)
#   - writes to a unified naming scheme `train-XXXXX-of-00200.parquet` via
#     --total-shards-override 200 and a unique --start-shard
#   - pings Discord on start/finish (DISCORD_WEBHOOK_URL must be set)
#
# Layout — each batch produces 25 shards, total 200 across 8 batches → +400k:
#   Batch C: --offset 140000 --start-shard   0  → shards 000..024 of-00200
#   Batch D: --offset 190000 --start-shard  25  → shards 025..049 of-00200
#   ...
#   Batch J: --offset 490000 --start-shard 175  → shards 175..199 of-00200
#
# Cumulative recode-bench corpus after this run:
#   Phase A  20k + Phase B  80k + Phase B' 40k + this 400k = 540k
#
# Usage:
#   set -a; source .env; set +a
#   nohup bash data_prep/run_recode_bench_50k_batches.sh > logs/recode_bench_500k.log 2>&1 &
#
# Resume after crash: comment out completed batches in BATCHES below; re-run.

set -euo pipefail

# Source ~/.bashrc to pick up DISCORD_WEBHOOK_URL (interactive shells inherit
# this; nohup-launched bash scripts do not). Use a guard for non-interactive.
if [[ -f ~/.bashrc ]]; then
    # bashrc may early-return on non-interactive shells — extract just the
    # exports we need
    eval "$(grep '^export DISCORD' ~/.bashrc 2>/dev/null || true)"
fi

# Allow .env or shell to provide HF_TOKEN; bashrc provides DISCORD_WEBHOOK_URL.
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set"; exit 1; }

BATCH_SIZE=50000
SHARD_SIZE=2000
TOTAL_SHARDS=200
WORKERS=4
MAX_TASKS_PER_CHILD=100

# label  offset   start_shard
BATCHES=(
    "C 140000   0"
    "D 190000  25"
    "E 240000  50"
    "F 290000  75"
    "G 340000 100"
    "H 390000 125"
    "I 440000 150"
    "J 490000 175"
)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

notify() {
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        local msg="$*"
        # use python rather than jq to escape (jq not always installed).
        # IMPORTANT: User-Agent must be set; Discord 403s the default
        # `Python-urllib/X.Y` UA.
        DISCORD_MSG="$msg" python3 -c "
import os, json, urllib.request
url = os.environ['DISCORD_WEBHOOK_URL']
msg = os.environ['DISCORD_MSG']
data = json.dumps({'content': msg}).encode()
req = urllib.request.Request(
    url, data=data,
    headers={'Content-Type':'application/json',
             'User-Agent':'cadrille-batch-runner/1.0'})
try:
    urllib.request.urlopen(req, timeout=5).read()
except Exception as e:
    print(f'discord ping failed: {e}', flush=True)
" || true
    fi
}

T0=$(date +%s)
notify "🚀 cadrille recode-bench scale-up start: 8× 50k batches → ~540k total recode-bench"

for batch in "${BATCHES[@]}"; do
    read label offset start <<< "$batch"
    LOG="logs/batch_${label}.log"

    notify "▶️ Batch ${label}: --offset ${offset} --n ${BATCH_SIZE} --start-shard ${start}"
    BSTART=$(date +%s)

    if uv run python -m data_prep.prepare_hf_cadrecode_v2 \
        --offset "$offset" \
        --n "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --shard-size "$SHARD_SIZE" \
        --max-tasks-per-child "$MAX_TASKS_PER_CHILD" \
        --start-shard "$start" \
        --total-shards-override "$TOTAL_SHARDS" \
        > "$LOG" 2>&1; then
        DUR=$(( $(date +%s) - BSTART ))
        DURMIN=$(( DUR / 60 ))
        SUCC=$(grep "total successes" "$LOG" | tail -1 | awk -F: '{print $2}' | tr -d ' ')
        notify "✅ Batch ${label} DONE in ${DURMIN}min — ${SUCC}"
    else
        notify "❌ Batch ${label} FAILED at offset=${offset} — see ${LOG}; aborting chain"
        exit 1
    fi
done

T_END=$(date +%s)
TOTAL_MIN=$(( (T_END - T0) / 60 ))
notify "🎉 All 8 batches DONE in ${TOTAL_MIN}min — recode-bench at ~540k on Hula0401/cad-sft"
