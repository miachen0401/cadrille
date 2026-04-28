#!/usr/bin/env bash
# Stream BenchCAD/benchcad-easy (109k rows, composite_png already provided) and
# repackage into Hula0401/cad-sft/benchcad-easy/ using the canonical
# image-conditioned schema (stem / code / render_img / family / difficulty /
# n_ops / ops_json / base_plane). No rendering — just stream + repack + upload.
#
# Usage:
#   set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
#   nohup bash data_prep/run_benchcad_easy_upload.sh > logs/benchcad_easy.log 2>&1 &

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

if [[ -f ~/.bashrc ]]; then
    eval "$(grep '^export DISCORD' ~/.bashrc 2>/dev/null || true)"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set"; exit 1; }

SHARD_SIZE=2000
WORKERS=6
START_SHARD=4   # resume after the 4 bellows shards already on HF
TASK_TIMEOUT=60  # seconds per render task (SIGALRM in worker)

notify() {
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        DISCORD_MSG="$*" python3 -c "
import os, json, urllib.request
url = os.environ['DISCORD_WEBHOOK_URL']
msg = os.environ['DISCORD_MSG']
req = urllib.request.Request(
    url, data=json.dumps({'content': msg}).encode(),
    headers={'Content-Type':'application/json',
             'User-Agent':'cadrille-benchcad-easy-uploader/1.0'})
try: urllib.request.urlopen(req, timeout=5).read()
except Exception as e: print(f'discord ping failed: {e}', flush=True)
" || true
    fi
}

T0=$(date +%s)
notify "🚀 benchcad-easy → Hula0401/cad-sft repackage start (workers=${WORKERS}, start_shard=${START_SHARD}, ~97k renders + 12k passthrough; ETA ~3.5h)"

if uv run python -m data_prep.import_benchcad_easy \
    --shard-size "$SHARD_SIZE" \
    --workers "$WORKERS" \
    --start-shard "$START_SHARD" \
    --per-task-timeout-sec "$TASK_TIMEOUT" \
    > logs/benchcad_easy_inner.log 2>&1; then
    DUR_MIN=$(( ($(date +%s) - T0) / 60 ))
    SHARDS=$(grep -c "uploaded in" logs/benchcad_easy_inner.log || echo "?")
    notify "✅ benchcad-easy upload DONE in ${DUR_MIN}min — ${SHARDS} shards now at https://huggingface.co/datasets/Hula0401/cad-sft/tree/main/benchcad-easy"
else
    notify "❌ benchcad-easy upload FAILED — see logs/benchcad_easy_inner.log"
    exit 1
fi
