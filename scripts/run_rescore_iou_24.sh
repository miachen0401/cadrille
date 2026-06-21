#!/usr/bin/env bash
# Drive scripts/analysis/rescore_iou_24.py on all 3 cad_bench_722 model dirs
# with Discord pings on start / per-model finish / overall finish.
#
# Usage:
#   set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
#   nohup bash scripts/run_rescore_iou_24.sh > logs/rescore_iou_24.log 2>&1 &

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs

[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set"; exit 1; }

ROOT="eval_outputs/cad_bench_722"
WORKERS=6
EARLY=0.95
TIMEOUT=240

notify() {
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        DISCORD_MSG="$*" python3 -c "
import os, json, urllib.request
url = os.environ['DISCORD_WEBHOOK_URL']
msg = os.environ['DISCORD_MSG']
data = json.dumps({'content': msg}).encode()
req = urllib.request.Request(url, data=data,
    headers={'Content-Type':'application/json',
             'User-Agent':'cadrille-iou24-rescore/1.0'})
try: urllib.request.urlopen(req, timeout=5).read()
except Exception as e: print(f'discord ping failed: {e}', flush=True)
" || true
    fi
}

T0=$(date +%s)
notify "🔄 cad_bench_722 iou-24 rescore start (workers=${WORKERS}, early-stop=${EARLY})"

uv run python scripts/analysis/rescore_iou_24.py \
    --root "$ROOT" \
    --hf-repo BenchCAD/cad_bench_722 --split train \
    --workers "$WORKERS" --early-stop "$EARLY" \
    --per-sample-timeout "$TIMEOUT" 2>&1 | tee logs/rescore_iou_24_inner.log

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    notify "❌ iou-24 rescore FAILED — see logs/rescore_iou_24_inner.log"
    exit 1
fi

# Build a Discord-friendly summary table from summary_iou_24.json
SUMMARY_JSON="$ROOT/summary_iou_24.json"
if [[ -f "$SUMMARY_JSON" ]]; then
    DC_TEXT=$(python3 -c "
import json
s = json.load(open('$SUMMARY_JSON'))
m = s['models']
lines = ['📊 **cad_bench_722 iou-24 rescore** (early-stop ${EARLY})\n']
lines.append('\`\`\`')
lines.append(f'{\"model\":<22} {\"n\":>4} {\"iou\":>7} {\"iou_24\":>7} {\"Δ\":>7} {\"rot win\":>8}')
lines.append('-' * 65)
for name, d in m.items():
    n   = d.get('n_iou_24_ok', 0)
    iou = f'{d[\"mean_iou\"]:>7.4f}'    if d.get('mean_iou')    is not None else '      —'
    i24 = f'{d[\"mean_iou_24\"]:>7.4f}' if d.get('mean_iou_24') is not None else '      —'
    dlt = f'{d[\"mean_delta\"]:>+7.4f}' if d.get('mean_delta')  is not None else '      —'
    rw  = f'{d.get(\"n_rotated_win\", 0):>4}/{n:<3}'
    lines.append(f'{name:<22} {n:>4} {iou} {i24} {dlt} {rw:>8}')
lines.append('\`\`\`')
print('\n'.join(lines))
")
    notify "$DC_TEXT"
fi

DUR_MIN=$(( ($(date +%s) - T0) / 60 ))
notify "✅ iou-24 rescore DONE in ${DUR_MIN}min — see $SUMMARY_JSON"
