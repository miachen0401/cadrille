#!/usr/bin/env bash
# Run all 4 baselines × {DeepCAD, Fusion360} (300 sampled each) through
# eval/bench_stl.py. Each (model, dataset) writes its own metadata.jsonl.
#
# Usage:
#   set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
#   nohup bash scripts/run_stl_eval_all.sh > logs/stl_eval_all.log 2>&1 &

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs/stl

if [[ -f ~/.bashrc ]]; then
    eval "$(grep '^export DISCORD' ~/.bashrc 2>/dev/null || true)"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set"; exit 1; }

N=300
SEED=42
ATTN=sdpa

notify() {
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        DISCORD_MSG="$*" python3 -c "
import os, json, urllib.request
url = os.environ['DISCORD_WEBHOOK_URL']
msg = os.environ['DISCORD_MSG']
req = urllib.request.Request(url, data=json.dumps({'content': msg}).encode(),
    headers={'Content-Type':'application/json',
             'User-Agent':'cad-stl-eval/1.0'})
try: urllib.request.urlopen(req, timeout=5).read()
except Exception as e: print(f'discord ping failed: {e}', flush=True)
" || true
    fi
}

# label  model_type  ckpt  backbone  base_model
RUNS=(
    "cadrille_rl       cadrille  checkpoints/cadrille-rl                                   qwen2_vl Qwen/Qwen2-VL-2B-Instruct"
    "cadevolve_rl1     cadevolve checkpoints/cadevolve-rl1                                  qwen2_vl Qwen/Qwen2-VL-2B-Instruct"
    "qwen25vl_3b_zs    qwen25vl_zs none                                                    qwen2_vl Qwen/Qwen2.5-VL-3B-Instruct"
    "cadrille_qwen3vl_v3 cadrille checkpoints/cadrille-qwen3vl-v3-clean-50k/checkpoint-34000 qwen3_vl Qwen/Qwen2-VL-2B-Instruct"
)

DATASETS=(
    "deepcad   data/deepcad_test_mesh"
    "fusion360 data/fusion360_test_mesh"
)

T0=$(date +%s)
notify "🚀 STL eval start: 4 models × {deepcad, fusion360} × n=${N}"

for ds_line in "${DATASETS[@]}"; do
    read DS_NAME DS_DIR <<< "$ds_line"
    OUT_ROOT="eval_outputs/${DS_NAME}_n${N}"
    for run in "${RUNS[@]}"; do
        read LABEL MTYPE CKPT BACKBONE BASE_MODEL <<< "$run"
        OUT_DIR="$OUT_ROOT/$LABEL"
        LOG="logs/stl/${DS_NAME}_${LABEL}.log"
        BSTART=$(date +%s)
        notify "▶️ $DS_NAME / $LABEL …"

        CKPT_ARG=""
        [[ "$CKPT" != "none" ]] && CKPT_ARG="--ckpt $CKPT"

        if uv run python -m eval.bench_stl \
            $CKPT_ARG --model-type "$MTYPE" --backbone "$BACKBONE" \
            --base-model "$BASE_MODEL" \
            --stl-dir "$DS_DIR" --n-samples "$N" --seed "$SEED" \
            --batch-size 4 --score-workers 4 --attn-impl "$ATTN" \
            --out "$OUT_DIR" --label "${LABEL}-${DS_NAME}-n${N}" \
            > "$LOG" 2>&1; then
            DUR=$(( ($(date +%s) - BSTART) / 60 ))
            SUMMARY=$(grep -A4 "^Summary:" "$LOG" | tail -4 | tr -d '\n' | tr -s ' ')
            notify "✅ $DS_NAME / $LABEL DONE in ${DUR}min — $SUMMARY"
        else
            notify "❌ $DS_NAME / $LABEL FAILED — see $LOG"
        fi
    done
done

DUR=$(( ($(date +%s) - T0) / 60 ))
notify "🎉 STL eval ALL DONE in ${DUR}min"
