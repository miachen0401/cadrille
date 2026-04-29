#!/usr/bin/env bash
# Evaluate three baselines on BenchCAD/cad_bench_722 (720 samples, single train split):
#   1. cadrille-rl   (official filapro/cadrille RL ckpt, point-cloud + image)
#   2. cadevolve-rl1 (official kulibinai/cadevolve RL ckpt, 8-view image input)
#   3. Qwen2.5-VL-3B-Instruct (zero-shot, off-the-shelf VLM, single composite_png)
#
# Per-sample records → eval_outputs/cad_bench_722/<model>/metadata.jsonl
# Final aggregated summary → eval_outputs/cad_bench_722/summary.json,
# also pinged to Discord (DISCORD_WEBHOOK_URL).
#
# Usage:
#   set -a; source .env; eval "$(grep '^export DISCORD' ~/.bashrc)"; set +a
#   nohup bash scripts/eval_cad_bench_722.sh > logs/eval_cad_bench_722.log 2>&1 &

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs eval_outputs/cad_bench_722

[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set"; exit 1; }

OUT_ROOT="eval_outputs/cad_bench_722"
HF_REPO="BenchCAD/cad_bench_722"
SCORE_WORKERS=4
ATTN=sdpa  # flash-attn not installed in this venv

notify() {
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        DISCORD_MSG="$*" python3 -c "
import os, json, urllib.request
url = os.environ['DISCORD_WEBHOOK_URL']
msg = os.environ['DISCORD_MSG']
data = json.dumps({'content': msg}).encode()
req = urllib.request.Request(url, data=data,
    headers={'Content-Type':'application/json',
             'User-Agent':'cadrille-bench-runner/1.0'})
try: urllib.request.urlopen(req, timeout=5).read()
except Exception as e: print(f'discord ping failed: {e}', flush=True)
" || true
    fi
}

T0=$(date +%s)
notify "🚀 cad_bench_722 eval start: cadrille-rl + cadevolve-rl1 + Qwen2.5-VL-3B-zs (720 samples each)"

# -------- 1. Cadrille-RL ---------------------------------------------------
M1_OUT="$OUT_ROOT/cadrille_rl"
notify "▶️ [1/3] cadrille-rl on cad_bench_722 ..."
T1=$(date +%s)
uv run python -m eval.bench \
    --ckpt checkpoints/cadrille-rl \
    --model-type cadrille \
    --hf-repo "$HF_REPO" \
    --split train \
    --batch-size 4 \
    --score-workers "$SCORE_WORKERS" \
    --attn-impl "$ATTN" \
    --out "$M1_OUT" \
    --label cadrille-rl > logs/eval_cadrille_rl.log 2>&1 || {
        notify "❌ cadrille-rl FAILED — see logs/eval_cadrille_rl.log"; exit 1; }
S1=$(python3 -c "import json,sys; r=[json.loads(l) for l in open('$M1_OUT/metadata.jsonl')]; ok=[x for x in r if x.get('error_type')=='success']; ious=[x['iou'] for x in ok if x.get('iou') is not None]; cds=[x['cd'] for x in ok if x.get('cd') is not None]; print(f'n={len(r)} exec={len(ok)/len(r)*100:.1f}% iou={sum(ious)/len(ious):.4f if ious else 0:.4f} cd={sum(cds)/len(cds):.6f}' if ious and cds else f'n={len(r)} exec={len(ok)/len(r)*100:.1f}%')" 2>/dev/null || echo "summary parse failed")
DUR=$(( ($(date +%s) - T1) / 60 ))
notify "✅ [1/3] cadrille-rl DONE in ${DUR}min — $S1"

# -------- 2. CADEvolve-RL (official 8-view) -------------------------------
M2_OUT="$OUT_ROOT/cadevolve_rl1"
notify "▶️ [2/3] cadevolve-rl1 (8-view official) on cad_bench_722 ..."
T2=$(date +%s)
uv run python -m experiments.cadevolve.eval \
    --ckpt checkpoints/cadevolve-rl1 \
    --hf-repo "$HF_REPO" \
    --split train \
    --batch-size 2 \
    --score-workers "$SCORE_WORKERS" \
    --attn-impl "$ATTN" \
    --out "$M2_OUT" \
    --label cadevolve-rl1 > logs/eval_cadevolve_rl1.log 2>&1 || {
        notify "❌ cadevolve-rl1 FAILED — see logs/eval_cadevolve_rl1.log"; exit 1; }
S2=$(python3 -c "import json,sys; r=[json.loads(l) for l in open('$M2_OUT/metadata.jsonl')]; ok=[x for x in r if x.get('error_type')=='success']; ious=[x['iou'] for x in ok if x.get('iou') is not None]; cds=[x['cd'] for x in ok if x.get('cd') is not None]; print(f'n={len(r)} exec={len(ok)/len(r)*100:.1f}% iou={sum(ious)/len(ious):.4f} cd={sum(cds)/len(cds):.6f}' if ious and cds else f'n={len(r)} exec={len(ok)/len(r)*100:.1f}%')" 2>/dev/null || echo "summary parse failed")
DUR=$(( ($(date +%s) - T2) / 60 ))
notify "✅ [2/3] cadevolve-rl1 DONE in ${DUR}min — $S2"

# -------- 3. Qwen2.5-VL-3B zero-shot --------------------------------------
M3_OUT="$OUT_ROOT/qwen25vl_3b_zs"
notify "▶️ [3/3] Qwen2.5-VL-3B-Instruct (zero-shot) on cad_bench_722 ..."
T3=$(date +%s)
uv run python -m eval.bench \
    --model-type qwen25vl_zs \
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \
    --hf-repo "$HF_REPO" \
    --split train \
    --batch-size 2 \
    --score-workers "$SCORE_WORKERS" \
    --attn-impl "$ATTN" \
    --max-new-tokens 1024 \
    --out "$M3_OUT" \
    --label qwen25vl-3b-zs > logs/eval_qwen25vl_3b_zs.log 2>&1 || {
        notify "❌ qwen2.5-vl-3b zs FAILED — see logs/eval_qwen25vl_3b_zs.log"; exit 1; }
S3=$(python3 -c "import json,sys; r=[json.loads(l) for l in open('$M3_OUT/metadata.jsonl')]; ok=[x for x in r if x.get('error_type')=='success']; ious=[x['iou'] for x in ok if x.get('iou') is not None]; cds=[x['cd'] for x in ok if x.get('cd') is not None]; print(f'n={len(r)} exec={len(ok)/len(r)*100:.1f}% iou={sum(ious)/len(ious):.4f} cd={sum(cds)/len(cds):.6f}' if ious and cds else f'n={len(r)} exec={len(ok)/len(r)*100:.1f}%')" 2>/dev/null || echo "summary parse failed")
DUR=$(( ($(date +%s) - T3) / 60 ))
notify "✅ [3/3] qwen2.5-vl-3b-zs DONE in ${DUR}min — $S3"

# -------- Aggregate ---------------------------------------------------------
SUMMARY_JSON="$OUT_ROOT/summary.json"
uv run python scripts/analysis/aggregate_cad_bench_722.py \
    --root "$OUT_ROOT" \
    --out "$SUMMARY_JSON" \
    --discord 2>&1 | tee logs/aggregate_cad_bench_722.log || {
        notify "❌ aggregate FAILED — see logs/aggregate_cad_bench_722.log"; exit 1; }

T_END=$(date +%s)
TOTAL_MIN=$(( (T_END - T0) / 60 ))
notify "🎉 cad_bench_722 eval ALL DONE in ${TOTAL_MIN}min. Summary at $SUMMARY_JSON"
