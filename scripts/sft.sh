#!/usr/bin/env bash
# One-liner SFT launcher. Preps smoke data the first time, then trains.
#
# Usage:
#   bash scripts/sft.sh                                   # smoke (600 steps, ~25 min on 4080)
#   bash scripts/sft.sh --config configs/sft/default.yaml # full (12k steps) — needs full meshes
#   bash scripts/sft.sh --config configs/sft/mix_1_2_2.yaml
#
# First run auto-generates STL meshes + train.pkl in data/cad-recode-v1.5/ from
# the 982-file val/ subset (~5 min). Subsequent runs skip prep.
#
# W&B: project `cadrille-sft` (set WANDB_API_KEY in .env).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Load secrets (HF_TOKEN, WANDB_API_KEY) if .env exists
[[ -f .env ]] && set -a && source .env && set +a

CONFIG="configs/sft/smoke.yaml"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *)        EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "[sft] config: $CONFIG"

# ─── Prep data if missing ────────────────────────────────────────────────
if [[ ! -f data/cad-recode-v1.5/train.pkl ]]; then
    if [[ ! -d data/cad-recode-v1.5/val ]]; then
        echo "[sft] ERROR: data/cad-recode-v1.5/ not found. Download first:"
        echo "      GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/filapro/cad-recode-v1.5 data/cad-recode-v1.5"
        echo "      cd data/cad-recode-v1.5 && git lfs pull && cd ../.."
        exit 1
    fi
    echo "[sft] train.pkl missing — generating from val/ subset (~5 min) ..."
    uv run python scripts/prep_smoke_data.py
fi

# ─── Train ───────────────────────────────────────────────────────────────
N_GPUS="$(uv run python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
echo "[sft] GPUs: $N_GPUS"

if [[ "$N_GPUS" -gt 1 ]]; then
    uv run torchrun --nproc_per_node="$N_GPUS" \
        train.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
else
    uv run python train.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
fi
