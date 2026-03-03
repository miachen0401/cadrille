#!/usr/bin/env bash
# Evaluation pipeline: generate CadQuery scripts from a checkpoint, then compute metrics.
#
# Usage:
#   bash scripts/run_eval.sh --checkpoint ./checkpoints/cadrille-sft/checkpoint-final
#   bash scripts/run_eval.sh --checkpoint ./checkpoints/cadrille-rl/checkpoint-final \
#                            --split deepcad_test_mesh --mode pc
#
# Arguments (all optional, override defaults below):
#   --checkpoint PATH    Checkpoint directory or HuggingFace model ID
#   --split      NAME    Dataset split name (default: deepcad_test_mesh)
#   --mode       MODE    Input modality: pc | img | pc_img (default: pc_img)
#   --py-path    DIR     Directory to write generated .py files (default: auto)
#   --data-path  DIR     Data root directory (default: ./data)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# --- Defaults ---
CKPT="maksimko123/cadrille"
SPLIT="deepcad_test_mesh"
MODE="pc_img"
PY_PATH=""
DATA_PATH="./data"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CKPT="$2";      shift 2 ;;
        --split)      SPLIT="$2";     shift 2 ;;
        --mode)       MODE="$2";      shift 2 ;;
        --py-path)    PY_PATH="$2";   shift 2 ;;
        --data-path)  DATA_PATH="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-generate py-path from checkpoint name if not specified
if [[ -z "$PY_PATH" ]]; then
    CKPT_NAME="$(basename "$CKPT")"
    PY_PATH="./eval_outputs/${CKPT_NAME}/${SPLIT}_${MODE}"
fi

echo "Checkpoint : $CKPT"
echo "Split      : $SPLIT"
echo "Mode       : $MODE"
echo "Py output  : $PY_PATH"
echo ""

# --- Step 1: Generate CadQuery scripts ---
echo "=== Step 1: Generating CadQuery scripts ==="
python test.py \
    --checkpoint-path "$CKPT" \
    --data-path "$DATA_PATH" \
    --split "$SPLIT" \
    --mode "$MODE" \
    --py-path "$PY_PATH"

# --- Step 2: Compute metrics (IoU + CD) ---
echo ""
echo "=== Step 2: Computing metrics ==="
python evaluate.py --py-path "$PY_PATH"

echo ""
echo "Done. Results in: $PY_PATH"
