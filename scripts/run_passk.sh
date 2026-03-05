#!/usr/bin/env bash
# Pass@k evaluation for a single checkpoint or a full checkpoint sweep.
#
# Usage examples
# ──────────────
# 1. Single checkpoint, default settings:
#    bash scripts/run_passk.sh --checkpoint ./checkpoints/cadrille-rl/checkpoint-10000
#
# 2. Sweep all checkpoint-XXXXX dirs in a run, log to existing W&B run:
#    bash scripts/run_passk.sh \
#        --checkpoint-sweep ./checkpoints/cadrille-rl \
#        --wandb-project cadrille-rl \
#        --wandb-run-id   <run-id-from-training>
#
# 3. RTX 4080 (16 GB) — smaller batch fits in 16 GB:
#    bash scripts/run_passk.sh \
#        --checkpoint ./checkpoints/cadrille-rl/checkpoint-10000 \
#        --eval-batch-size 4
#
# Arguments (all optional, override defaults below):
#   --checkpoint PATH        Single checkpoint dir or HF model ID
#   --checkpoint-sweep DIR   Directory with checkpoint-XXXXX subdirs
#   --val-dir DIR            GT mesh directory (default: ./data/deepcad_test_mesh)
#   --n-examples N           Val examples to use (default: 50)
#   --n-samples  N           Samples per example (default: 16)
#   --k-values   LIST        Comma-separated k values (default: 1,2,4,8)
#   --threshold  F           IoU threshold for "correct" (default: 0.5)
#   --temperature F          Sampling temperature (default: 1.0)
#   --sequential             Force sequential generation (saves VRAM)
#   --output-dir DIR         Where to write JSON results
#   --wandb-project NAME     W&B project (omit to skip)
#   --wandb-run-id  ID       Resume existing W&B run
#   --wandb-offline          Use W&B offline mode
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────
CHECKPOINT=""
CHECKPOINT_SWEEP=""
VAL_DIR="./data/deepcad_test_mesh"
N_EXAMPLES=50
N_SAMPLES=5
K_VALUES="1,5"
THRESHOLD=0.5
TEMPERATURE=1.0
SEQUENTIAL=""
EVAL_BATCH_SIZE=8
OUTPUT_DIR="./eval_outputs/passk"
WANDB_PROJECT=""
WANDB_RUN_ID=""
WANDB_OFFLINE=""

# ── Argument parsing ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)        CHECKPOINT="$2";       shift 2 ;;
        --checkpoint-sweep)  CHECKPOINT_SWEEP="$2"; shift 2 ;;
        --val-dir)           VAL_DIR="$2";           shift 2 ;;
        --n-examples)        N_EXAMPLES="$2";        shift 2 ;;
        --n-samples)         N_SAMPLES="$2";         shift 2 ;;
        --k-values)          K_VALUES="$2";          shift 2 ;;
        --threshold)         THRESHOLD="$2";         shift 2 ;;
        --temperature)       TEMPERATURE="$2";       shift 2 ;;
        --sequential)        SEQUENTIAL="--sequential"; shift ;;
        --eval-batch-size)   EVAL_BATCH_SIZE="$2";   shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2";        shift 2 ;;
        --wandb-project)     WANDB_PROJECT="$2";     shift 2 ;;
        --wandb-run-id)      WANDB_RUN_ID="$2";      shift 2 ;;
        --wandb-offline)     WANDB_OFFLINE="--wandb-offline"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" && -z "$CHECKPOINT_SWEEP" ]]; then
    echo "ERROR: provide --checkpoint or --checkpoint-sweep"
    exit 1
fi

# ── Build python command ─────────────────────────────────────────────────────
CMD=(python rl/eval_passk.py
    --val-dir        "$VAL_DIR"
    --n-examples     "$N_EXAMPLES"
    --n-samples      "$N_SAMPLES"
    --k-values       "$K_VALUES"
    --threshold      "$THRESHOLD"
    --temperature    "$TEMPERATURE"
    --max-new-tokens  1000
    --reward-workers  8
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --output-dir      "$OUTPUT_DIR"
)

if [[ -n "$CHECKPOINT" ]];       then CMD+=(--checkpoint       "$CHECKPOINT");       fi
if [[ -n "$CHECKPOINT_SWEEP" ]]; then CMD+=(--checkpoint-sweep "$CHECKPOINT_SWEEP"); fi
if [[ -n "$SEQUENTIAL" ]];       then CMD+=($SEQUENTIAL);                             fi
if [[ -n "$WANDB_PROJECT" ]];    then CMD+=(--wandb-project    "$WANDB_PROJECT");    fi
if [[ -n "$WANDB_RUN_ID" ]];     then CMD+=(--wandb-run-id     "$WANDB_RUN_ID");     fi
if [[ -n "$WANDB_OFFLINE" ]];    then CMD+=($WANDB_OFFLINE);                          fi

# ── Print config and run ────────────────────────────────────────────────────
echo "Pass@k evaluation"
echo "  checkpoint  : ${CHECKPOINT:-$CHECKPOINT_SWEEP (sweep)}"
echo "  val dir     : $VAL_DIR"
echo "  n_examples  : $N_EXAMPLES"
echo "  n_samples   : $N_SAMPLES"
echo "  k values    : $K_VALUES"
echo "  threshold   : $THRESHOLD"
echo "  temperature : $TEMPERATURE"
echo "  output dir  : $OUTPUT_DIR"
[[ -n "$WANDB_PROJECT" ]] && echo "  wandb       : $WANDB_PROJECT"
echo ""

exec "${CMD[@]}"
