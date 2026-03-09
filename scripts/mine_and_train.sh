#!/usr/bin/env bash
# Sequential mining + training pipeline.
# Run AFTER DeepCAD mining (PID 46532) is already running.
# This script: waits for DeepCAD mining to finish, then mines Fusion360,
# merges pkls, and restarts training on mined data.
#
# Usage: bash scripts/mine_and_train.sh [MINE_PID]
# Default MINE_PID: 46532 (DeepCAD mining PID)

set -euo pipefail
cd "$(dirname "$0")/.."

MINE_PID=${1:-46532}
LOG_DIR=logs
MINED_DIR=data/mined
mkdir -p "$MINED_DIR" "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"; }

# ── 1. Wait for DeepCAD mining ──────────────────────────────────────────────
log "Waiting for DeepCAD mining (PID $MINE_PID) to finish..."
while kill -0 "$MINE_PID" 2>/dev/null; do
    PROCESSED=$(wc -l < "$MINED_DIR/deepcad_hard.pkl.processed" 2>/dev/null || echo 0)
    HARD=$(python3 -c "import pickle; d=pickle.load(open('$MINED_DIR/deepcad_hard.pkl','rb')); print(len(d))" 2>/dev/null || echo "?")
    log "  DeepCAD mining: $PROCESSED processed, $HARD hard so far"
    sleep 1800  # check every 30 minutes
done
log "DeepCAD mining finished."

# ── 2. Mine Fusion360 ────────────────────────────────────────────────────────
log "Starting Fusion360 mining (max 8000 samples)..."
python3 rl/mine.py \
    --checkpoint-path ./checkpoints/cadrille-sft \
    --data-dir ./data/cadrille_training/fusion360 \
    --output "$MINED_DIR/fusion360_hard.pkl" \
    --modality img --K 1 --R-th 0.75 \
    --max-samples 8000 --max-new-tokens 400 \
    --reward-workers 4 --checkpoint-every 500 \
    2>&1 | tee "$LOG_DIR/mine_fusion360.log"
log "Fusion360 mining finished."

# ── 3. Merge pkls ────────────────────────────────────────────────────────────
log "Merging DeepCAD + Fusion360 hard examples..."
python3 - <<'EOF'
import pickle, os

mined_dir = "data/mined"
out_path  = os.path.join(mined_dir, "combined_hard.pkl")

dc = pickle.load(open(os.path.join(mined_dir, "deepcad_hard.pkl"), "rb"))
f3 = pickle.load(open(os.path.join(mined_dir, "fusion360_hard.pkl"), "rb"))

combined = dc + f3
print(f"DeepCAD hard: {len(dc)}, Fusion360 hard: {len(f3)}, Combined: {len(combined)}")
with open(out_path, "wb") as f:
    pickle.dump(combined, f)
print(f"Saved → {out_path}")
EOF
log "Merge complete: data/mined/combined_hard.pkl"

# ── 4. Upload to HuggingFace ─────────────────────────────────────────────────
log "Uploading mined data to Hula0401/mine_CAD..."
huggingface-cli upload Hula0401/mine_CAD \
    "$MINED_DIR/combined_hard.pkl" combined_hard.pkl \
    --repo-type=dataset || log "HF upload failed (continuing)"
log "Upload done."

# ── 5. Restart training on mined data ────────────────────────────────────────
log "Starting training Run 8 (mined data)..."
RUN_NAME="cadrille-rl-run8-mined-$(date +%m%d-%H%M)"
TIMESTAMP=$(date +%m%d-%H%M)

# Patch the config: set hard_examples_pkl, clear data_dir
python3 - <<PYEOF
import re
cfg = open("configs/rl/4080.yaml").read()
# Set hard_examples_pkl
cfg = re.sub(r'^hard_examples_pkl:.*$', 'hard_examples_pkl: ./data/mined/combined_hard.pkl', cfg, flags=re.M)
# Clear data_dir (keep null or comment out)
cfg = re.sub(r'^data_dir:.*$', 'data_dir: null', cfg, flags=re.M)
open("configs/rl/4080.yaml", "w").write(cfg)
print("Config updated: hard_examples_pkl set, data_dir cleared")
PYEOF

nohup python3 rl/train.py \
    --config configs/rl/4080.yaml \
    --run-name "$RUN_NAME" \
    > "$LOG_DIR/rl-run8-$TIMESTAMP.log" 2>&1 &
TRAIN_PID=$!
log "Training started: PID=$TRAIN_PID run=$RUN_NAME"
echo "$TRAIN_PID" > "$LOG_DIR/train_run8.pid"
log "Pipeline complete. Monitor: tail -f $LOG_DIR/rl-run8-*.log"
