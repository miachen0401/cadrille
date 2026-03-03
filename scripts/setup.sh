#!/usr/bin/env bash
# One-time environment setup for a new machine (H100 VM, Colab, etc.)
#
# Usage:
#   bash scripts/setup.sh
#   bash scripts/setup.sh --data-from-hf    # download data from HuggingFace
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "=== Cadrille environment setup ==="

# ---------- Dependencies ----------
echo "[1/3] Installing Python dependencies..."
pip install -q \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    transformers>=4.50 accelerate \
    qwen-vl-utils \
    flash-attn --no-build-isolation \
    cadquery trimesh open3d scipy \
    wandb tqdm pyyaml

echo "[2/3] Verifying GPU access..."
python -c "
import torch
n = torch.cuda.device_count()
print(f'  {n} GPU(s) found')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}  {props.total_memory // 1024**3} GB')
"

# ---------- Data ----------
echo "[3/3] Checking data..."
if [[ ! -d "$REPO_DIR/data/cad-recode-v1.5" ]]; then
    echo "  data/cad-recode-v1.5 not found."
    if [[ "${1:-}" == "--data-from-hf" ]]; then
        echo "  Downloading from HuggingFace..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='filaPro/cad-recode', repo_type='dataset',
                  local_dir='./data/cad-recode-v1.5')
"
    else
        echo "  → Copy your data directory to $REPO_DIR/data/"
        echo "    or run:  bash scripts/setup.sh --data-from-hf"
    fi
else
    python -c "
import pickle, os
path = 'data/cad-recode-v1.5/train.pkl'
with open(path, 'rb') as f: d = pickle.load(f)
print(f'  train split: {len(d)} samples')
path = 'data/cad-recode-v1.5/val.pkl'
with open(path, 'rb') as f: d = pickle.load(f)
print(f'  val   split: {len(d)} samples')
"
fi

echo ""
echo "=== Setup complete ==="
echo "Run SFT : bash scripts/run_sft.sh"
echo "Run RL  : bash scripts/run_rl.sh --checkpoint-path <sft-checkpoint>"
