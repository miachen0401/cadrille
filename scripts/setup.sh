#!/usr/bin/env bash
# scripts/setup.sh — install deps and (optionally) download data for RL training
#
# Usage:
#   bash scripts/setup.sh            # install Python deps only
#   bash scripts/setup.sh --data     # deps + download checkpoint + mesh data from HF
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── 1. uv ──────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[1/4] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/4] uv $(uv --version) ✓"
fi

# ── 2. Standard deps (pyproject.toml) ─────────────────────────────────────────
echo "[2/4] Installing deps from pyproject.toml ..."
uv sync --no-install-project

# ── 3. Packages that need special build flags ──────────────────────────────────
echo "[3/4] Installing pytorch3d, cadquery (git), flash-attn ..."

# pytorch3d: git-only; --no-deps avoids re-resolving torch/numpy which are already present
uv pip install --no-deps \
    "git+https://github.com/facebookresearch/pytorch3d@06a76ef8ddd00b6c889768dfc990ae8cb07c6f2f"

# cadquery: git version has fixes not yet released on PyPI
uv pip install \
    "git+https://github.com/CadQuery/cadquery@e99a15df3cf6a88b69101c405326305b5db8ed94"

# flash-attn: needs torch CUDA headers visible at build time
uv pip install flash-attn==2.7.2.post1 --no-build-isolation

# ── GPU sanity check ──────────────────────────────────────────────────────────
uv run python - <<'EOF'
import torch
n = torch.cuda.device_count()
if n == 0:
    print("  WARNING: no CUDA GPUs found")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB")
EOF

# ── 4. Data + checkpoint (optional) ───────────────────────────────────────────
if [[ "${1:-}" == "--data" ]]; then
    echo "[4/4] Downloading checkpoint and mesh data from HuggingFace ..."
    huggingface-cli download maksimko123/cadrille \
        --repo-type model   --local-dir checkpoints/cadrille-sft
    huggingface-cli download maksimko123/deepcad_test_mesh \
        --repo-type dataset --local-dir data/deepcad_test_mesh
    huggingface-cli download maksimko123/fusion360_test_mesh \
        --repo-type dataset --local-dir data/fusion360_test_mesh
    echo "[4/4] Data download complete."
else
    echo "[4/4] Skipping data download  (re-run with --data to download from HuggingFace)"
fi

echo ""
echo "Setup complete."
echo "  Train : uv run python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1"
echo "  Resume: uv run python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1 \\"
echo "              --checkpoint-path checkpoints/cadrille-rl-v1/checkpoint-<N>"
