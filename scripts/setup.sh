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

    # SFT checkpoint — small number of files, no rate-limit issue
    huggingface-cli download maksimko123/cadrille \
        --repo-type model --local-dir checkpoints/cadrille-sft

    # Mesh datasets — downloaded as a single zip to avoid HF's 5000 req/5min
    # rate limit. huggingface-cli download resolves every file individually before
    # downloading; deepcad_test_mesh has 8048 files which blows past the limit.
    # hf_hub_download fetches a single zip file (1 resolver request).
    uv run python - <<'EOF'
import os, zipfile
from huggingface_hub import hf_hub_download

def download_zip(repo_id, zip_name, out_dir):
    n_existing = len([f for f in os.listdir(out_dir) if f.endswith('.stl')]) \
                 if os.path.isdir(out_dir) else 0
    if n_existing > 0:
        print(f"  {out_dir}: {n_existing} STL files already present, skipping")
        return
    print(f"  Downloading {zip_name} from {repo_id} ...")
    local_zip = hf_hub_download(repo_id=repo_id, filename=zip_name,
                                repo_type="dataset", local_dir="data/_zips")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Extracting → {out_dir} ...")
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(out_dir)
    n = len([f for f in os.listdir(out_dir) if f.endswith('.stl')])
    print(f"  {out_dir}: {n} STL files extracted")

download_zip("Hula0401/deepCAD_test",          "deepcad_test_mesh.zip",      "data/deepcad_test_mesh")
download_zip("Hula0401/fusion360_test_mesh",   "fusion360_test_mesh.zip",    "data/fusion360_test_mesh")
download_zip("Hula0401/deepcad_train_mesh",    "deepcad_train_mesh.zip",     "data/deepcad_train_mesh")
download_zip("Hula0401/fusion360_train_mesh",  "fusion360_train_mesh.zip",   "data/fusion360_train_mesh")
EOF

    # cadrille_training/ — combined training dir used by RL configs as data_dir
    # MeshDataset globs **/*.stl recursively, so subdirs work fine.
    mkdir -p data/cadrille_training
    if [[ ! -L "data/cadrille_training/deepcad" ]]; then
        ln -sfn "$(pwd)/data/deepcad_train_mesh" "data/cadrille_training/deepcad"
        echo "  data/cadrille_training/deepcad → deepcad_train_mesh symlink created"
    fi
    if [[ ! -L "data/cadrille_training/fusion360" ]]; then
        ln -sfn "$(pwd)/data/fusion360_train_mesh" "data/cadrille_training/fusion360"
        echo "  data/cadrille_training/fusion360 → fusion360_train_mesh symlink created"
    fi

    echo "[4/4] Data download complete."
else
    echo "[4/4] Skipping data download  (re-run with --data to download from HuggingFace)"
fi

echo ""
echo "Setup complete."
echo "  Train : uv run python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1"
echo "  Resume: uv run python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1 \\"
echo "              --checkpoint-path checkpoints/cadrille-rl-v1/checkpoint-<N>"
