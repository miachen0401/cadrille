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

# pytorch3d: git-only; --no-build-isolation lets its setup.py see the already-installed torch
uv pip install --no-deps --no-build-isolation \
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

    # Mesh datasets + hard examples — all downloaded as single zips to avoid HF's
    # 5000 req/5min rate limit (deepcad_test_mesh has 8048 files alone).
    uv run python - <<'EOF'
import os, zipfile, pickle
from huggingface_hub import hf_hub_download

def download_zip(repo_id, zip_name, out_dir, repo_type="dataset"):
    n_existing = len([f for f in os.listdir(out_dir) if f.endswith('.stl')]) \
                 if os.path.isdir(out_dir) else 0
    if n_existing > 0:
        print(f"  {out_dir}: {n_existing} STL files already present, skipping")
        return
    print(f"  Downloading {zip_name} from {repo_id} ...")
    local_zip = hf_hub_download(repo_id=repo_id, filename=zip_name,
                                repo_type=repo_type, local_dir="data/_zips")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Extracting → {out_dir} ...")
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(out_dir)
    n = len([f for f in os.listdir(out_dir) if f.endswith('.stl')])
    print(f"  {out_dir}: {n} STL files extracted")

# Test meshes
download_zip("Hula0401/deepCAD_test",        "deepcad_test_mesh.zip",   "data/deepcad_test_mesh")
download_zip("Hula0401/fusion360_test_mesh", "fusion360_test_mesh.zip", "data/fusion360_test_mesh")

# Hard examples (training set for RL)
if os.path.exists("data/mined/combined_hard.pkl"):
    print("  data/mined/combined_hard.pkl already present, skipping")
else:
    os.makedirs("data/mined", exist_ok=True)
    download_zip("Hula0401/mine_CAD", "combined_hard_stls.zip", "data/mined")
    pkl = hf_hub_download("Hula0401/mine_CAD", "combined_hard.pkl",
                          repo_type="dataset", local_dir="data/mined/hf")
    with open(pkl, "rb") as f:
        rows = pickle.load(f)
    for r in rows:
        r["gt_mesh_path"] = f"./data/mined/{r['dataset']}/{r['file_name']}.stl"
    with open("data/mined/combined_hard.pkl", "wb") as f:
        pickle.dump(rows, f)
    print(f"  data/mined/combined_hard.pkl ready: {len(rows)} hard examples")
EOF

    echo "[4/4] Data download complete."
else
    echo "[4/4] Skipping data download  (re-run with --data to download from HuggingFace)"
fi

echo ""
echo "Setup complete."
echo "  Train : PYTHONUNBUFFERED=1 uv run python3 -u rl/train.py --config configs/rl/h100.yaml"
echo "  Resume: PYTHONUNBUFFERED=1 uv run python3 -u rl/train.py --config configs/rl/h100.yaml \\"
echo "              --checkpoint-path checkpoints/<run-name>/checkpoint-<N>"
