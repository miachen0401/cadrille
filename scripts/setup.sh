#!/usr/bin/env bash
# scripts/setup.sh — install deps and (optionally) download training + eval data.
#
# Usage:
#   bash scripts/setup.sh            # install Python deps only
#   bash scripts/setup.sh --data     # deps + SFT data (BenchCAD + cad-sft) + eval meshes
#   bash scripts/setup.sh --full     # --data + RL hard-mined examples + full training meshes
#
# Env: .env must contain HF_TOKEN and BenchCAD_HF_TOKEN (for private repos).
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ── 0. /ephemeral directories ──────────────────────────────────────────────────
if [ -d /ephemeral ] && [ ! -w /ephemeral ]; then
    echo "[0/4] Fixing /ephemeral permissions ..."
    sudo chown "$USER" /ephemeral
fi
if [ -d /ephemeral ]; then
    mkdir -p /ephemeral/checkpoints
    echo "[0/4] /ephemeral/checkpoints ✓"
fi

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

# setuptools is required for --no-build-isolation source builds below
# (pytorch3d and flash-attn invoke setuptools directly without declaring it).
uv pip install setuptools

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

# ── 4. Data (optional) ────────────────────────────────────────────────────────
if [[ "${1:-}" == "--data" || "${1:-}" == "--full" ]]; then
    echo "[4/4] Downloading data from HuggingFace ..."
    [[ "${1:-}" == "--full" ]] && export DOWNLOAD_TRAIN_MESHES=1

    # Source .env so HF_TOKEN / BenchCAD_HF_TOKEN are present for private repos.
    if [ -f .env ]; then set -a; source .env; set +a; fi

    # 4.A — BenchCAD training corpus (18k .py/.stl/_render.png + metadata)
    echo "[4.A] BenchCAD training corpus ..."
    if [ -f data/benchcad/train.pkl ]; then
        echo "  data/benchcad already present, skipping"
    else
        uv run python -m data_prep.fetch_benchcad --workers 8
    fi

    # 4.B — cad-sft (cad-recode-20k with pre-rendered PNGs + text2cad descriptions)
    echo "[4.B] Hula0401/cad-sft (cad-recode-20k + text2cad) ..."
    if [ -f data/cad-recode-20k/train.pkl ]; then
        echo "  data/cad-recode-20k already present, skipping"
    else
        uv run python -m data_prep.fetch_cad_sft --what all
    fi

    # 4.C — Test meshes for eval (DeepCAD 8046 + Fusion360 1725, both with _render.png)
    # Reference SFT checkpoint — only fetched under --full to save bandwidth for SFT-only users
    if [[ "${1:-}" == "--full" ]]; then
        echo "[4.C-full] reference SFT checkpoint ..."
        uv run huggingface-cli download maksimko123/cadrille \
            --repo-type model --local-dir checkpoints/cadrille-sft
    fi

    echo "[4.C] Eval test meshes (DeepCAD + Fusion360) ..."
    uv run python - <<'EOF'
import os, sys, zipfile, pickle
from huggingface_hub import hf_hub_download

FULL = "--full" in sys.argv  # noqa: only read, not passed here (handled in bash below)

def download_zip(repo_id, zip_name, out_dir, repo_type="dataset", skip_if_ext='.stl', skip_dir=None):
    """Download a zip from HuggingFace and extract to out_dir.
    Skips if skip_dir (default: out_dir) already contains files with skip_if_ext.
    Use skip_if_ext='_render.png' and skip_dir='data/mined/deepcad' for render zips
    where out_dir='data/mined' but the zip expands into a deepcad/ subdirectory.
    """
    check_dir = skip_dir or out_dir
    if os.path.isdir(check_dir):
        n_existing = len([f for f in os.listdir(check_dir) if f.endswith(skip_if_ext)])
        if n_existing > 0:
            print(f"  {check_dir}: {n_existing} {skip_if_ext} files already present, skipping {zip_name}")
            return
    print(f"  Downloading {zip_name} from {repo_id} ...")
    local_zip = hf_hub_download(repo_id=repo_id, filename=zip_name,
                                repo_type=repo_type, local_dir="data/_zips")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Extracting → {out_dir} ...")
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(out_dir)
    n_stl = len([f for f in os.listdir(out_dir) if f.endswith('.stl')])
    n_png = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    print(f"  {out_dir}: {n_stl} STLs, {n_png} PNGs")

# Test meshes (STLs + pre-rendered PNGs) — always downloaded
download_zip("Hula0401/deepCAD_test",        "deepcad_test_mesh.zip",      "data/deepcad_test_mesh")
download_zip("Hula0401/deepCAD_test",        "deepcad_test_renders.zip",   "data/deepcad_test_mesh",   skip_if_ext='_render.png')
download_zip("Hula0401/fusion360_test_mesh", "fusion360_test_mesh.zip",    "data/fusion360_test_mesh")
download_zip("Hula0401/fusion360_test_mesh", "fusion360_test_renders.zip", "data/fusion360_test_mesh", skip_if_ext='_render.png')

# Hard-mined examples — only needed for RL. Gated on --full.
_FULL = bool(os.environ.get("DOWNLOAD_TRAIN_MESHES"))
if not _FULL:
    print("  Skipping RL hard-mined examples (only needed for RL training). Re-run with --full.")
elif os.path.exists("data/mined/combined_hard.pkl"):
    print("  data/mined/combined_hard.pkl already present, skipping")
else:
    os.makedirs("data/mined", exist_ok=True)
    download_zip("Hula0401/mine_CAD", "combined_hard_stls.zip",      "data/mined")
    download_zip("Hula0401/mine_CAD", "deepcad_hard_renders.zip",    "data/mined/deepcad",   skip_if_ext='_render.png')
    download_zip("Hula0401/mine_CAD", "fusion360_hard_renders.zip",  "data/mined/fusion360", skip_if_ext='_render.png')
    pkl = hf_hub_download("Hula0401/mine_CAD", "combined_hard.pkl",
                          repo_type="dataset", local_dir="data/mined/hf")
    with open(pkl, "rb") as f:
        rows = pickle.load(f)
    for r in rows:
        r["gt_mesh_path"] = f"./data/mined/{r['dataset']}/{r['file_name']}.stl"
    with open("data/mined/combined_hard.pkl", "wb") as f:
        pickle.dump(rows, f)
    print(f"  data/mined/combined_hard.pkl ready: {len(rows)} hard examples")

# Full training meshes — only needed for mining new hard examples (--full flag)
if os.environ.get("DOWNLOAD_TRAIN_MESHES"):
    print("  [--full] Downloading full training meshes for mining ...")
    download_zip("Hula0401/deepcad_train_mesh",    "deepcad_train_mesh.zip",      "data/deepcad_train_mesh")
    download_zip("Hula0401/deepcad_train_mesh",    "deepcad_train_renders.zip",   "data/deepcad_train_mesh",   skip_if_ext='_render.png')
    download_zip("Hula0401/fusion360_train_mesh",  "fusion360_train_mesh.zip",    "data/fusion360_train_mesh")
    download_zip("Hula0401/fusion360_train_mesh",  "fusion360_train_renders.zip", "data/fusion360_train_mesh", skip_if_ext='_render.png')
else:
    print("  Skipping full training meshes (only needed for mining). Re-run with --full to download.")
EOF

    echo "[4/4] Data download complete."
else
    echo "[4/4] Skipping data download  (re-run with --data to download from HuggingFace)"
fi

echo ""
echo "Setup complete."
echo ""
echo "  SFT (BenchCAD + cad-recode-20k):"
echo "    uv run python -m train.sft --config configs/sft/benchcad_full.yaml"
echo ""
echo "  Eval on SFT ckpt:"
echo "    uv run python -m eval.bench_sweep \\"
echo "      --ckpt checkpoints/<run-name>/checkpoint-final \\"
echo "      --datasets benchcad,deepcad,fusion360 --limit 30 \\"
echo "      --out eval_outputs/<tag>"
echo ""
echo "  RL (needs --full data):"
echo "    uv run python -m train.rl.train --config configs/rl/h100.yaml"
