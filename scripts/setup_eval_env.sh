#!/usr/bin/env bash
# scripts/setup_eval_env.sh — bootstrap a dedicated paper-repro venv at .venv-eval/.
#
# Why a separate env:
#   The main .venv/ pins transformers>=5.6 (modern training stack).
#   filapro/cadrille (cadrille-rl) and kulibinai/cadevolve-rl1 were trained
#   on transformers==4.50.3 and DRIFT in the 5.x backbone — DeepCAD IoU
#   falls 0.92 → 0.14. To reproduce paper numbers we need the 4.50.3 stack
#   + the same source-built open3d the main env uses.
#
# Usage:
#   bash scripts/setup_eval_env.sh           # create .venv-eval/ if missing, otherwise verify
#   bash scripts/setup_eval_env.sh --rebuild # nuke .venv-eval/ and reinstall from scratch
#
# After install:
#   .venv-eval/bin/python research/repro_official/run_official.sh
#   .venv-eval/bin/python research/repro_official/run_cadevolve.py ...
#
# Prereq: scripts/setup.sh must have built the open3d wheel (step [4]) so
# this script can copy it in. If it hasn't, run that first.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VENV=.venv-eval
OPEN3D_WHL_DIR=/tmp/Open3D-build/build/lib/python_package/pip_package

# ── Args ──────────────────────────────────────────────────────────────────────
REBUILD=0
if [[ "${1:-}" == "--rebuild" ]]; then
    REBUILD=1
fi

if [[ "$REBUILD" -eq 1 ]] && [[ -d "$VENV" ]]; then
    echo "[--rebuild] removing existing $VENV …"
    rm -rf "$VENV"
fi

# ── 1. uv ─────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found — install via scripts/setup.sh first"; exit 1
fi

# ── 2. venv ───────────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "[1/4] creating $VENV (python 3.11) …"
    uv venv "$VENV" --python 3.11
fi

# Helper: run pip with the eval venv's interpreter
PY="$VENV/bin/python"
PIP="uv pip install --python $PY"

# ── 3. paper-era pins (filapro/cadrille trained against these) ────────────────
echo "[2/4] paper-era HF stack (transformers==4.50.3, …) …"
$PIP \
    transformers==4.50.3 \
    tokenizers==0.21.0 \
    accelerate==0.34.2 \
    huggingface-hub==0.27.0 \
    safetensors \
    sentencepiece \
    protobuf \
    qwen-vl-utils

# Torch — the main .venv pins a specific CUDA build; just install whatever
# pip ships for the running Python (eval is CPU-light, GPU just for inference).
echo "[2/4] torch (cuda available) …"
$PIP "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu124 || \
    $PIP "torch==2.5.1" "torchvision==0.20.1"

# ── 4. CAD + render stack ────────────────────────────────────────────────────
echo "[3/4] cadquery (git), trimesh, pyvista, datasets, etc …"
$PIP \
    "git+https://github.com/CadQuery/cadquery@e99a15df3cf6a88b69101c405326305b5db8ed94" \
    trimesh \
    pyvista \
    datasets \
    pandas \
    pyarrow \
    scikit-image \
    matplotlib \
    pillow \
    tqdm

# ── 5. open3d (prefer source-built headless wheel; fall back to PyPI) ────────
echo "[4/4] open3d (headless if source-built wheel exists, else PyPI stopgap) …"
WHL=""
if compgen -G "$OPEN3D_WHL_DIR/open3d_cpu*.whl" > /dev/null; then
    WHL=$(compgen -G "$OPEN3D_WHL_DIR/open3d_cpu*.whl" | head -1)
fi
if [[ -n "$WHL" ]]; then
    $PIP "$WHL"
    echo "    ✓ source-built $(basename "$WHL")"
else
    echo "    ⚠ no source-built wheel at $OPEN3D_WHL_DIR — falling back to open3d-cpu==0.18.0 (segfaults on some meshes)"
    $PIP "open3d-cpu==0.18.0"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
$PY - <<'EOF'
import sys
print(f"  python    {sys.version.split()[0]}")
import transformers; print(f"  transformers  {transformers.__version__}")
import tokenizers;   print(f"  tokenizers    {tokenizers.__version__}")
import accelerate;   print(f"  accelerate    {accelerate.__version__}")
import torch;        print(f"  torch         {torch.__version__}  (cuda={torch.cuda.is_available()})")
import cadquery;     print(f"  cadquery      {cadquery.__version__}")
import trimesh;      print(f"  trimesh       {trimesh.__version__}")
import pyvista;      print(f"  pyvista       {pyvista.__version__}")
import open3d as o3d
hl = o3d._build_config.get('ENABLE_HEADLESS_RENDERING', False)
print(f"  open3d        {o3d.__version__}  HEADLESS={hl}{'  ✓' if hl else '  ⚠ PyPI wheel; expect SIGSEGV on ~5% of meshes'}")
EOF

echo ""
echo "Done.  Run paper-repro with:"
echo "  $PY research/repro_official/run_cadevolve.py --dataset deepcad --n-samples 300 ..."
echo "  $PY -c \"from common.metrics import compute_metrics; ...\""
echo ""
echo "Switch helpers:"
echo "  source scripts/use_eval_env.sh    # activates $VENV"
echo "  bash   scripts/setup_eval_env.sh --rebuild   # nuke + reinstall"
