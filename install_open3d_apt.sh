#!/usr/bin/env bash
# /tmp/sudo_install_open3d_apt.sh — apt deps for Open3D headless source build
#
# Mirrors Dockerfile.official line 4 (the apt-get install line).
# Run as: sudo bash /tmp/sudo_install_open3d_apt.sh
#
# Why: PyPI open3d is built with ENABLE_HEADLESS_RENDERING=False (uses GLFW,
# needs X11/Wayland → segfaults on headless). cadrille requires img mode →
# needs Open3D rebuilt with -DENABLE_HEADLESS_RENDERING=ON, which links
# against system libosmesa6 + libGL + libGLU. These are the apt deps for that.
#
# After this finishes, Claude (uid 1000) will:
#   1. Clone isl-org/Open3D@8e434558a (commit pinned by Dockerfile.official)
#   2. cmake -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF ...
#   3. make -j8  (~30-45 min)
#   4. make install-pip-package → installs into /workspace/.venv

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run with sudo (currently uid=$(id -u))" >&2
    exit 1
fi

echo "[1/2] apt-get update ..."
apt-get update

echo "[2/2] Installing Open3D headless build deps ..."
# Mirrors Dockerfile.official line 4 + a few build-essentials
apt-get install -y --no-install-recommends \
    git git-lfs wget \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libosmesa6 \
    libosmesa6-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libxi-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxrandr-dev \
    pkg-config

echo ""
echo "Done. System libs installed:"
dpkg -l libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev 2>/dev/null \
    | grep -E "^ii" | awk '{printf "  %-25s %s\n", $2, $3}'

echo ""
echo "Next: Claude (uid 1000) will run the Open3D source build — no further sudo needed."
