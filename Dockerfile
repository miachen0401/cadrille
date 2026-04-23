# Dockerfile — ML training/evaluation environment
# Usage:
#   docker build -t cadrille .
#   docker run --gpus all --rm \
#       -v $(pwd)/data:/workspace/data \
#       -v $(pwd)/checkpoints:/workspace/checkpoints \
#       cadrille python rl/train.py --config configs/rl/h100.yaml --run-name cadrille-rl-v1

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git git-lfs wget curl \
    libgl1-mesa-glx libosmesa6-dev libglu1-mesa-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Open3D — built headless (no display) for mesh rendering in evaluation
RUN git clone https://github.com/isl-org/Open3D.git \
    && cd Open3D \
    && git checkout 8e434558a9b1ecacba7854da7601a07e8bdceb26 \
    && mkdir build && cd build \
    && cmake -DENABLE_HEADLESS_RENDERING=ON -DBUILD_GUI=OFF \
             -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF .. \
    && make -j$(nproc) && make install-pip-package \
    && cd / && rm -rf Open3D

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# All standard deps — torch via CUDA 12.4 index, rest from PyPI
COPY pyproject.toml .
RUN uv pip install --system -e . --no-build-isolation

# pytorch3d: git-only; --no-build-isolation exposes system torch to setup.py at build time
RUN uv pip install --system --no-deps --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d@06a76ef8ddd00b6c889768dfc990ae8cb07c6f2f"

# cadquery: git version has fixes not yet on PyPI
RUN uv pip install --system \
    "git+https://github.com/CadQuery/cadquery@e99a15df3cf6a88b69101c405326305b5db8ed94"

# flash-attn: separate step — needs torch CUDA headers visible at build time
RUN uv pip install --system flash-attn==2.7.2.post1 --no-build-isolation

WORKDIR /workspace
