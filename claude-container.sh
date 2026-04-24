#!/usr/bin/env bash
# Run Claude Code inside the cadrille CUDA container.

set -euo pipefail

IMAGE="cadrille-claude"
DOCKERFILE="Dockerfile.claude"

if ! docker info &>/dev/null; then
    echo "ERROR: cannot reach Docker daemon."
    echo "Make sure Docker Desktop is running and WSL integration is enabled."
    exit 1
fi

if ! docker image inspect "${IMAGE}" &>/dev/null; then
    echo "[host] Building image: ${IMAGE}"
    docker build -t "${IMAGE}" -f "${DOCKERFILE}" .
fi

mkdir -p "${HOME}/.claude"
mkdir -p "${HOME}/.claude-npm-global"
touch "${HOME}/.claude.json"

exec docker run --rm -it \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace \
    -e NPM_CONFIG_PREFIX=/npm-global \
    -e PATH=/npm-global/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    -v "$(pwd):/workspace" \
    -v "${HOME}/.claude.json:/workspace/.claude.json" \
    -v "${HOME}/.claude:/workspace/.claude" \
    -v "${HOME}/.claude-npm-global:/npm-global" \
    -w /workspace \
    "${IMAGE}" \
    "$@"
