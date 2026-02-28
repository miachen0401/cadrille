#!/usr/bin/env bash
# Run Claude Code inside a container with --dangerously-skip-permissions.
# Usage: ./claude-container.sh [extra claude args...]

set -euo pipefail

IMAGE="cadrille-claude"

# Ensure Docker daemon is reachable
if ! docker info &>/dev/null; then
    echo "ERROR: cannot reach Docker daemon."
    echo "Make sure Docker Desktop is running and WSL integration is enabled."
    exit 1
fi

# Build image if missing
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building image: $IMAGE"
    docker build -t "$IMAGE" -f Dockerfile.claude .
fi

exec docker run --rm -it \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    -v "$(pwd):/workspace" \
    -v "${HOME}/.claude.json:/workspace/.claude.json" \
    -v "${HOME}/.claude:/workspace/.claude" \
    -w /workspace \
    "${IMAGE}" \
    claude --dangerously-skip-permissions "$@"
