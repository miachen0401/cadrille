#!/usr/bin/env bash
# Run Claude Code inside a container with --dangerously-skip-permissions.
#
# Usage:
#   ./claude-container.sh                    # start a new Claude session (new container)
#   ./claude-container.sh --model opus       # start with a specific model
#
# Re-entering an existing running container (avoids creating a new one):
#   docker exec -it festive_shtern claude --dangerously-skip-permissions
#
# Or use this alias (auto-detects whichever cadrille container is running):
#   docker exec -it $(docker ps --filter ancestor=cadrille-claude -q | head -1) claude --dangerously-skip-permissions
#
# Check running containers:
#   docker ps --filter ancestor=cadrille-claude
#
# Monitor training from host (no need to enter container):
#   docker exec -it festive_shtern bash -c "~/.local/bin/nvitop"
#   docker exec -it festive_shtern tail -f /workspace/checkpoints/<run>/log.txt
#
# Kill training from host:
#   docker exec festive_shtern pkill -f "python train.py"

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

mkdir -p "${HOME}/.claude-npm-global"

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
    -v "${HOME}/.claude-npm-global:/npm-global" \
    -w /workspace \
    "${IMAGE}" \
    claude --dangerously-skip-permissions "$@"
