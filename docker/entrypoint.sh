#!/usr/bin/env bash
# Install or update Claude Code before running the requested command.

set -euo pipefail

export NPM_CONFIG_PREFIX=/npm-global
export PATH="/npm-global/bin:${PATH}"

mkdir -p /npm-global

# Best-effort: try to update Claude Code at startup, but don't abort the
# entrypoint on transient npm/network failures. set -e would otherwise kill
# the container before exec "$@" if npm rate-limits or the host is offline.
# Set CLAUDE_SKIP_UPDATE=1 to skip the update entirely (e.g. for reproducible
# long-running training jobs where you want the build-time pinned version).
if [[ "${CLAUDE_SKIP_UPDATE:-0}" == "1" ]]; then
    echo "[entrypoint] CLAUDE_SKIP_UPDATE=1 — using build-time installed Claude Code"
elif command -v claude >/dev/null 2>&1; then
    echo "[entrypoint] Best-effort updating Claude Code to latest ..."
    npm install -g @anthropic-ai/claude-code@latest || \
        echo "[entrypoint] WARN: npm update failed — falling back to existing Claude Code"
else
    echo "[entrypoint] No Claude Code on PATH — installing ..."
    npm install -g @anthropic-ai/claude-code@latest || {
        echo "[entrypoint] FATAL: claude not installed and npm install failed"
        exit 1
    }
fi

echo "[entrypoint] Claude Code version:"
claude --version || true

exec "$@"
