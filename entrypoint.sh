#!/usr/bin/env bash
# Install or update Claude Code before running the requested command.

set -euo pipefail

export NPM_CONFIG_PREFIX=/npm-global
export PATH="/npm-global/bin:${PATH}"

mkdir -p /npm-global

echo "[entrypoint] Updating Claude Code to latest..."
npm install -g @anthropic-ai/claude-code@latest

echo "[entrypoint] Claude Code version:"
claude --version || true

exec "$@"
