# scripts/use_eval_env.sh — activate the dedicated paper-repro venv.
#
# Usage:
#   source scripts/use_eval_env.sh           # activate
#   .venv-eval/bin/deactivate                # later, to exit (or just `deactivate`)
#
# The eval venv (.venv-eval/) ships transformers==4.50.3 + the same
# source-built open3d wheel as .venv/.  See scripts/setup_eval_env.sh for
# how it's built.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"

if [[ ! -d "$REPO_DIR/.venv-eval" ]]; then
    echo "[!] .venv-eval/ not found — bootstrap with: bash scripts/setup_eval_env.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate
source "$REPO_DIR/.venv-eval/bin/activate"

echo "🔁 switched to .venv-eval (paper-repro: transformers $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo '?'))"
echo "   to switch back to main .venv:  deactivate && source .venv/bin/activate"
