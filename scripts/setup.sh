#!/usr/bin/env bash
# Create a reproducible local environment from pyproject.toml and uv.lock.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is required but was not found.
Install it using the official instructions, then run this script again:
https://docs.astral.sh/uv/getting-started/installation/
EOF
  exit 1
fi

if [[ ! -f "uv.lock" ]]; then
  echo "uv.lock is missing. Run 'uv lock' from the repository root first." >&2
  exit 1
fi

mkdir -p data/raw data/interim data/processed data/internal outputs/checkpoints \
  outputs/metrics outputs/predictions

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add credentials only if your task needs them."
fi

echo "Synchronizing the locked Python environment..."
uv sync --all-groups --locked

echo "Running environment checks..."
uv run python --version
uv run python -c "import pandas, pydantic, pyarrow, sklearn, yaml; print('Core dependencies: OK')"
uv run ruff check .

echo
echo "Setup complete. Activate the environment with: source .venv/bin/activate"
