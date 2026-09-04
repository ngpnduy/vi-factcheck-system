#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v uv >/dev/null 2>&1; then
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    exec uv run python -m factcheck.data.vinumfcr "$@"
fi

if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    exec "${PROJECT_DIR}/.venv/bin/python" -m factcheck.data.vinumfcr "$@"
fi

echo "Neither uv nor .venv/bin/python is available. Run: bash scripts/setup.sh" >&2
exit 1
