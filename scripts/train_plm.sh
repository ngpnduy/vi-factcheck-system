#!/usr/bin/env bash
# Train a PLM classifier for ViFactCheck.
#
# Usage:
#   bash scripts/train_plm.sh [--model-name MODEL] [--input-mode MODE] [--output-dir DIR]
#
# Examples:
#   bash scripts/train_plm.sh
#   bash scripts/train_plm.sh --model-name vinai/phobert-large --input-mode full_context
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    exec "${PROJECT_DIR}/.venv/bin/python" -m factcheck.classification.plm.train "$@"
fi

if command -v uv >/dev/null 2>&1; then
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    exec uv run --locked --group baseline python -m factcheck.classification.plm.train "$@"
fi

echo "Neither uv nor .venv/bin/python is available. Run: bash scripts/setup.sh" >&2
exit 1
