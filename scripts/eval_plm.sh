#!/usr/bin/env bash
# Evaluate PLM predictions against the gold test set.
#
# Usage:
#   bash scripts/eval_plm.sh --predictions PATH --gold PATH [--output PATH]
#
# Example:
#   bash scripts/eval_plm.sh \
#     --predictions outputs/phobert_base_gold/test_predictions.jsonl \
#     --gold data/processed/vifactcheck/test.jsonl
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if command -v uv >/dev/null 2>&1; then
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    exec uv run python -m factcheck.evaluation.evaluator "$@"
fi

if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    exec "${PROJECT_DIR}/.venv/bin/python" -m factcheck.evaluation.evaluator "$@"
fi

echo "Neither uv nor .venv/bin/python is available. Run: bash scripts/setup.sh" >&2
exit 1
