#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/run_m0.sh {setup|prepare|train|predict|evaluate} [arguments]" >&2
    exit 2
fi
case "$1" in
    setup) module=factcheck.data.m0_assets ;;
    prepare) module=factcheck.data.prepare_m0 ;;
    train) module=factcheck.training.train_m0 ;;
    predict) module=factcheck.inference.predict_m0 ;;
    evaluate) module=factcheck.evaluation.evaluate_m0 ;;
    *) echo "Unknown M0 command: $1" >&2; exit 2 ;;
esac
shift
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    exec "${PROJECT_DIR}/.venv/bin/python" -m "${module}" "$@"
elif command -v uv >/dev/null 2>&1; then
    exec uv run --locked --all-groups python -m "${module}" "$@"
fi
echo "Install dependencies first: bash scripts/setup.sh" >&2
exit 1
