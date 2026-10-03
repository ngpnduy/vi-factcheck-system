#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

SPLIT="${1:-validation}"
TOP_K="${2:-5}"
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "validation" && "${SPLIT}" != "test" ]]; then
  echo "Usage: bash scripts/run_vifactcheck_bm25.sh [train|validation|test] [top_k]" >&2
  exit 2
fi

OUTPUT_DIR="outputs/rag/vifactcheck_bm25s_top${TOP_K}"
COMMAND=(
  -m factcheck.retrieval.corpus_bm25
  --corpus data/interim/vifactcheck/rag_corpus/corpus_chunks.jsonl
  --index-dir data/interim/vifactcheck/rag_corpus/bm25s_index
  --queries "data/processed/vifactcheck/${SPLIT}.jsonl"
  --output "${OUTPUT_DIR}/${SPLIT}_retrieval.jsonl"
  --expected-split "${SPLIT}"
  --top-k "${TOP_K}"
)

if command -v uv >/dev/null 2>&1; then
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" exec uv run python "${COMMAND[@]}"
fi

if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    exec "${PROJECT_DIR}/.venv/bin/python" "${COMMAND[@]}"
fi

echo "Neither uv nor .venv/bin/python is available. Run: bash scripts/setup.sh" >&2
exit 1
