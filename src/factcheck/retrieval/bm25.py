"""Fixed BM25 retrieval over a label-free fact-checking corpus.

Usage:
    python -m factcheck.retrieval.bm25 \
        --corpus data/interim/vifactcheck/rag_corpus/corpus_chunks.jsonl \
        --queries data/processed/vifactcheck/validation.jsonl \
        --output outputs/rag/vifactcheck_bm25_top5/validation_retrieval.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from factcheck.data.schemas import DatasetRecord
from factcheck.retrieval.build_corpus import CORPUS_SCHEMA_VERSION, CorpusChunk
from factcheck.retrieval.tokenizer import TOKENIZER_NAME, tokenize_vietnamese

try:
    import bm25s
except ImportError:  # pragma: no cover - dependency check is exercised during setup
    bm25s = None  # type: ignore[assignment]

RETRIEVAL_SCHEMA_VERSION = "1.0.0"
RETRIEVER_NAME = "bm25s"
BM25S_METHOD = "lucene"


class RetrievalError(ValueError):
    """Raised when corpus, query, or retrieval output is invalid."""


class RetrievedChunk(BaseModel):
    """A corpus result, including the score produced by one retriever run."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    score: float
    text: str = Field(min_length=1)
    source_url: str | None = None
    topic: str | None = None


class RetrievalResult(BaseModel):
    """All top-k chunks retrieved for one claim."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = RETRIEVAL_SCHEMA_VERSION
    retriever: str = RETRIEVER_NAME
    tokenizer: str = TOKENIZER_NAME
    query_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: str
    claim: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    retrieved_chunks: list[RetrievedChunk]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetrievalError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise RetrievalError(f"{path}:{line_number}: record must be an object")
            yield line_number, row


def load_corpus(path: Path) -> list[CorpusChunk]:
    """Load and validate a corpus created by ``build_corpus``."""
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")
    chunks: list[CorpusChunk] = []
    seen_chunk_ids: set[str] = set()
    for line_number, row in _iter_jsonl(path):
        try:
            chunk = CorpusChunk.model_validate(row)
        except ValidationError as exc:
            raise RetrievalError(f"{path}:{line_number}: invalid corpus chunk") from exc
        if chunk.schema_version != CORPUS_SCHEMA_VERSION:
            raise RetrievalError(
                f"{path}:{line_number}: unsupported corpus schema {chunk.schema_version!r}"
            )
        if chunk.chunk_id in seen_chunk_ids:
            raise RetrievalError(f"{path}:{line_number}: duplicate chunk ID {chunk.chunk_id!r}")
        seen_chunk_ids.add(chunk.chunk_id)
        chunks.append(chunk)
    if not chunks:
        raise RetrievalError(f"Corpus is empty: {path}")
    return chunks


def load_queries(path: Path, *, expected_split: str | None = None) -> list[DatasetRecord]:
    """Load canonical claims; labels/evidence are not used by retrieval."""
    if not path.exists():
        raise FileNotFoundError(f"Missing query file: {path}")
    queries: list[DatasetRecord] = []
    seen_query_ids: set[str] = set()
    for line_number, row in _iter_jsonl(path):
        try:
            query = DatasetRecord.model_validate(row)
        except ValidationError as exc:
            raise RetrievalError(f"{path}:{line_number}: invalid processed query") from exc
        if expected_split is not None and query.split != expected_split:
            raise RetrievalError(
                f"{path}:{line_number}: expected split {expected_split!r}, got {query.split!r}"
            )
        if query.id in seen_query_ids:
            raise RetrievalError(f"{path}:{line_number}: duplicate query ID {query.id!r}")
        seen_query_ids.add(query.id)
        queries.append(query)
    if not queries:
        raise RetrievalError(f"Query file is empty: {path}")
    return queries


@dataclass(frozen=True)
class BM25Retriever:
    """A BM25S index over corpus chunks, using the project's Vietnamese tokenizer."""

    chunks: list[CorpusChunk]
    _index: Any

    @classmethod
    def build(cls, chunks: list[CorpusChunk]) -> BM25Retriever:
        if bm25s is None:
            raise RuntimeError(
                "bm25s is not installed. Run: bash scripts/setup.sh"
            )
        tokenized_corpus = [tokenize_vietnamese(chunk.text) for chunk in chunks]
        if any(not tokens for tokens in tokenized_corpus):
            raise RetrievalError("Corpus contains a chunk with no indexable tokens")
        index = bm25s.BM25(method=BM25S_METHOD, backend="numpy")
        index.index(tokenized_corpus, show_progress=False)
        return cls(chunks=chunks, _index=index)

    @classmethod
    def load(cls, chunks: list[CorpusChunk], index_dir: Path) -> BM25Retriever:
        """Load a previously saved BM25S sparse index for unchanged corpus data."""
        if bm25s is None:
            raise RuntimeError("bm25s is not installed. Run: bash scripts/setup.sh")
        return cls(
            chunks=chunks,
            _index=bm25s.BM25.load(index_dir, load_corpus=False, show_progress=False),
        )

    def retrieve(self, claim: str, top_k: int) -> list[RetrievedChunk]:
        """Rank chunks for one claim, breaking equal-score ties by stable chunk ID."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_tokens = tokenize_vietnamese(claim)
        if not query_tokens:
            raise RetrievalError("Claim has no indexable tokens")

        results = self._index.retrieve(
            [query_tokens],
            k=min(top_k, len(self.chunks)),
            show_progress=False,
        )
        ranked_indices = results.documents[0]
        scores = results.scores[0]
        return [
            RetrievedChunk(
                rank=rank,
                chunk_id=self.chunks[int(index)].chunk_id,
                document_id=self.chunks[int(index)].document_id,
                score=float(score),
                text=self.chunks[int(index)].text,
                source_url=self.chunks[int(index)].source_url,
                topic=self.chunks[int(index)].topic,
            )
            for rank, (index, score) in enumerate(zip(ranked_indices, scores, strict=True), start=1)
        ]


def _index_manifest(index_dir: Path) -> Path:
    return index_dir / "index_manifest.json"


def _load_or_build_index(
    chunks: list[CorpusChunk],
    index_dir: Path | None,
    *,
    corpus_sha256: str,
) -> tuple[BM25Retriever, bool]:
    """Reuse a compatible saved BM25S index or build one from corpus chunks."""
    if index_dir is None:
        return BM25Retriever.build(chunks), False

    manifest_path = _index_manifest(index_dir)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RetrievalError(f"Invalid BM25S index manifest: {manifest_path}") from exc
        expected = {
            "retriever": RETRIEVER_NAME,
            "bm25s_version": bm25s.__version__ if bm25s is not None else None,
            "method": BM25S_METHOD,
            "tokenizer": TOKENIZER_NAME,
            "corpus_sha256": corpus_sha256,
            "corpus_chunks": len(chunks),
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            return BM25Retriever.load(chunks, index_dir), True

    retriever = BM25Retriever.build(chunks)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{index_dir.name}.", dir=index_dir.parent))
    try:
        retriever._index.save(temporary_dir, show_progress=False)
        index_manifest = {
            "retriever": RETRIEVER_NAME,
            "bm25s_version": bm25s.__version__ if bm25s is not None else None,
            "method": BM25S_METHOD,
            "tokenizer": TOKENIZER_NAME,
            "corpus_sha256": corpus_sha256,
            "corpus_chunks": len(chunks),
        }
        (temporary_dir / "index_manifest.json").write_text(
            json.dumps(index_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if index_dir.exists():
            if not index_dir.is_dir():
                raise RetrievalError(f"BM25S index path is not a directory: {index_dir}")
            shutil.rmtree(index_dir)
        temporary_dir.replace(index_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return retriever, False


def run_bm25_retrieval(
    corpus_path: Path,
    queries_path: Path,
    output_path: Path,
    *,
    top_k: int = 5,
    expected_split: str | None = None,
    index_dir: Path | None = None,
) -> dict[str, Any]:
    """Retrieve fixed top-k corpus chunks for every canonical claim."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    chunks = load_corpus(corpus_path)
    queries = load_queries(queries_path, expected_split=expected_split)
    corpus_sha256 = _sha256(corpus_path)
    retriever, index_reused = _load_or_build_index(
        chunks,
        index_dir,
        corpus_sha256=corpus_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".tmp", dir=output_path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for query in queries:
                result = RetrievalResult(
                    query_id=query.id,
                    dataset=query.dataset,
                    split=query.split,
                    claim=query.claim,
                    top_k=top_k,
                    retrieved_chunks=retriever.retrieve(query.claim, top_k),
                )
                handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    manifest = {
        "kind": "retrieval_run",
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "retriever": RETRIEVER_NAME,
        "bm25s_version": bm25s.__version__ if bm25s is not None else None,
        "method": BM25S_METHOD,
        "tokenizer": TOKENIZER_NAME,
        "top_k": top_k,
        "corpus": {"path": str(corpus_path), "sha256": corpus_sha256},
        "index": {
            "path": str(index_dir) if index_dir is not None else None,
            "reused": index_reused,
        },
        "queries": {
            "path": str(queries_path),
            "sha256": _sha256(queries_path),
            "records": len(queries),
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "records": len(queries),
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed BM25 retrieval")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/interim/vifactcheck/rag_corpus/corpus_chunks.jsonl"),
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("data/processed/vifactcheck/validation.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rag/vifactcheck_bm25_top5/validation_retrieval.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--expected-split", choices=("train", "validation", "test"))
    parser.add_argument(
        "--index-dir", type=Path, default=Path("data/interim/vifactcheck/rag_corpus/bm25s_index")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_bm25_retrieval(
        args.corpus,
        args.queries,
        args.output,
        top_k=args.top_k,
        expected_split=args.expected_split,
        index_dir=args.index_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
