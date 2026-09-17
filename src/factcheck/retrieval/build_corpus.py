"""Build a label-free, chunked retrieval corpus from processed datasets.

The corpus deliberately contains only source text and public source metadata.
Gold labels and gold evidence remain in ``data/processed`` and are never written
to the retrieval index.

Example:
    bash scripts/build_vifactcheck_rag_corpus.sh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from factcheck.data.schemas import DatasetRecord

CORPUS_SCHEMA_VERSION = "1.0.0"
DEFAULT_SPLITS = ("train", "validation", "test")
CORPUS_FILENAME = "corpus_chunks.jsonl"
_WORD_PATTERN = re.compile(r"\S+")


class CorpusBuildError(ValueError):
    """Raised when a processed dataset cannot safely become a retrieval corpus."""


class CorpusChunk(BaseModel):
    """One label-free text unit that can be ranked by a retriever."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = CORPUS_SCHEMA_VERSION
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    source_split: str
    chunk_index: int = Field(ge=0)
    word_start: int = Field(ge=0)
    word_end: int = Field(gt=0)
    text: str = Field(min_length=1)
    source_url: str | None = None
    topic: str | None = None

    @model_validator(mode="after")
    def validate_word_range(self) -> CorpusChunk:
        if self.word_end <= self.word_start:
            raise ValueError("word_end must be greater than word_start")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_chunking_options(chunk_words: int, overlap_words: int) -> None:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than zero")
    if overlap_words < 0:
        raise ValueError("overlap_words must not be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")


def chunk_text(text: str, *, chunk_words: int, overlap_words: int) -> list[tuple[int, int, str]]:
    """Split text into deterministic, overlapping whitespace-token windows.

    The returned start/end offsets are token offsets in the normalized whitespace
    representation. A later retrieval evaluator can use them to inspect evidence
    alignment without needing labels inside the corpus.
    """
    _validate_chunking_options(chunk_words, overlap_words)
    words = _WORD_PATTERN.findall(text)
    windows: list[tuple[int, int, str]] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        if start >= end:
            break
        windows.append((start, end, " ".join(words[start:end])))
        if end == len(words):
            break
    return windows


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusBuildError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise CorpusBuildError(f"{path}:{line_number}: record must be an object")
            yield line_number, row


def _source_url(record: DatasetRecord) -> str | None:
    """Take public source metadata without copying the gold evidence text."""
    for evidence in record.evidence:
        if evidence.source_url is not None:
            return str(evidence.source_url)
    return None


def document_id_for_record(record: DatasetRecord) -> str:
    """Derive a stable source-document ID without using labels or evidence text.

    ViFactCheck has multiple claim records for the same news article. The URL is
    the preferred identity; context bytes are a safe fallback when the release
    does not provide a URL. This prevents copies of one article from dominating
    BM25 scores merely because it generated several claims.
    """
    source_url = _source_url(record)
    if source_url is not None:
        identity = f"url:{source_url}"
    elif record.context is not None:
        identity = f"context:{record.context}"
    else:
        raise CorpusBuildError(f"Record {record.id!r} has neither source URL nor context")
    fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{record.dataset}:document:{fingerprint}"


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write a generated corpus atomically so a failed build keeps the old one."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


def build_corpus(
    processed_dir: Path,
    output_dir: Path,
    *,
    dataset_name: str = "vifactcheck",
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    chunk_words: int = 200,
    overlap_words: int = 40,
) -> dict[str, Any]:
    """Build and manifest a label-free corpus from canonical processed files.

    ``processed_dir`` is read-only input. This function never writes predictions,
    labels, or gold evidence to the corpus. It accepts all three official splits
    because this is a closed-corpus setup: test documents are searchable, while
    their labels and annotations remain unavailable to the retriever and verifier.
    """
    _validate_chunking_options(chunk_words, overlap_words)
    if not splits:
        raise ValueError("at least one split is required")
    if len(set(splits)) != len(splits):
        raise ValueError("splits must not contain duplicates")

    input_paths = {split: processed_dir / f"{split}.jsonl" for split in splits}
    for split, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split {split!r}: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / CORPUS_FILENAME
    source_records: dict[str, int] = {split: 0 for split in splits}
    seen_record_ids: set[str] = set()
    source_documents: dict[str, str] = {}
    documents_with_context = 0
    skipped_without_context = 0
    duplicate_source_records = 0

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal documents_with_context, skipped_without_context, duplicate_source_records
        for requested_split, path in input_paths.items():
            for line_number, row in _iter_jsonl(path):
                try:
                    record = DatasetRecord.model_validate(row)
                except ValidationError as exc:
                    raise CorpusBuildError(
                        f"{path}:{line_number}: does not satisfy DatasetRecord schema"
                    ) from exc
                if record.dataset != dataset_name:
                    raise CorpusBuildError(
                        f"{path}:{line_number}: dataset must be {dataset_name!r}, "
                        f"got {record.dataset!r}"
                    )
                if record.split != requested_split:
                    raise CorpusBuildError(
                        f"{path}:{line_number}: split must be {requested_split!r}, "
                        f"got {record.split!r}"
                    )
                if record.id in seen_record_ids:
                    raise CorpusBuildError(
                        f"{path}:{line_number}: duplicate record ID {record.id!r}"
                    )

                seen_record_ids.add(record.id)
                source_records[requested_split] += 1
                if record.context is None:
                    skipped_without_context += 1
                    continue

                document_id = document_id_for_record(record)
                existing_context = source_documents.get(document_id)
                if existing_context is not None:
                    if existing_context != record.context:
                        raise CorpusBuildError(
                            f"{path}:{line_number}: source document {document_id!r} "
                            "has conflicting contexts"
                        )
                    duplicate_source_records += 1
                    continue

                source_documents[document_id] = record.context
                documents_with_context += 1
                topic = record.metadata.get("topic")
                topic_value = topic.strip() if isinstance(topic, str) and topic.strip() else None
                source_url = _source_url(record)
                for chunk_index, (word_start, word_end, text) in enumerate(
                    chunk_text(
                        record.context,
                        chunk_words=chunk_words,
                        overlap_words=overlap_words,
                    )
                ):
                    chunk = CorpusChunk(
                        chunk_id=f"{document_id}:chunk:{chunk_index:04d}",
                        document_id=document_id,
                        dataset=record.dataset,
                        source_split=record.split,
                        chunk_index=chunk_index,
                        word_start=word_start,
                        word_end=word_end,
                        text=text,
                        source_url=source_url,
                        topic=topic_value,
                    )
                    yield chunk.model_dump(mode="json")

    chunks = _atomic_write_jsonl(corpus_path, rows())
    manifest = {
        "kind": "retrieval_corpus",
        "schema_version": CORPUS_SCHEMA_VERSION,
        "dataset": dataset_name,
        "source": {
            "processed_dir": str(processed_dir),
            "splits": list(splits),
            "files": {
                f"{split}.jsonl": {
                    "records": source_records[split],
                    "sha256": _sha256(path),
                }
                for split, path in input_paths.items()
            },
        },
        "chunking": {
            "strategy": "whitespace_window",
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
        },
        "records": {
            "source_records": len(seen_record_ids),
            "documents": len(source_documents),
            "documents_with_context": documents_with_context,
            "skipped_without_context": skipped_without_context,
            "duplicate_source_records": duplicate_source_records,
            "chunks": chunks,
        },
        "files": {
            CORPUS_FILENAME: {
                "records": chunks,
                "sha256": _sha256(corpus_path),
            }
        },
        "privacy": {
            "contains_labels": False,
            "contains_gold_evidence_text": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a label-free retrieval corpus")
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed/vifactcheck")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/interim/vifactcheck/rag_corpus")
    )
    parser.add_argument("--dataset-name", default="vifactcheck")
    parser.add_argument("--splits", nargs="+", choices=DEFAULT_SPLITS, default=list(DEFAULT_SPLITS))
    parser.add_argument("--chunk-words", type=int, default=200)
    parser.add_argument("--overlap-words", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_corpus(
        args.processed_dir,
        args.output_dir,
        dataset_name=args.dataset_name,
        splits=tuple(args.splits),
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
