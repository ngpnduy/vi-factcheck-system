from __future__ import annotations

import json
from pathlib import Path

import pytest

from factcheck.data.schemas import DatasetRecord, EvidenceRecord, VerdictLabel
from factcheck.retrieval.build_corpus import (
    CORPUS_FILENAME,
    build_corpus,
    chunk_text,
    document_id_for_record,
)


def _write_records(path: Path, records: list[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
        for record in records
    ]
    path.write_text(
        "".join(lines),
        encoding="utf-8",
    )


def _record(record_id: str, split: str, context: str | None) -> DatasetRecord:
    return DatasetRecord(
        id=record_id,
        claim="Một claim để kiểm thử.",
        context=context,
        evidence=[
            EvidenceRecord(
                evidence_id=f"{record_id}:e1",
                text="Đây là gold evidence, không được đi vào corpus.",
                source_url=f"https://example.org/{record_id.replace(':', '-')}",
            )
        ],
        label=VerdictLabel.SUPPORTED,
        dataset="vifactcheck",
        split=split,
        metadata={"topic": "Health"},
    )


def test_chunk_text_keeps_overlap_and_offsets() -> None:
    chunks = chunk_text("một hai ba bốn năm sáu bảy tám chín", chunk_words=4, overlap_words=1)

    assert chunks == [
        (0, 4, "một hai ba bốn"),
        (3, 7, "bốn năm sáu bảy"),
        (6, 9, "bảy tám chín"),
    ]


def test_build_corpus_writes_label_free_chunks_and_manifest(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    _write_records(
        processed_dir / "train.jsonl",
        [_record("vifactcheck:train:1", "train", "một hai ba bốn năm sáu bảy")],
    )
    _write_records(
        processed_dir / "validation.jsonl",
        [_record("vifactcheck:validation:2", "validation", None)],
    )
    _write_records(
        processed_dir / "test.jsonl",
        [_record("vifactcheck:test:3", "test", "tám chín mười")],
    )

    manifest = build_corpus(processed_dir, tmp_path / "corpus", chunk_words=4, overlap_words=1)
    rows = [
        json.loads(line)
        for line in (tmp_path / "corpus" / CORPUS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["records"] == {
        "source_records": 3,
        "documents": 2,
        "documents_with_context": 2,
        "skipped_without_context": 1,
        "duplicate_source_records": 0,
        "chunks": 3,
    }
    assert manifest["privacy"] == {"contains_labels": False, "contains_gold_evidence_text": False}
    assert [row["chunk_id"] for row in rows] == [
        f"{document_id_for_record(_record('vifactcheck:train:1', 'train', 'x'))}:chunk:0000",
        f"{document_id_for_record(_record('vifactcheck:train:1', 'train', 'x'))}:chunk:0001",
        f"{document_id_for_record(_record('vifactcheck:test:3', 'test', 'x'))}:chunk:0000",
    ]
    assert all("label" not in row and "evidence" not in row for row in rows)
    assert all("gold evidence" not in row["text"] for row in rows)
    assert rows[0]["source_url"] == "https://example.org/vifactcheck-train-1"


def test_build_corpus_rejects_invalid_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smaller than chunk_words"):
        build_corpus(tmp_path / "processed", tmp_path / "corpus", chunk_words=10, overlap_words=10)


def test_build_corpus_deduplicates_records_from_one_source_article(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    first = _record("vifactcheck:train:1", "train", "Context dùng chung")
    second = _record("vifactcheck:validation:2", "validation", "Context dùng chung")
    second.evidence[0].source_url = first.evidence[0].source_url
    _write_records(processed_dir / "train.jsonl", [first])
    _write_records(processed_dir / "validation.jsonl", [second])
    _write_records(processed_dir / "test.jsonl", [])

    manifest = build_corpus(processed_dir, tmp_path / "corpus", chunk_words=50, overlap_words=0)

    assert manifest["records"]["source_records"] == 2
    assert manifest["records"]["documents"] == 1
    assert manifest["records"]["duplicate_source_records"] == 1
