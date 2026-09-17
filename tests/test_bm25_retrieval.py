from __future__ import annotations

import json
from pathlib import Path

from factcheck.data.schemas import DatasetRecord, EvidenceRecord, VerdictLabel
from factcheck.retrieval.bm25 import run_bm25_retrieval
from factcheck.retrieval.build_corpus import CORPUS_FILENAME, build_corpus, document_id_for_record
from factcheck.retrieval.evaluate_bm25 import evaluate_retrieval
from factcheck.retrieval.tokenizer import tokenize_vietnamese


def _write_records(path: Path, records: list[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _record(record_id: str, split: str, claim: str, context: str, evidence: str) -> DatasetRecord:
    return DatasetRecord(
        id=record_id,
        claim=claim,
        context=context,
        evidence=[
            EvidenceRecord(
                evidence_id=f"{record_id}:e1",
                text=evidence,
                source_url=f"https://example.org/{record_id.replace(':', '-')}",
            )
        ],
        label=VerdictLabel.SUPPORTED,
        dataset="vifactcheck",
        split=split,
    )


def test_tokenizer_preserves_vietnamese_diacritics() -> None:
    assert tokenize_vietnamese("TP.HCM có dịch bệnh!") == ["tp", "hcm", "có", "dịch", "bệnh"]


def test_bm25_retrieval_and_evaluation_are_label_free(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    validation_record = _record(
        "vifactcheck:validation:1",
        "validation",
        "Khu vực có dịch sốt xuất huyết không?",
        "Báo cáo xác nhận khu vực có dịch sốt xuất huyết vào tháng sáu.",
        "khu vực có dịch sốt xuất huyết vào tháng sáu",
    )
    unrelated_record = _record(
        "vifactcheck:train:2",
        "train",
        "Đội tuyển có thi đấu không?",
        "Đội tuyển bóng đá giành chiến thắng trong trận chung kết.",
        "đội tuyển bóng đá giành chiến thắng",
    )
    another_unrelated_record = _record(
        "vifactcheck:test:3",
        "test",
        "Thời tiết hôm nay thế nào?",
        "Nhiệt độ buổi sáng giảm mạnh tại vùng núi.",
        "nhiệt độ buổi sáng giảm mạnh",
    )
    _write_records(processed_dir / "train.jsonl", [unrelated_record])
    _write_records(processed_dir / "validation.jsonl", [validation_record])
    _write_records(processed_dir / "test.jsonl", [another_unrelated_record])

    corpus_dir = tmp_path / "corpus"
    build_corpus(processed_dir, corpus_dir, chunk_words=50, overlap_words=0)
    results_path = tmp_path / "outputs" / "validation_retrieval.jsonl"
    first_run = run_bm25_retrieval(
        corpus_dir / CORPUS_FILENAME,
        processed_dir / "validation.jsonl",
        results_path,
        top_k=1,
        expected_split="validation",
        index_dir=corpus_dir / "bm25s_index",
    )
    second_run = run_bm25_retrieval(
        corpus_dir / CORPUS_FILENAME,
        processed_dir / "validation.jsonl",
        results_path,
        top_k=1,
        expected_split="validation",
        index_dir=corpus_dir / "bm25s_index",
    )

    result = json.loads(results_path.read_text(encoding="utf-8"))
    assert result["retrieved_chunks"][0]["document_id"] == document_id_for_record(validation_record)
    assert result["retriever"] == "bm25s"
    assert "label" not in result
    assert "evidence" not in result
    assert first_run["index"]["reused"] is False
    assert second_run["index"]["reused"] is True

    metrics = evaluate_retrieval(
        corpus_dir / CORPUS_FILENAME,
        processed_dir / "validation.jsonl",
        results_path,
        tmp_path / "outputs" / "metrics.json",
    )
    assert metrics["source_document_recall_at_k"] == 1.0
    assert metrics["source_document_mrr"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
