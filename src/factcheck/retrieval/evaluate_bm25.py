"""Evaluate BM25 retrieval against gold evidence without modifying the corpus."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from factcheck.data.schemas import DatasetRecord
from factcheck.retrieval.bm25 import RetrievalError, RetrievalResult, _iter_jsonl, load_corpus
from factcheck.retrieval.build_corpus import CorpusChunk, document_id_for_record
from factcheck.retrieval.tokenizer import normalize_for_evidence_match


class RetrievalEvaluationError(ValueError):
    """Raised when a retrieval output cannot be evaluated safely."""


def _load_records(path: Path) -> dict[str, DatasetRecord]:
    records: dict[str, DatasetRecord] = {}
    for line_number, row in _iter_jsonl(path):
        try:
            record = DatasetRecord.model_validate(row)
        except ValidationError as exc:
            raise RetrievalEvaluationError(f"{path}:{line_number}: invalid query record") from exc
        if record.id in records:
            raise RetrievalEvaluationError(
                f"{path}:{line_number}: duplicate query ID {record.id!r}"
            )
        records[record.id] = record
    if not records:
        raise RetrievalEvaluationError(f"Query file is empty: {path}")
    return records


def _load_results(path: Path) -> Iterator[RetrievalResult]:
    seen_query_ids: set[str] = set()
    for line_number, row in _iter_jsonl(path):
        try:
            result = RetrievalResult.model_validate(row)
        except ValidationError as exc:
            raise RetrievalEvaluationError(
                f"{path}:{line_number}: invalid retrieval result"
            ) from exc
        if result.query_id in seen_query_ids:
            raise RetrievalEvaluationError(
                f"{path}:{line_number}: duplicate query ID {result.query_id!r}"
            )
        seen_query_ids.add(result.query_id)
        yield result


def _gold_chunk_ids(record: DatasetRecord, chunks: list[CorpusChunk]) -> set[str]:
    """Find chunks from the record's source document containing gold evidence."""
    evidence_texts = [
        normalized
        for evidence in record.evidence
        if (normalized := normalize_for_evidence_match(evidence.text))
    ]
    if not evidence_texts:
        return set()
    return {
        chunk.chunk_id
        for chunk in chunks
        if any(
            evidence_text in normalize_for_evidence_match(chunk.text)
            for evidence_text in evidence_texts
        )
    }


def evaluate_retrieval(
    corpus_path: Path,
    queries_path: Path,
    results_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Report query-level Recall@k, chunk recall, MRR, and Precision@k.

    A chunk is relevant only if it belongs to the same source document as the
    query and contains one of that record's gold evidence strings. Records whose
    evidence cannot be aligned to any corpus chunk are excluded and reported.
    """
    try:
        corpus = load_corpus(corpus_path)
    except RetrievalError as exc:
        raise RetrievalEvaluationError(str(exc)) from exc
    records = _load_records(queries_path)
    chunks_by_document: dict[str, list[CorpusChunk]] = defaultdict(list)
    for chunk in corpus:
        chunks_by_document[chunk.document_id].append(chunk)

    query_hits = 0
    source_document_hits = 0
    total_relevant_retrieved = 0
    total_gold_chunks = 0
    reciprocal_rank_sum = 0.0
    source_document_reciprocal_rank_sum = 0.0
    precision_sum = 0.0
    evaluable_queries = 0
    skipped_query_ids: list[str] = []
    seen_results: set[str] = set()
    top_k_values: set[int] = set()
    retriever_names: set[str] = set()

    for result in _load_results(results_path):
        if result.query_id not in records:
            raise RetrievalEvaluationError(
                f"Result references query ID absent from query file: {result.query_id!r}"
            )
        seen_results.add(result.query_id)
        top_k_values.add(result.top_k)
        retriever_names.add(result.retriever)
        query = records[result.query_id]
        source_document_id = document_id_for_record(query)
        source_document_ranks = [
            rank
            for rank, chunk in enumerate(result.retrieved_chunks, start=1)
            if chunk.document_id == source_document_id
        ]
        if source_document_ranks:
            source_document_hits += 1
            source_document_reciprocal_rank_sum += 1.0 / source_document_ranks[0]
        gold_chunk_ids = _gold_chunk_ids(
            query, chunks_by_document[source_document_id]
        )
        if not gold_chunk_ids:
            skipped_query_ids.append(result.query_id)
            continue

        evaluable_queries += 1
        retrieved_ids = [chunk.chunk_id for chunk in result.retrieved_chunks]
        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(retrieved_ids, start=1)
            if chunk_id in gold_chunk_ids
        ]
        relevant_retrieved = len(relevant_ranks)
        total_relevant_retrieved += relevant_retrieved
        total_gold_chunks += len(gold_chunk_ids)
        precision_sum += relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
        if relevant_ranks:
            query_hits += 1
            reciprocal_rank_sum += 1.0 / relevant_ranks[0]

    missing_result_ids = sorted(set(records) - seen_results)
    if missing_result_ids:
        raise RetrievalEvaluationError(
            f"Missing retrieval results for {len(missing_result_ids)} queries; "
            f"first: {missing_result_ids[0]!r}"
        )
    if len(top_k_values) != 1:
        raise RetrievalEvaluationError("All retrieval results must use one shared top_k value")
    if len(retriever_names) != 1:
        raise RetrievalEvaluationError("All retrieval results must use one retriever")

    top_k = top_k_values.pop()
    retriever_name = retriever_names.pop()
    diagnostics_path = output_path.with_suffix(".unaligned_evidence.jsonl")
    diagnostics_path.write_text(
        "".join(json.dumps({"query_id": query_id}) + "\n" for query_id in skipped_query_ids),
        encoding="utf-8",
    )
    metrics = {
        "kind": "retrieval_metrics",
        "retriever": retriever_name,
        "top_k": top_k,
        "queries": len(records),
        "source_document_recall_at_k": source_document_hits / len(records),
        "source_document_mrr": source_document_reciprocal_rank_sum / len(records),
        "evaluable_queries": evaluable_queries,
        "skipped_unaligned_evidence_queries": len(skipped_query_ids),
        "unaligned_evidence_diagnostics": str(diagnostics_path),
        "recall_at_k": query_hits / evaluable_queries if evaluable_queries else None,
        "chunk_recall_at_k": (
            total_relevant_retrieved / total_gold_chunks if total_gold_chunks else None
        ),
        "mrr": reciprocal_rank_sum / evaluable_queries if evaluable_queries else None,
        "precision_at_k": precision_sum / evaluable_queries if evaluable_queries else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a fixed BM25 retrieval run")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/interim/vifactcheck/rag_corpus/corpus_chunks.jsonl"),
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_retrieval(args.corpus, args.queries, args.results, args.output)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
