"""Prepare gold-free M0 BM25 inputs and separate supervision/audit artifacts.

Run with ``python -m factcheck.data.prepare_m0 --help``. Train and validation
are the defaults; test preparation requires explicitly requesting that split.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from factcheck.data.evidence_alignment import ALIGNMENT_VERSION, align_evidence
from factcheck.data.schemas import SCHEMA_VERSION, DatasetRecord, VerdictLabel
from factcheck.data.sentences import SPLITTER_VERSION, Sentence, context_id, split_sentences
from factcheck.data.vifactcheck import _read_jsonl, _sha256, _write_jsonl
from factcheck.retrieval.context_bm25 import ContextBM25Retriever
from factcheck.retrieval.tokenizer import TOKENIZER_NAME

PREPARATION_VERSION = "m0_within_context_v1"


def _check_source_manifest(
    data_dir: Path, splits: list[str]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    sources: dict[str, Any] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        checksum = _sha256(path)
        rows = _read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty source split: {path}")
        if manifest is not None:
            expected = manifest.get("files", {}).get(path.name)
            if expected is None:
                raise ValueError(f"Source manifest does not list {path.name}")
            if expected.get("sha256") != checksum or expected.get("records") != len(rows):
                raise ValueError(f"Source manifest checksum/count mismatch: {path.name}")
        sources[split] = {
            "path": str(path.resolve()), "sha256": checksum, "records": len(rows),
            "manifest_verified": manifest is not None,
        }
        rows_by_split[split] = rows
    return sources, rows_by_split


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    *,
    splits: tuple[str, ...] | list[str] = ("train", "validation"),
    candidate_k: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
    pilot_size: int = 100,
) -> dict[str, Any]:
    """Validate and prepare all selected records, without silently dropping any.

    Files are staged before publishing. A manifest records hashes of source and
    output bytes, exact preprocessing versions and selected splits. Each input
    row contains only the claim, context ID and ranked context sentences. Gold
    evidence and labels live exclusively in targets and alignment reports.
    """
    splits = list(splits)
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Request one or more distinct splits")
    if not set(splits) <= {"train", "validation", "test"}:
        raise ValueError("Splits must be train, validation or test")
    if candidate_k < 1 or pilot_size < 0:
        raise ValueError("candidate_k must be positive and pilot_size non-negative")
    if data_dir.resolve() == output_dir.resolve():
        raise ValueError("M0 output must be separate from the processed source directory")
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if set(previous.get("splits", [])) != set(splits):
            raise ValueError("Use a separate output directory for a different set of splits; "
                             "do not replace a train/validation manifest with a test manifest")
    sources, rows_by_split = _check_source_manifest(data_dir, splits)
    cache: dict[str, tuple[list[Sentence], ContextBM25Retriever]] = {}
    seen_ids: set[str] = set()
    context_sets: dict[str, set[str]] = {}
    stats: dict[str, Any] = {}
    pilot: list[dict[str, Any]] = []
    records_by_split: dict[str, list[DatasetRecord]] = {}
    # Validate all selected rows before doing expensive retrieval. Do not read
    # unrequested split contents; the default command never evaluates test.
    for split, rows in rows_by_split.items():
        records: list[DatasetRecord] = []
        for row in rows:
            record = DatasetRecord.model_validate(row)
            if record.schema_version != SCHEMA_VERSION or record.dataset != "vifactcheck":
                raise ValueError(f"Unexpected dataset/schema for {record.id}")
            if record.split != split:
                raise ValueError(f"Record {record.id} split is not {split}")
            if record.id in seen_ids:
                raise ValueError(f"Duplicate record ID: {record.id}")
            if record.context is None:
                raise ValueError(f"M0 requires context for record {record.id}")
            seen_ids.add(record.id)
            records.append(record)
        records_by_split[split] = records
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".m0-prepare-", dir=output_dir.parent) as temp_dir:
        staging = Path(temp_dir)
        for split, records in records_by_split.items():
            inputs: list[dict[str, Any]] = []
            targets: list[dict[str, Any]] = []
            problems: list[dict[str, Any]] = []
            label_counts: Counter[str] = Counter()
            alignment_counts: Counter[str] = Counter()
            context_sets[split] = set()
            sentence_counts: list[int] = []
            zero_score_claims = 0
            for record in records:
                assert record.context is not None
                cid = context_id(record.context)
                context_sets[split].add(cid)
                if cid not in cache:
                    sentences = split_sentences(record.context)
                    retriever = ContextBM25Retriever.build(sentences, k1=k1, b=b)
                    cache[cid] = sentences, retriever
                sentences, retriever = cache[cid]
                candidates = [
                    {key: value for key, value in candidate.items()
                     if key not in {"context_id", "rank"}}
                    for candidate in retriever.retrieve(record.claim, top_k=candidate_k)
                ]
                inputs.append(
                    {"id": record.id, "claim": record.claim, "context_id": cid,
                     "sentence_count": len(sentences), "candidates": candidates}
                )
                zero_score_claims += int(all(candidate["score"] == 0 for candidate in candidates))
                sentence_counts.append(len(sentences))
                if record.label == VerdictLabel.NOT_ENOUGH_INFO:
                    alignment = {"alignment_status": "not_applicable", "gold_sentence_ids": [],
                                 "alignments": []}
                else:
                    alignment = align_evidence(
                        record.context, sentences, [evidence.text for evidence in record.evidence]
                    )
                targets.append({"id": record.id, "label": record.label.value, **alignment})
                alignment_counts[alignment["alignment_status"]] += 1
                label_counts[record.label.value] += 1
                if alignment["alignment_status"] not in {"aligned", "not_applicable"}:
                    problems.append(
                        {"id": record.id, "context_id": cid, "label": record.label.value,
                         **alignment}
                    )
            _write_jsonl(staging / f"{split}.inputs.jsonl", inputs)
            _write_jsonl(staging / f"{split}.targets.jsonl", targets)
            _write_jsonl(staging / f"{split}.alignment_review.jsonl", problems)
            stats[split] = {
                "records": len(records), "label_counts": dict(label_counts),
                "unique_contexts": len(context_sets[split]), "missing_context": 0,
                "alignment_counts": dict(alignment_counts), "zero_score_claims": zero_score_claims,
                "sentences_min": min(sentence_counts), "sentences_max": max(sentence_counts),
                "sentences_mean_per_claim": sum(sentence_counts) / len(sentence_counts),
            }
        # Sample unique contexts deterministically across train/validation, using
        # a hash order rather than the first 100 source rows. No test pilot.
        pilot_contexts: dict[str, tuple[str, str]] = {}
        for split, records in records_by_split.items():
            if split == "test":
                continue
            for record in records:
                assert record.context is not None
                pilot_contexts.setdefault(context_id(record.context), (split, record.context))
        for cid in sorted(pilot_contexts)[:pilot_size]:
            split, context = pilot_contexts[cid]
            pilot.append({"context_id": cid, "source_split": split, "context": context,
                          "sentences": [asdict(sentence) for sentence in cache[cid][0]],
                          "review_status": "pending"})
        pilot_sentence_count = 0
        for example in pilot:
            context = example["context"]
            previous_end = 0
            for sentence in example["sentences"]:
                start, end = sentence["start"], sentence["end"]
                if not (previous_end <= start < end <= len(context)):
                    raise ValueError(f"Invalid pilot offsets in {example['context_id']}")
                if context[start:end] != sentence["text"]:
                    raise ValueError(f"Pilot text/offset mismatch in {example['context_id']}")
                if context[previous_end:start].strip():
                    raise ValueError(f"Dropped pilot text in {example['context_id']}")
                previous_end = end
                pilot_sentence_count += 1
            if context[previous_end:].strip():
                raise ValueError(f"Dropped pilot trailing text in {example['context_id']}")
        _write_jsonl(staging / "sentence_pilot.jsonl", pilot)
        overlaps = {
            f"{left}__{right}": len(context_sets[left] & context_sets[right])
            for i, left in enumerate(splits) for right in splits[i + 1:]
        }
        audit = {
            "splits": stats, "context_overlap_unique_counts": overlaps,
            "context_overlap_note": "Exact context overlap is not proof of label leakage; "
            "official splits do not establish generalization to unseen documents.",
            "pilot": {"contexts": len(pilot), "manual_review_complete": False,
                      "selection": "unique train/validation contexts sorted by SHA-256",
                      "automated_offset_and_coverage_check": "passed",
                      "sentences_checked": pilot_sentence_count},
        }
        (staging / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "preparation_version": PREPARATION_VERSION, "scope": "within_context",
            "splits": splits, "candidate_k": candidate_k, "splitter": SPLITTER_VERSION,
            "alignment": ALIGNMENT_VERSION,
            "bm25": {"method": "lucene", "k1": k1, "b": b, "tokenizer": TOKENIZER_NAME,
                     "library_version": importlib.metadata.version("bm25s"),
                     "ties": "sentence_index_ascending", "statistics": "per_context"},
            "sources": sources,
            "outputs": {path.name: {"sha256": _sha256(path)} for path in sorted(staging.iterdir())},
            "pilot_review_required": True,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        # Publish manifest last so successful readers can verify the full set.
        paths = sorted(staging.iterdir(), key=lambda path: path.name == "manifest.json")
        for path in paths:
            os.replace(path, output_dir / path.name)
    return {"manifest": manifest, "audit": audit}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/vifactcheck"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/vifactcheck/m0"))
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"),
                        default=["train", "validation"])
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--pilot-size", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_dataset(**vars(args))
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
