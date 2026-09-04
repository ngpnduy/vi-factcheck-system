"""Normalize ViNumFCR source rows into the shared fact-checking contract."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from factcheck.data.schemas import DatasetRecord, EvidenceRecord, VerdictLabel
from factcheck.data.vifactcheck import (
    _read_jsonl,
    _write_jsonl,
    publish_splits,
    validate_dataset,
)

DATASET_NAME = "vinumfcr"
DATASET_ID = "ViNumFCR-official-release"
SOURCE_SPLITS = ("train", "dev", "test")
CANONICAL_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
FIELD_ALIASES = {
    "claim": ("claim", "Claim", "statement", "Statement"),
    "context": ("paragraph", "Paragraph", "premise", "Premise", "context", "Context"),
    "evidence": ("evidence", "Evidence", "gold_evidence", "GoldEvidence"),
    "label": ("label", "Label", "labels"),
    "topic": ("topic", "Topic", "category", "Category"),
    "id": ("id", "ID", "annotation_id", "index"),
}
NUMERIC_EXPRESSION = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)(?:\s*(?:%|triệu|tỷ|nghìn|USD|đồng|km²|km2|năm|ngày|tháng))?",
    re.IGNORECASE,
)


def _field(source: Mapping[str, Any], logical_name: str, required: bool = False) -> Any:
    for alias in FIELD_ALIASES[logical_name]:
        if alias in source:
            return source[alias]
    if required:
        aliases = ", ".join(FIELD_ALIASES[logical_name])
        raise ValueError(f"Missing ViNumFCR {logical_name} field; accepted aliases: {aliases}")
    return None


def _text(value: Any, logical_name: str, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"ViNumFCR {logical_name} must be present")
        return None
    if not isinstance(value, str):
        raise ValueError(f"ViNumFCR {logical_name} must be a string")
    value = value.strip()
    if required and not value:
        raise ValueError(f"ViNumFCR {logical_name} must not be empty")
    return value or None


def _label(value: Any) -> VerdictLabel:
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        mapping = {
            0: VerdictLabel.SUPPORTED,
            1: VerdictLabel.REFUTED,
            2: VerdictLabel.NOT_ENOUGH_INFO,
        }
        try:
            return mapping[int(value)]
        except KeyError as exc:
            raise ValueError(f"Unknown ViNumFCR numeric label: {value!r}") from exc
    normalized = str(value).replace("_", "").replace(" ", "").lower()
    mapping = {
        "supported": VerdictLabel.SUPPORTED,
        "refuted": VerdictLabel.REFUTED,
        "notenoughinfo": VerdictLabel.NOT_ENOUGH_INFO,
        "notenoughinformation": VerdictLabel.NOT_ENOUGH_INFO,
        "nei": VerdictLabel.NOT_ENOUGH_INFO,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown ViNumFCR label: {value!r}") from exc


def normalize_row(source: Mapping[str, Any], source_split: str, row_number: int) -> DatasetRecord:
    """Map one raw ViNumFCR source row without modifying its numerical text."""
    if source_split not in CANONICAL_SPLITS:
        raise ValueError(f"Unknown ViNumFCR source split: {source_split}")
    source_id = _field(source, "id")
    if source_id is None:
        source_id = row_number
    record_id = f"{DATASET_NAME}:{source_split}:{source_id}"
    claim = _text(_field(source, "claim", required=True), "claim", required=True)
    context = _text(_field(source, "context", required=True), "context", required=True)
    evidence_text = _text(_field(source, "evidence"), "evidence")
    evidence: list[EvidenceRecord] = []
    if evidence_text:
        evidence.append(EvidenceRecord(evidence_id=f"{record_id}:e1", text=evidence_text))
    numeric_text = " ".join(part for part in (claim, context, evidence_text) if part)
    return DatasetRecord(
        id=record_id,
        claim=claim or "",
        context=context,
        evidence=evidence,
        label=_label(_field(source, "label", required=True)),
        dataset=DATASET_NAME,
        split=CANONICAL_SPLITS[source_split],
        metadata={
            "topic": _text(_field(source, "topic"), "topic"),
            "source_split": source_split,
            "source_id": source_id,
            "numeric_expressions": NUMERIC_EXPRESSION.findall(numeric_text),
        },
    )


def normalize_dataset(raw_dir: Path, interim_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source_split in SOURCE_SPLITS:
        source_path = raw_dir / f"{source_split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw source split: {source_path}")
        records = [
            normalize_row(row, source_split, row_number).model_dump(mode="json")
            for row_number, row in enumerate(_read_jsonl(source_path), start=1)
        ]
        counts[source_split] = _write_jsonl(interim_dir / f"{source_split}.jsonl", records)
    return counts


def run_pipeline(raw_dir: Path, interim_dir: Path, processed_dir: Path) -> None:
    normalize_dataset(raw_dir, interim_dir)
    validate_dataset(interim_dir, dataset_name=DATASET_NAME, split_mapping=CANONICAL_SPLITS)
    publish_splits(
        interim_dir,
        processed_dir,
        dataset_name=DATASET_NAME,
        dataset_id=DATASET_ID,
        split_mapping=CANONICAL_SPLITS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ViNumFCR data pipeline")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/vinumfcr"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/vinumfcr"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/vinumfcr"))
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.interim_dir, args.processed_dir)
    print("ViNumFCR pipeline completed")


if __name__ == "__main__":
    main()
