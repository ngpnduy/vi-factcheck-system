"""Freeze an adjudicated private internal test set with a checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from factcheck.data.schemas import SCHEMA_VERSION, DatasetRecord
from factcheck.data.vifactcheck import _read_jsonl, _write_jsonl

RELEASE_ID = "test_v1"
DATASET_NAME = "internal_test_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_for_freeze(rows: list[dict[str, Any]]) -> list[DatasetRecord]:
    if not rows:
        raise ValueError("Internal test set must contain at least one adjudicated record")
    records: list[DatasetRecord] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        try:
            record = DatasetRecord.model_validate(row)
        except ValidationError as exc:
            raise ValueError(f"line {line_number} violates the data contract: {exc}") from exc
        if record.dataset != DATASET_NAME or record.split != "test":
            raise ValueError(
                f"line {line_number} must use dataset={DATASET_NAME!r} and split='test'"
            )
        if record.id in seen_ids:
            raise ValueError(f"line {line_number} duplicates ID {record.id!r}")
        seen_ids.add(record.id)
        annotation = record.metadata.get("annotation")
        provenance = record.metadata.get("provenance")
        if not isinstance(annotation, dict) or annotation.get("status") != "adjudicated":
            raise ValueError(f"line {line_number} is missing adjudicated annotation metadata")
        annotators = annotation.get("annotator_ids")
        if not isinstance(annotators, list) or len(annotators) < 2:
            raise ValueError(f"line {line_number} needs at least two annotator IDs")
        if not annotation.get("adjudicated_by"):
            raise ValueError(f"line {line_number} is missing adjudicated_by")
        if not isinstance(provenance, dict) or not provenance.get("source_url"):
            raise ValueError(f"line {line_number} is missing provenance.source_url")
        records.append(record)
    return records


def freeze(input_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate input, then create immutable test and manifest artifacts once."""
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"{RELEASE_ID} is already frozen; create a new test-set version instead"
        )
    records = _validate_for_freeze(_read_jsonl(input_path))
    _write_jsonl(output_path, (record.model_dump(mode="json") for record in records))
    label_counts = Counter(record.label.value for record in records)
    manifest = {
        "release_id": RELEASE_ID,
        "dataset": DATASET_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "frozen_at": datetime.now(UTC).isoformat(),
        "records": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "file": output_path.name,
        "sha256": _sha256(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an adjudicated internal test set")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--input", type=Path, default=Path("data/internal/adjudicated/test_v1.jsonl")
    )
    freeze_parser.add_argument("--output", type=Path, default=Path("data/internal/test_v1.jsonl"))
    freeze_parser.add_argument(
        "--manifest", type=Path, default=Path("data/internal/test_v1.manifest.json")
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.input, args.output, args.manifest), ensure_ascii=False))


if __name__ == "__main__":
    main()
