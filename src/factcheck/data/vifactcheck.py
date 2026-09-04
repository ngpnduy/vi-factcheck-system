"""Reproducible ViFactCheck data pipeline.

Usage:
    bash scripts/download_vifactcheck.sh
    bash scripts/preprocess_vifactcheck.sh
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from factcheck.data.schemas import SCHEMA_VERSION, DatasetRecord, EvidenceRecord, VerdictLabel

DATASET_ID = "tranthaihoa/vifactcheck"
DATASET_NAME = "vifactcheck"
SOURCE_SPLITS = ("train", "dev", "test")
CANONICAL_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
LABELS = {
    0: VerdictLabel.SUPPORTED,
    1: VerdictLabel.REFUTED,
    2: VerdictLabel.NOT_ENOUGH_INFO,
}


class DatasetValidationError(ValueError):
    """Raised when a dataset fails a publishing gate."""


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise DatasetValidationError(f"{path}:{line_number}: record must be an object")
            rows.append(value)
    return rows


def _required_text(source: Mapping[str, Any], field: str) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ViFactCheck field {field!r} is required and must be non-empty")
    return value.strip()


def _optional_text(source: Mapping[str, Any], field: str) -> str | None:
    value = source.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"ViFactCheck field {field!r} must be a string when present")
    value = value.strip()
    return value or None


def normalize_row(source: Mapping[str, Any], source_split: str) -> DatasetRecord:
    """Map one original ViFactCheck row to the project data contract."""
    if source_split not in CANONICAL_SPLITS:
        raise ValueError(f"Unknown ViFactCheck source split: {source_split}")

    try:
        source_label = int(source["labels"])
        label = LABELS[source_label]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ViFactCheck label: {source.get('labels')!r}") from exc

    annotation_id = source.get("annotation_id")
    source_index = source.get("index")
    if source_index is None:
        raise ValueError("ViFactCheck row needs index for a stable, record-level ID")
    # annotation_id identifies a source annotation/article and repeats across claims.
    # index is unique for each claim-evidence record in the official release.
    record_id = f"{DATASET_NAME}:{source_split}:{source_index}"

    author = _optional_text(source, "Author")
    url = _optional_text(source, "Url")
    evidence_text = _optional_text(source, "Evidence")
    evidence: list[EvidenceRecord] = []
    if evidence_text is not None:
        evidence.append(
            EvidenceRecord(
                evidence_id=f"{record_id}:e1",
                text=evidence_text,
                source_title=author,
                source_url=url,
            )
        )

    return DatasetRecord(
        id=record_id,
        claim=_required_text(source, "Statement"),
        context=_optional_text(source, "Context"),
        evidence=evidence,
        label=label,
        dataset=DATASET_NAME,
        split=CANONICAL_SPLITS[source_split],
        metadata={
            "topic": _optional_text(source, "Topic"),
            "source_split": source_split,
            "annotation_id": annotation_id,
            "source_index": source_index,
        },
    )


def download_dataset(output_dir: Path, dataset_id: str = DATASET_ID) -> dict[str, int]:
    """Download official source splits and preserve their original fields as JSONL."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - setup enforces this dependency
        raise RuntimeError("Install project dependencies with bash scripts/setup.sh") from exc

    dataset = load_dataset(dataset_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in SOURCE_SPLITS:
        if split not in dataset:
            raise DatasetValidationError(f"Source dataset is missing required split: {split}")
        counts[split] = _write_jsonl(output_dir / f"{split}.jsonl", dataset[split])

    metadata = {
        "dataset_id": dataset_id,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_splits": counts,
        "format": "original rows serialized to JSONL",
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return counts


def normalize_dataset(raw_dir: Path, interim_dir: Path) -> dict[str, int]:
    """Normalize all official source splits into canonical records."""
    counts: dict[str, int] = {}
    for source_split in SOURCE_SPLITS:
        source_path = raw_dir / f"{source_split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw source split: {source_path}")
        records = [
            normalize_row(row, source_split).model_dump(mode="json")
            for row in _read_jsonl(source_path)
        ]
        counts[source_split] = _write_jsonl(interim_dir / f"{source_split}.jsonl", records)
    return counts


def validate_dataset(
    dataset_dir: Path,
    *,
    dataset_name: str = DATASET_NAME,
    split_mapping: Mapping[str, str] = CANONICAL_SPLITS,
) -> dict[str, Any]:
    """Validate canonical interim records and write a machine-readable report."""
    report: dict[str, Any] = {"dataset": dataset_name, "valid": True, "splits": {}, "errors": []}
    seen_ids: set[str] = set()
    for source_split, canonical_split in split_mapping.items():
        path = dataset_dir / f"{source_split}.jsonl"
        if not path.exists():
            report["valid"] = False
            report["errors"].append({"split": source_split, "error": "missing split file"})
            continue
        records = _read_jsonl(path)
        report["splits"][source_split] = {
            "records": len(records),
            "canonical_split": canonical_split,
        }
        for line_number, row in enumerate(records, start=1):
            try:
                record = DatasetRecord.model_validate(row)
                if record.dataset != dataset_name:
                    raise ValueError(f"dataset must be {dataset_name!r}, got {record.dataset!r}")
                if record.split != canonical_split:
                    raise ValueError(f"split must be {canonical_split!r}, got {record.split!r}")
                if record.id in seen_ids:
                    raise ValueError(f"duplicate ID: {record.id}")
                seen_ids.add(record.id)
            except (ValidationError, ValueError) as exc:
                report["valid"] = False
                report["errors"].append(
                    {"split": source_split, "line": line_number, "error": str(exc)}
                )

    report["total_records"] = sum(item["records"] for item in report["splits"].values())
    report["unique_ids"] = len(seen_ids)
    report_path = dataset_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["valid"]:
        raise DatasetValidationError(f"Validation failed; see {report_path}")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_splits(
    interim_dir: Path,
    processed_dir: Path,
    *,
    dataset_name: str = DATASET_NAME,
    dataset_id: str = DATASET_ID,
    split_mapping: Mapping[str, str] = CANONICAL_SPLITS,
) -> dict[str, Any]:
    """Publish official splits after validation, with checksums in a manifest."""
    validate_dataset(interim_dir, dataset_name=dataset_name, split_mapping=split_mapping)
    processed_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for source_split, canonical_split in split_mapping.items():
        source_path = interim_dir / f"{source_split}.jsonl"
        output_path = processed_dir / f"{canonical_split}.jsonl"
        rows = _read_jsonl(source_path)
        _write_jsonl(output_path, rows)
        files[output_path.name] = {"records": len(rows), "sha256": _sha256(output_path)}

    manifest = {
        "dataset": dataset_name,
        "dataset_id": dataset_id,
        "schema_version": SCHEMA_VERSION,
        "source_split_mapping": dict(split_mapping),
        "files": files,
    }
    (processed_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_pipeline(raw_dir: Path, interim_dir: Path, processed_dir: Path, download: bool) -> None:
    """Run the ordered ViFactCheck pipeline, optionally downloading first."""
    if download:
        download_dataset(raw_dir)
    normalize_dataset(raw_dir, interim_dir)
    validate_dataset(interim_dir)
    publish_splits(interim_dir, processed_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViFactCheck dataset pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_paths(command: argparse.ArgumentParser) -> None:
        command.add_argument("--raw-dir", type=Path, default=Path("data/raw/vifactcheck"))
        command.add_argument("--interim-dir", type=Path, default=Path("data/interim/vifactcheck"))
        command.add_argument(
            "--processed-dir", type=Path, default=Path("data/processed/vifactcheck")
        )

    download = subparsers.add_parser("download", help="Download and preserve original source rows")
    download.add_argument("--output-dir", type=Path, default=Path("data/raw/vifactcheck"))
    download.add_argument("--dataset-id", default=DATASET_ID)

    for name, help_text in (
        ("normalize", "Normalize raw ViFactCheck files"),
        ("validate", "Validate normalized ViFactCheck files"),
        ("split", "Publish official splits and manifest"),
        ("run", "Run normalize, validate, and split; optionally download first"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        add_paths(command)
        if name == "run":
            command.add_argument(
                "--download", action="store_true", help="Download raw data before processing"
            )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        counts = download_dataset(args.output_dir, args.dataset_id)
        print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    elif args.command == "normalize":
        counts = normalize_dataset(args.raw_dir, args.interim_dir)
        print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    elif args.command == "validate":
        print(json.dumps(validate_dataset(args.interim_dir), ensure_ascii=False, sort_keys=True))
    elif args.command == "split":
        print(json.dumps(publish_splits(args.interim_dir, args.processed_dir), ensure_ascii=False))
    elif args.command == "run":
        run_pipeline(args.raw_dir, args.interim_dir, args.processed_dir, download=args.download)
        print("ViFactCheck pipeline completed")


if __name__ == "__main__":
    main()
