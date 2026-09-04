from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from factcheck.data.schemas import VerdictLabel
from factcheck.data.vifactcheck import (
    DatasetValidationError,
    normalize_dataset,
    normalize_row,
    publish_splits,
    validate_dataset,
)

FIXTURE_DIR = Path("tests/fixtures/vifactcheck_raw")


def test_normalize_row_maps_vifactcheck_fields() -> None:
    source = {
        "Statement": " Claim ",
        "Context": " Context ",
        "Evidence": " Evidence ",
        "Topic": "Health",
        "Author": "Publisher",
        "Url": "https://example.org/article",
        "labels": 0,
        "annotation_id": 12,
        "index": 7,
    }

    record = normalize_row(source, "train")

    assert record.id == "vifactcheck:train:7"
    assert record.claim == "Claim"
    assert record.split == "train"
    assert record.label is VerdictLabel.SUPPORTED
    assert str(record.evidence[0].source_url) == "https://example.org/article"
    assert record.metadata["source_index"] == 7


def test_normalize_row_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Invalid ViFactCheck label"):
        normalize_row({"labels": 99}, "train")


def test_pipeline_normalizes_validates_and_publishes_official_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    shutil.copytree(FIXTURE_DIR, raw_dir)

    counts = normalize_dataset(raw_dir, interim_dir)
    report = validate_dataset(interim_dir)
    manifest = publish_splits(interim_dir, processed_dir)

    assert counts == {"train": 1, "dev": 1, "test": 1}
    assert report["valid"] is True
    assert manifest["files"]["validation.jsonl"]["records"] == 1
    assert (processed_dir / "train.jsonl").exists()
    assert (processed_dir / "validation.jsonl").exists()
    assert (processed_dir / "test.jsonl").exists()
    assert (
        json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))["schema_version"]
        == "1.0.0"
    )


def test_validation_rejects_duplicate_ids_across_splits(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    shutil.copytree(FIXTURE_DIR, tmp_path / "raw")
    normalize_dataset(tmp_path / "raw", interim_dir)
    dev_path = interim_dir / "dev.jsonl"
    duplicate = json.loads(
        (interim_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    duplicate["split"] = "validation"
    dev_path.write_text(json.dumps(duplicate, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Validation failed"):
        validate_dataset(interim_dir)
