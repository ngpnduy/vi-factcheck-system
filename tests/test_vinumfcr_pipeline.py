from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from factcheck.data.schemas import VerdictLabel
from factcheck.data.vinumfcr import normalize_row, run_pipeline

FIXTURE_DIR = Path("tests/fixtures/vinumfcr_raw")


def test_normalize_vinumfcr_preserves_numeric_expressions() -> None:
    record = normalize_row(
        {
            "paragraph": "Năm 2024 đạt 1,5 tỷ đồng.",
            "claim": "Kết quả là 1,5 tỷ đồng.",
            "evidence": "Đạt 1,5 tỷ đồng.",
            "label": "Supported",
            "id": "a1",
        },
        "train",
        1,
    )

    assert record.label is VerdictLabel.SUPPORTED
    assert "1,5 tỷ" in record.metadata["numeric_expressions"]
    assert record.claim == "Kết quả là 1,5 tỷ đồng."


def test_normalize_vinumfcr_rejects_unknown_source_columns() -> None:
    with pytest.raises(ValueError, match="Missing ViNumFCR claim field"):
        normalize_row({"label": 0}, "train", 1)


def test_vinumfcr_pipeline_publishes_canonical_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURE_DIR, raw_dir)

    run_pipeline(raw_dir, tmp_path / "interim", tmp_path / "processed")

    manifest_path = tmp_path / "processed" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "vinumfcr"
    assert set(manifest["files"]) == {"train.jsonl", "validation.jsonl", "test.jsonl"}
