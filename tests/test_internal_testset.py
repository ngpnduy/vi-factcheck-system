from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from factcheck.data.internal_testset import freeze

FIXTURE = Path("tests/fixtures/internal_test_adjudicated.jsonl")


def test_freeze_writes_immutable_test_and_checksum_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "adjudicated.jsonl"
    output_path = tmp_path / "test_v1.jsonl"
    manifest_path = tmp_path / "test_v1.manifest.json"
    shutil.copy(FIXTURE, input_path)

    manifest = freeze(input_path, output_path, manifest_path)

    assert manifest["status"] == "frozen"
    assert len(manifest["sha256"]) == 64
    assert manifest["records"] == 1
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"] == manifest["sha256"]
    with pytest.raises(FileExistsError, match="already frozen"):
        freeze(input_path, output_path, manifest_path)
