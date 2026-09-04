from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_version_show_reports_authoritative_version() -> None:
    result = subprocess.run(
        ["bash", "scripts/version.sh", "show"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Version: 0.1.0" in result.stdout


def test_version_bump_updates_version_and_pyproject(tmp_path: Path) -> None:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "version.sh", scripts / "version.sh")
    shutil.copy(ROOT / "VERSION", project / "VERSION")
    shutil.copy(ROOT / "pyproject.toml", project / "pyproject.toml")

    result = subprocess.run(
        ["bash", "scripts/version.sh", "bump", "patch"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "0.1.0 -> 0.1.1" in result.stdout
    assert (project / "VERSION").read_text(encoding="utf-8").strip() == "0.1.1"
    assert 'version = "0.1.1"' in (project / "pyproject.toml").read_text(encoding="utf-8")
