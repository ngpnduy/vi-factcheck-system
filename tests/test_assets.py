from vifactcheck.assets import ensure_assets
from vifactcheck.io import sha256, write_json
import pytest


def test_offline_corruption_is_rejected_and_online_repairs_original_pin(tmp_path, monkeypatch):
    asset = tmp_path / "model.bin"
    asset.write_bytes(b"original pinned model")
    manifest = {"files": [{"path": "model.bin", "url": "https://example.invalid/pinned", "sha256": sha256(asset)}]}
    write_json(tmp_path / "assets.json", manifest)
    asset.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ensure_assets(tmp_path, offline=True)
    def download(url, target):
        assert url == manifest["files"][0]["url"]
        target.write_bytes(b"original pinned model")
    monkeypatch.setattr("vifactcheck.assets.download", download)
    assert ensure_assets(tmp_path) == manifest
    assert asset.read_bytes() == b"original pinned model"
