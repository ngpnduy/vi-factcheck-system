"""Download public pinned assets; never execute remote Python code."""
import json
import os
import urllib.request
from pathlib import Path

from .config import DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION
from .io import sha256, write_json


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"Downloading {destination.name}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        os.replace(temporary, destination)


def ensure_assets(cache=".cache", offline=False):
    cache = Path(cache)
    manifest_path = cache / "assets.json"
    if offline and not manifest_path.exists():
        raise FileNotFoundError("Offline setup needs a completed assets.json manifest")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            target = cache / entry["path"]
            if target.is_file() and sha256(target) == entry["sha256"]:
                continue
            if offline:
                raise ValueError("Cached asset missing or checksum mismatch")
            if target.exists():
                target.unlink()
            download(entry["url"], target)
            if sha256(target) != entry["sha256"]:
                raise ValueError(f"Downloaded asset differs from its pinned checksum: {entry['path']}")
        return manifest
    with urllib.request.urlopen("https://api.github.com/repos/vncorenlp/VnCoreNLP/commits/master", timeout=30) as response:
        segmenter_revision = json.load(response)["sha"]
    assets = {}
    for split in ("train", "dev", "test"):
        assets[f"dataset/{split}.parquet"] = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/data/{split}-00000-of-00001.parquet"
    for name in ("config.json", "vocab.txt", "bpe.codes", "pytorch_model.bin"):
        assets[f"phobert/{name}"] = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
    for name in ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr"):
        assets[f"vncorenlp/{name}"] = f"https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/{segmenter_revision}/{name}"
    entries = []
    for name, url in assets.items():
        # Unmanifested VnCoreNLP files must be replaced from the resolved commit.
        target = cache / name
        if name.startswith("vncorenlp/") and target.exists():
            target.unlink()
        download(url, target)
        entries.append({"path": name, "url": url, "sha256": sha256(target), "bytes": target.stat().st_size})
    manifest = {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "segmenter_revision": segmenter_revision, "files": entries}
    write_json(manifest_path, manifest)
    return manifest
