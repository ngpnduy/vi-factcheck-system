"""Explicit download of pinned, public resources needed by M0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

from factcheck.data.vifactcheck import _sha256
from factcheck.data.word_segmentation import VNCORE_FILES, VNCORE_REVISION
from factcheck.m0_config import M0Config


def download_assets(config: M0Config) -> None:
    from huggingface_hub import snapshot_download

    root = Path(config.word_segmenter_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "assets.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    hashes = {}
    for name in VNCORE_FILES:
        path = root / name
        expected = previous.get("sha256", {}).get(name)
        if (not path.exists() or not expected or previous.get("revision") != VNCORE_REVISION
                or _sha256(path) != expected):
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/{VNCORE_REVISION}/{name}"
            temporary = path.with_suffix(path.suffix + ".download")
            print(f"Downloading {name}", flush=True)
            urlretrieve(url, temporary)
            temporary.replace(path)
        hashes[name] = _sha256(path)
    manifest_path.write_text(json.dumps({
        "repository": "https://github.com/vncorenlp/VnCoreNLP",
        "revision": VNCORE_REVISION, "sha256": hashes,
    }, indent=2) + "\n", encoding="utf-8")
    snapshot = snapshot_download(
        config.model_name, revision=config.model_revision, cache_dir=config.model_cache,
        allow_patterns=["config.json", "pytorch_model.bin", "*.safetensors", "vocab.txt",
                        "bpe.codes", "tokenizer_config.json", "special_tokens_map.json",
                        "LICENSE", "README.md"],
    )
    print(f"PhoBERT ready: {snapshot}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/m0.yaml"))
    args = parser.parse_args()
    download_assets(M0Config.load(args.config))


if __name__ == "__main__":
    main()
