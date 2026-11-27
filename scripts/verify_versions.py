"""Check archived source/result bytes without importing ML dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    registry = json.loads((ROOT / "versions/registry.json").read_text())
    checked = 0
    failures = []
    manifests = [(item["manifest"], ROOT / item["source"])
                 for item in registry["versions"]]
    manifests += [(item["collection"], ROOT / item["path"])
                  for item in registry["results"]]
    for manifest_path, base in manifests:
        manifest = json.loads((ROOT / manifest_path).read_text())
        for relative, metadata in manifest["files"].items():
            path = base / relative
            if not path.is_file():
                failures.append(f"Missing: {path.relative_to(ROOT)}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                failures.append(f"Changed: {path.relative_to(ROOT)}")
            checked += 1
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {checked} archived files across {len(manifests)} manifests.")
    print("This verifies archive integrity, not training reproducibility or model quality.")


if __name__ == "__main__":
    main()
