"""Use the official VnCoreNLP wseg model without a Python/JNI dependency."""
from __future__ import annotations

import base64
import json
import os
import selectors
import shutil
import subprocess
import tempfile
from pathlib import Path

from factcheck.data.vifactcheck import _sha256

VNCORE_REVISION = "62bbc58fe5d113c898eae112656be97dcf50b3a0"
VNCORE_FILES = (
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/wordsegmenter.rdr",
    "models/wordsegmenter/vi-vocab",
)


class VnCoreNLPSegmenter:
    """Own one Java process; call only from the main DataLoader process."""

    def __init__(self, model_dir: Path, timeout: float = 120):
        self.model_dir = model_dir.resolve()
        self.timeout = timeout
        manifest = self.model_dir / "assets.json"
        if not manifest.is_file():
            raise FileNotFoundError("Run scripts/setup_m0.sh to download VnCoreNLP assets")
        self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if self.manifest.get("revision") != VNCORE_REVISION:
            raise ValueError("VnCoreNLP resource revision differs from the pinned protocol")
        for name in VNCORE_FILES:
            if _sha256(self.model_dir / name) != self.manifest["sha256"][name]:
                raise ValueError(f"VnCoreNLP resource checksum mismatch: {name}")
        if shutil.which("java") is None:
            raise RuntimeError("Install Java 11+ to run the official VnCoreNLP segmenter")
        # Upstream VnCoreNLP interprets its JAR URL as a filesystem path without
        # URL decoding. Stage verified assets in a temporary ASCII path so spaces
        # and Vietnamese characters in the checkout path cannot break loading.
        self._runtime_dir = tempfile.TemporaryDirectory(prefix="m0-vncorenlp-")
        runtime = Path(self._runtime_dir.name)
        for name in VNCORE_FILES:
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(self.model_dir / name, target)
            except OSError:
                shutil.copy2(self.model_dir / name, target)
        self._log = (self.model_dir / "bridge.log").open("a", encoding="utf-8")
        bridge = Path(__file__).with_name("VnCoreNLPBridge.java").resolve()
        self._process = subprocess.Popen(
            ["java", "-Xmx1g", "-cp", str(runtime / VNCORE_FILES[0]), str(bridge)],
            cwd=runtime, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, encoding="utf-8", bufsize=1,
        )
        try:
            if self._readline() != "READY":
                raise RuntimeError("VnCoreNLP startup failed; inspect bridge.log (Java 11+ needed)")
        except Exception:
            self.close()
            raise

    def _readline(self) -> str:
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError("VnCoreNLP timed out; inspect bridge.log")
            line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("VnCoreNLP exited unexpectedly; inspect bridge.log")
        return line.rstrip("\r\n")

    def __call__(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Cannot word-segment empty text")
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self._process.stdin.write(encoded + "\n")
        self._process.stdin.flush()
        result = base64.b64decode(self._readline(), validate=True).decode("utf-8")
        if not result.strip():
            raise ValueError("VnCoreNLP returned no words")
        return result

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._process.stdout:
            self._process.stdout.close()
        self._log.close()
        self._runtime_dir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
