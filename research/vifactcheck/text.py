"""Segmentation with source offsets and conservative gold alignment."""
import re
import shutil
import subprocess
import tempfile
import unicodedata
import uuid
from pathlib import Path


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _offset_key(character):
    # Used only to recover monotonic original offsets after VnCoreNLP tone
    # normalization. Evidence matching below never strips accents or punctuation.
    return "".join(c for c in unicodedata.normalize("NFD", character.casefold()) if not unicodedata.combining(c) and not c.isspace() and c != "_")


def recover_sentences(raw, segmented_sentences):
    source, offsets = [], []
    for i, char in enumerate(raw):
        for part in _offset_key(char):
            source.append(part)
            offsets.append(i)
    source = "".join(source)
    cursor, sentences = 0, []
    for segmented in segmented_sentences:
        key = "".join(_offset_key(c) for c in segmented)
        if not key:
            continue
        if source[cursor:cursor + len(key)] != key:
            raise ValueError(f"Cannot recover original sentence offsets near character {offsets[cursor] if cursor < len(offsets) else len(raw)}")
        start, end = offsets[cursor], offsets[cursor + len(key) - 1] + 1
        sentences.append({"id": len(sentences), "start": start, "end": end, "text": raw[start:end], "segmented": segmented})
        cursor += len(key)
    if cursor != len(source):
        raise ValueError("Segmentation omitted source content")
    if not sentences:
        raise ValueError("Context contains no sentences")
    return sentences


class VnCoreSegmenter:
    def __init__(self, root=".cache/vncorenlp", java=None):
        self.root = Path(root).resolve()
        self.java = java or shutil.which("java")
        if not self.java:
            raise RuntimeError("Java 8+ is required for VnCoreNLP")
        if not (self.root / "VnCoreNLP-1.2.jar").exists():
            raise FileNotFoundError("VnCoreNLP assets missing; run prepare first")
        self._runtime_temp = None
        if any(c.isspace() for c in str(self.root)) or not str(self.root).isascii():
            # VnCoreNLP 1.2 resolves its models from an undecoded JAR URL.
            # A temporary ASCII path handles workspaces containing spaces.
            self._runtime_temp = tempfile.TemporaryDirectory(prefix="vifactcheck_vncorenlp_")
            runtime = Path(self._runtime_temp.name)
            shutil.copy2(self.root / "VnCoreNLP-1.2.jar", runtime)
            shutil.copytree(self.root / "models", runtime / "models")
            self.root = runtime

    def segment(self, text):
        return self.segment_many([text])[0]

    def segment_many(self, texts):
        if any(not normalize(text) for text in texts):
            raise ValueError("Cannot segment empty text")
        if not texts:
            return []
        # Lowercase alphabetic markers avoid VnCoreNLP joining them with a
        # preceding capitalized proper name or normalizing numeric fragments.
        prefix = "vfcdocboundary" + "".join(chr(97 + int(c, 16)) for c in uuid.uuid4().hex) + "boundary"
        markers = {prefix + str(i): i for i in range(len(texts))}
        lines = []
        for i, text in enumerate(texts):
            lines.extend([prefix + str(i) + ".", normalize(text)])
        # The command-line component is cross-platform and avoids an in-process
        # Java VM holding RAM while two PhoBERT encoders train.
        with tempfile.TemporaryDirectory(prefix="vifactcheck_", dir=self.root) as temporary:
            inp, out = Path(temporary) / "input.txt", Path(temporary) / "output.txt"
            inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = subprocess.run([self.java, "-Xmx512m", "-Dfile.encoding=UTF-8", "-jar", str(self.root / "VnCoreNLP-1.2.jar"), "-fin", str(inp), "-fout", str(out), "-annotators", "wseg"], cwd=self.root, capture_output=True, timeout=max(120, len(texts) * 2))
            if result.returncode:
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
            content = out.read_text(encoding="utf-8")
        documents, tokens, current = [[] for _ in texts], [], None
        for line in content.splitlines() + [""]:
            if not line.strip():
                if tokens:
                    # A source without terminal punctuation can share a
                    # sentence block with the next marker. Locate markers
                    # inside the block as well as in standalone sentences.
                    joined = "".join(tokens)
                    starts, at = [], 0
                    for token in tokens:
                        starts.append(at)
                        at += len(token)
                    consumed = 0
                    for match in re.finditer(re.escape(prefix) + r"\d+", joined):
                        key = match.group()
                        if key not in markers:
                            raise RuntimeError("Unexpected segmentation boundary marker")
                        first = next(i for i, pos in enumerate(starts) if pos == match.start())
                        last = next((i for i, pos in enumerate(starts) if pos >= match.end()), len(tokens))
                        before = tokens[consumed:first]
                        if before and current is not None:
                            documents[current].append(" ".join(before))
                        current = markers[key]
                        consumed = last + (1 if last < len(tokens) and tokens[last] == "." else 0)
                    if tokens[consumed:] and current is not None:
                        documents[current].append(" ".join(tokens[consumed:]))
                    tokens = []
            else:
                columns = line.split("\t")
                if len(columns) >= 2 and columns[0].isdigit():
                    tokens.append(columns[1])
        if any(not sentences for sentences in documents):
            missing = [i for i, sentences in enumerate(documents) if not sentences]
            debug = Path(".cache/segmentation_debug").resolve()
            debug.mkdir(parents=True, exist_ok=True)
            (debug / "last_input.txt").write_text("\n".join(lines), encoding="utf-8")
            (debug / "last_output.txt").write_text(content, encoding="utf-8")
            raise RuntimeError(f"VnCoreNLP produced no tokens for document positions {missing}; diagnostic input/output in {debug}")
        return documents


def align_evidence(context, sentences, evidence):
    if not evidence or not normalize(evidence):
        return {"status": "missing", "gold_sentence_ids": []}
    # Exact normalized spans only. Newlines can delimit multiple independently
    # annotated snippets. Unmatched, edited, or reordered snippets require review.
    snippets = [normalize(piece) for piece in evidence.splitlines() if normalize(piece)]
    # Build normalized-character offsets without modifying the stored source.
    chars, mapping, pending_space = [], [], False
    for match in re.finditer(r"[^\u0300-\u036f][\u0300-\u036f]*|[\u0300-\u036f]+", context):
        i, char = match.start(), match.group()
        if char.isspace():
            pending_space = bool(chars)
            continue
        if pending_space:
            chars.append(" ")
            mapping.append((i, match.end()))
            pending_space = False
        for c in unicodedata.normalize("NFC", char):
            chars.append(c)
            mapping.append((i, match.end()))
    normalized_context = "".join(chars)
    ids = set()
    for snippet in snippets:
        matches = [m.start() for m in re.finditer(re.escape(snippet), normalized_context)]
        if len(matches) != 1:
            return {"status": "ambiguous" if len(matches) > 1 else "unmatched", "gold_sentence_ids": []}
        at = matches[0]
        start, end = mapping[at][0], mapping[at + len(snippet) - 1][1]
        covered = [s["id"] for s in sentences if s["start"] < end and s["end"] > start]
        if not covered:
            return {"status": "unmatched", "gold_sentence_ids": []}
        ids.update(covered)
    return {"status": "valid", "gold_sentence_ids": sorted(ids)}
