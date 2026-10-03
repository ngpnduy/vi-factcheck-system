"""Deterministic sentence boundaries with offsets into the original context.

This conservative rule-based splitter is a reproducible starting point. Its
100-context pilot must be reviewed before freezing an experimental protocol.
It is not Vietnamese word segmentation for PhoBERT.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

SPLITTER_VERSION = "conservative_vi_v1"
_ABBREVIATION = re.compile(
    r"(?:\b(?:GS|PGS|TS|ThS|BS|KS|TP|Q|P|TX|TT|H|Mr|Mrs|Ms|Dr|Prof|St)"
    r"|\b(?:PGS\.TS|GS\.TS|TP\.HCM)|\bv\.v|\bv\.d|\b[A-ZĐ])\.$",
    re.IGNORECASE,
)
_CLOSERS = '\"\'”’»)]}'


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    context_id: str
    index: int
    text: str
    start: int
    end: int


def context_id(context: str) -> str:
    """Hash context bytes, never source URLs or gold evidence."""
    return "ctx_" + hashlib.sha256(context.encode("utf-8")).hexdigest()


def split_sentences(context: str) -> list[Sentence]:
    """Split without rewriting text; offsets use Python characters, end exclusive.

    Decimal/date separators and abbreviations remain inside a sentence. A
    punctuation boundary requires whitespace/end-of-text after any closing
    quotation marks. Blank lines also delimit paragraphs. Single line breaks
    without terminal punctuation are retained as potential line wrapping.
    """
    if not isinstance(context, str) or not context.strip():
        raise ValueError("M0 requires a non-empty context")
    cid = context_id(context)
    boundaries: list[int] = []
    for match in re.finditer(r"[.!?…]+|\n[ \t\r]*\n", context):
        punctuation = match.group()
        if punctuation.startswith("\n"):
            boundaries.append(match.end())
            continue
        start, end = match.span()
        if punctuation == ".":
            # Dates (01.10.2026), decimal numbers (3.14), URLs and acronyms
            # have no whitespace after an internal dot.
            if _ABBREVIATION.search(context[:end]):
                continue
        while end < len(context) and context[end] in _CLOSERS:
            end += 1
        if end < len(context) and not context[end].isspace():
            continue
        if punctuation in {"...", "…"}:
            following = context[end:].lstrip()
            if following and following[0].islower():
                continue
        boundaries.append(end)
    boundaries.append(len(context))
    sentences: list[Sentence] = []
    previous = 0
    for boundary in sorted(set(boundaries)):
        start, end = previous, boundary
        previous = boundary
        while start < end and context[start].isspace():
            start += 1
        while end > start and context[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        # Editorial punctuation such as `! ".` can create a standalone
        # closing-quote fragment. Attach it without dropping/replacing bytes.
        # All-punctuation contexts remain visible and fail at indexing.
        if sentences and (
            not any(char.isalnum() for char in context[start:end])
            or not any(char.isalnum() for char in sentences[-1].text)
        ):
            prior = sentences[-1]
            sentences[-1] = Sentence(
                prior.sentence_id, cid, prior.index, context[prior.start:end], prior.start, end
            )
            continue
        index = len(sentences)
        sentences.append(
            Sentence(f"{cid}:s{index:04d}", cid, index, context[start:end], start, end)
        )
    if not sentences:
        raise ValueError(f"Context {cid} produced no sentences")
    return sentences
