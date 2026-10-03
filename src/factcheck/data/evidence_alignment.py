"""Gold-only alignment; never imported by the retriever or prediction code."""

from __future__ import annotations

import re
from typing import Any

from factcheck.data.sentences import Sentence

ALIGNMENT_VERSION = "whitespace_exact_unique_v1"


def _normalize_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse whitespace with a map back to raw character intervals.

    Case, punctuation, accents, signs and numeric/date values are preserved.
    """
    characters: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S|\s+", text):
        characters.append(" " if match.group().isspace() else match.group())
        offsets.append(match.span())
    first, last = 0, len(characters)
    while first < last and characters[first] == " ":
        first += 1
    while last > first and characters[last - 1] == " ":
        last -= 1
    return "".join(characters[first:last]), offsets[first:last]


def align_evidence(
    context: str, sentences: list[Sentence], evidence: list[str]
) -> dict[str, Any]:
    """Map every unique exact normalized evidence span to overlapping sentences.

    A snippet may span several sentences; a record may contain several snippets.
    Ambiguous repeated snippets are never silently assigned to the first match.
    Only status='aligned' is suitable for the main retrieval metrics.
    """
    if not evidence:
        return {"alignment_status": "no_gold", "gold_sentence_ids": [], "alignments": []}
    normalized, offsets = _normalize_with_offsets(context)
    details: list[dict[str, Any]] = []
    aligned_ids: set[str] = set()
    for snippet in evidence:
        needle, _ = _normalize_with_offsets(snippet)
        positions: list[int] = []
        cursor = 0
        if needle:
            while (position := normalized.find(needle, cursor)) >= 0:
                positions.append(position)
                cursor = position + 1
        if not positions:
            details.append({"status": "unmatched", "text": snippet, "match_count": 0})
            continue
        if len(positions) > 1:
            details.append(
                {"status": "ambiguous", "text": snippet, "match_count": len(positions)}
            )
            continue
        position = positions[0]
        start = offsets[position][0]
        end = offsets[position + len(needle) - 1][1]
        ids = [s.sentence_id for s in sentences if s.start < end and s.end > start]
        if not ids:
            raise ValueError("Aligned evidence does not overlap any sentence")
        aligned_ids.update(ids)
        details.append(
            {"status": "aligned", "text": snippet, "match_count": 1,
             "start": start, "end": end, "sentence_ids": ids}
        )
    statuses = {item["status"] for item in details}
    status = next(iter(statuses)) if len(statuses) == 1 else "partial"
    return {
        "alignment_status": status,
        "gold_sentence_ids": [s.sentence_id for s in sentences if s.sentence_id in aligned_ids],
        "alignments": details,
    }
