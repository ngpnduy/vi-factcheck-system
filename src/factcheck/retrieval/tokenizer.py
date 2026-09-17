"""Deterministic text normalization for the initial Vietnamese BM25 baseline."""

from __future__ import annotations

import re
import unicodedata

TOKENIZER_NAME = "unicode_word_v1"
_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize_vietnamese(text: str) -> list[str]:
    """Return lowercase Unicode word tokens while preserving Vietnamese diacritics.

    This intentionally simple tokenizer is transparent and has no model or
    external-service dependency. It is a reproducible BM25 baseline; a later
    experiment may replace it with Vietnamese word segmentation without changing
    the retriever interface.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _WORD_PATTERN.findall(normalized)


def normalize_for_evidence_match(text: str) -> str:
    """Normalize text so evidence can be aligned to corpus chunks by substring."""
    return " ".join(tokenize_vietnamese(text))
