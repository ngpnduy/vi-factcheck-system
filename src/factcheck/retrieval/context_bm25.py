"""M0 sentence retrieval with IDF and length statistics local to one context."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import bm25s

from factcheck.data.sentences import Sentence
from factcheck.retrieval.tokenizer import tokenize_vietnamese


@dataclass
class ContextBM25Retriever:
    sentences: list[Sentence]
    _index: Any

    @classmethod
    def build(
        cls, sentences: list[Sentence], *, k1: float = 1.5, b: float = 0.75
    ) -> ContextBM25Retriever:
        if not sentences:
            raise ValueError("Cannot index an empty context")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        if len({s.context_id for s in sentences}) != 1:
            raise ValueError("A within-context index must contain exactly one context")
        if len({s.sentence_id for s in sentences}) != len(sentences):
            raise ValueError("Duplicate sentence IDs")
        sentences = sorted(sentences, key=lambda sentence: sentence.index)
        tokens = [tokenize_vietnamese(sentence.text) for sentence in sentences]
        for sentence, sentence_tokens in zip(sentences, tokens, strict=True):
            if not sentence_tokens:
                raise ValueError(f"Sentence has no indexable tokens: {sentence.sentence_id}")
        index = bm25s.BM25(method="lucene", backend="numpy", k1=k1, b=b)
        index.index(tokens, show_progress=False)
        return cls(sentences=sentences, _index=index)

    def retrieve(self, claim: str, top_k: int = 20) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = tokenize_vietnamese(claim)
        if not query:
            raise ValueError("Claim has no indexable tokens")
        # Score the full context before sorting: library top-k does not guarantee
        # sentence-order tie breaking at the cutoff.
        scores = self._index.get_scores(query)
        ranked = sorted(
            range(len(self.sentences)),
            key=lambda i: (-float(scores[i]), self.sentences[i].index),
        )[:top_k]
        return [
            {**asdict(self.sentences[i]), "score": float(scores[i]), "rank": rank}
            for rank, i in enumerate(ranked, start=1)
        ]
