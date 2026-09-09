"""Hybrid Retrieval: BM25 (candidate generation) + SBERT (reranking).

Ported exactly from ViFactCheck/src/retrieval/SBERT_BM25.ipynb.

The exact pipeline from the original notebook (Cell 12):
    1. BM25 retrieves top-10 candidate sentences AND their BM25 scores (weight).
    2. SBERT computes cosine similarity between claim and each candidate.
    3. Scaled cosine similarity is multiplied by BM25 score:
           final_score = scaled_cosine * bm25_weight
    4. Candidates are re-sorted by final_score → return top-5.

This is NOT a plain rerank: the BM25 score is used as a multiplicative weight
on the SBERT similarity, exactly as in the original code.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from factcheck.retrieval.bm25 import BM25Okapi, acc_at_k, f1_at_k, map_at_k
from factcheck.retrieval.sbert import SBERTRetriever, mean_pooling


class HybridRetriever:
    """BM25 + SBERT hybrid retriever ported from SBERT_BM25.ipynb.

    Step-by-step (matches Cell 9 + Cell 12 of original notebook):
        1. evidence_top_n: BM25 → top-10 candidates + their BM25 scores (weight)
        2. similarities:   SBERT cosine → MinMaxScale → multiply by BM25 weight
        3. Sort by combined score → return top-5
    """

    def __init__(self, sbert_retriever: SBERTRetriever, bm25_top_n: int = 10) -> None:
        self.sbert = sbert_retriever
        self.bm25_top_n = bm25_top_n

    # ------------------------------------------------------------------ #
    # Step 1: BM25 candidate generation (Cell 6 of SBERT_BM25.ipynb)     #
    # ------------------------------------------------------------------ #
    def _evidence_top_n(
        self, splited_sentences: list[str], query: str, top_n: int = 10
    ) -> tuple[list[str], list[float]]:
        tokenized_sentences = [str(doc).split(" ") for doc in splited_sentences]
        bm25 = BM25Okapi(tokenized_sentences)
        tokenized_query = query.split(" ")
        top_docs, top_scores = bm25.get_top_n(tokenized_query, splited_sentences, top_n)
        return top_docs, top_scores

    # ------------------------------------------------------------------ #
    # Step 2: SBERT reranking with BM25 weight (Cell 9 of SBERT_BM25)    #
    # ------------------------------------------------------------------ #
    def _similarities(
        self,
        context: list[str],
        claim: str,
        weight: list[float],
        top_n: int = 5,
    ) -> tuple[list[str], list[float], list[str]]:
        """Ported directly from Cell 9 of SBERT_BM25.ipynb.

        Returns:
            top_k  : top-1 evidence (list with 1 item)
            simi   : score of top-1
            top_5  : top-5 reranked evidences
        """
        sentences = [claim] + context

        encoded_input = self.sbert.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.sbert.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.sbert.model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

        sims: list[tuple[str, float]] = []
        claim_embedding = sentence_embeddings[0].unsqueeze(0)
        cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

        for i in range(1, len(sentence_embeddings)):
            evidence_embedding = sentence_embeddings[i].unsqueeze(0)
            similarity = cosine(
                claim_embedding.to(self.sbert.device),
                evidence_embedding.to(self.sbert.device),
            ).item()
            sims.append((sentences[i], similarity))

        # MinMaxScale then multiply by BM25 weight — exact from Cell 9
        simi_values = [s[1] for s in sims]
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(
            np.array(simi_values).reshape(-1, 1)
        ).flatten()
        # KEY LINE: scaled_cosine * bm25_weight
        sims = [
            (sentences[i + 1], float(scaled_values[i]) * weight[i])
            for i in range(len(scaled_values))
        ]

        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = [item[0] for item in sims[:1]]
        simi = [item[1] for item in sims[:1]]
        top_5 = [item[0] for item in sims[:top_n]]
        return top_k, simi, top_5

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #
    def retrieve(
        self, splited_sentences: list[str], claim: str, top_n: int = 5
    ) -> tuple[list[str], list[str]]:
        """Full hybrid pipeline matching Cell 12 of SBERT_BM25.ipynb.

        Returns:
            rank_5      : top-{top_n} sentences after BM25+SBERT reranking
            evidence_top1: the single best sentence (as a list)
        """
        if not splited_sentences:
            return [], []

        # Cell 12: evidence_top_n then similarities
        bm25_top_n = min(self.bm25_top_n, len(splited_sentences))
        evidence_top5, top5_bm25_scores = self._evidence_top_n(
            splited_sentences, claim, top_n=bm25_top_n
        )
        evidence_top1, top1_score, rank_5 = self._similarities(
            evidence_top5, claim, top5_bm25_scores, top_n=top_n
        )
        # Cell 12: evidence_top1 = "".join(evidence_top1)
        evidence_top1_str = "".join(evidence_top1)
        return rank_5, evidence_top1_str

