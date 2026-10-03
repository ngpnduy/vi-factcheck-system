"""BM25 Retrieval module for ViFactCheck.

This module ports the exact BM25Okapi implementation and evaluation metrics
(acc_at_k, map_at_k, f1_at_k) from the original ViFactCheck/src/retrieval/BM25.ipynb
to ensure identical score reproduction.
"""

import math
import re
from multiprocessing import Pool, cpu_count

import numpy as np


class BM25:
    """Base BM25 class ported directly from original repo."""
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size if self.corpus_size > 0 else 0
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        pool.close()
        pool.join()
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), (
            "The documents given don't match the index corpus!"
        )

        scores = self.get_scores(query)
        min_score = np.min(scores)
        max_score = np.max(scores)

        # Scale scores to 0-1 range
        if max_score != min_score:
            scaled_scores = (scores - min_score) / (max_score - min_score)
        else:
            scaled_scores = np.ones(self.corpus_size)

        top_n_indices = np.argsort(scaled_scores)[::-1][:n]
        top_n_scaled_scores = [scaled_scores[i] for i in top_n_indices]

        return [documents[i] for i in top_n_indices], top_n_scaled_scores


class BM25Okapi(BM25):
    """BM25Okapi implementation ported directly from original repo."""
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        
        self.average_idf = idf_sum / len(self.idf) if self.idf else 0

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                q_freq * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score.tolist()


def preprocess_text(text: str) -> str:
    """Preprocess text exactly as in original repo."""
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text


def evidence_top_n(splited_sentences: list[str], query: str, top_n: int = 5):
    """Retrieve top N sentences from a list using BM25Okapi."""
    if not splited_sentences:
        return [], []
        
    tokenized_sentences = [str(doc).split(" ") for doc in splited_sentences]
    bm25 = BM25Okapi(tokenized_sentences)
    tokenized_query = query.split(" ")
    top_docs, top_scores = bm25.get_top_n(tokenized_query, splited_sentences, top_n)
    return top_docs, top_scores


# --- Evaluation Metrics Ported from Original ---
def acc_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    top_k_retrieved = retrieved[:k]
    remaining_gold = list(gold)
    remaining_retrieved = list(top_k_retrieved)
    correct_retrievals = 0

    gold_indices_to_remove = set()
    retrieved_indices_to_remove = set()

    for gold_idx, gold_evidence in enumerate(remaining_gold):
        for retrieved_idx, retrieved_evidence in enumerate(remaining_retrieved):
            if gold_evidence in retrieved_evidence or retrieved_evidence in gold_evidence:
                correct_retrievals += 1
                gold_indices_to_remove.add(gold_idx)
                retrieved_indices_to_remove.add(retrieved_idx)
                break

    acc_k = correct_retrievals / len(gold) if gold else 0
    return acc_k


def map_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    top_k_retrieved = retrieved[:k]
    num_relevant = 0
    precision_at_i_sum = 0.0
    retrieved_gold_indices = set()

    for i, retrieved_evidence in enumerate(top_k_retrieved):
        for gold_idx, gold_evidence in enumerate(gold):
            if gold_idx in retrieved_gold_indices:
                continue
            if gold_evidence in retrieved_evidence or retrieved_evidence in gold_evidence:
                num_relevant += 1
                retrieved_gold_indices.add(gold_idx)
                precision_at_i_sum += num_relevant / (i + 1)
                break

    map_k = precision_at_i_sum / len(gold) if gold else 0
    return map_k


def f1_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    top_k_retrieved = retrieved[:k]
    num_correct_retrievals = 0
    gold_indices_to_remove = set()

    for retrieved_evidence in top_k_retrieved:
        for gold_idx, gold_evidence in enumerate(gold):
            if gold_idx in gold_indices_to_remove:
                continue
            if gold_evidence in retrieved_evidence or retrieved_evidence in gold_evidence:
                num_correct_retrievals += 1
                gold_indices_to_remove.add(gold_idx)
                break

    precision_k = num_correct_retrievals / k if k > 0 else 0
    recall_k = num_correct_retrievals / len(gold) if gold else 0

    if precision_k + recall_k == 0:
        return 0.0
    return 2 * (precision_k * recall_k) / (precision_k + recall_k)
