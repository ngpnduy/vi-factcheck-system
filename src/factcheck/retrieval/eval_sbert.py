"""Evaluate SBERT or Hybrid retrieval against the gold test set.

Usage::

    # SBERT only
    uv run python -m factcheck.retrieval.eval_sbert \\
        --method sbert \\
        --gold data/processed/vifactcheck/test.jsonl

    # Hybrid (BM25 + SBERT reranking)
    uv run python -m factcheck.retrieval.eval_sbert \\
        --method hybrid \\
        --bm25-candidates 20 \\
        --gold data/processed/vifactcheck/test.jsonl

NOTE: Requires a GPU for reasonable speed. CPU works but is very slow.
"""

import argparse
from pathlib import Path

import nltk
from tqdm import tqdm

from factcheck.data.schemas import DatasetRecord
from factcheck.retrieval.bm25 import acc_at_k, f1_at_k, map_at_k
from factcheck.retrieval.hybrid import HybridRetriever
from factcheck.retrieval.sbert import SBERTRetriever

# Install sentence resources explicitly: python -m nltk.downloader punkt_tab


def evaluate(data_path: Path, method: str, top_k: int, bm25_candidates: int) -> None:
    # Load SBERT once (expensive)
    sbert = SBERTRetriever()
    retriever = HybridRetriever(sbert, bm25_top_n=bm25_candidates) if method == "hybrid" else sbert

    all_acc, all_map, all_f1 = [], [], []

    with data_path.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc=f"Evaluating {method.upper()} Top-{top_k}"):
        record = DatasetRecord.model_validate_json(line)
        gold_evidence = [e.text for e in record.evidence] if record.evidence else []
        if not gold_evidence or not record.context:
            continue

        splited_sentences = nltk.tokenize.sent_tokenize(record.context)

        if method == "hybrid":
            retrieved, _ = retriever.retrieve(splited_sentences, record.claim, top_n=top_k)
        else:
            retrieved, _ = sbert.similarities(splited_sentences, record.claim, top_n=top_k)

        all_acc.append(acc_at_k(retrieved, gold_evidence, top_k))
        all_map.append(map_at_k(retrieved, gold_evidence, top_k))
        all_f1.append(f1_at_k(retrieved, gold_evidence, top_k))

    n = len(all_acc)
    print(f"\n=== {method.upper()} Retrieval Results (Top-{top_k}) ===")
    print(f"Evaluated {n} samples.")
    if not n:
        print("No records with gold evidence and context to evaluate.")
        return
    print(f"ACC@{top_k} : {sum(all_acc)/n:.4f}")
    print(f"mAP@{top_k} : {sum(all_map)/n:.4f}")
    print(f"F1@{top_k}  : {sum(all_f1)/n:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="sbert", choices=["sbert", "hybrid"])
    parser.add_argument("--gold", type=Path, default=Path("data/processed/vifactcheck/test.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bm25-candidates", type=int, default=20,
                        help="Number of BM25 candidates before SBERT reranking (hybrid only)")
    args = parser.parse_args()
    evaluate(args.gold, args.method, args.top_k, args.bm25_candidates)
