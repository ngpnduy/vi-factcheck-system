"""Evaluate BM25 retrieval against the gold test set.

Usage::

    uv run python -m factcheck.retrieval.eval_bm25 \\
        --gold data/processed/vifactcheck/test.jsonl

Reads the context, splits it into sentences, runs BM25 to get Top-K evidence,
and calculates ACC@K, mAP@K, and F1@K.
"""

import argparse
import json
from pathlib import Path
import nltk
from tqdm import tqdm

from factcheck.data.schemas import DatasetRecord
from factcheck.retrieval.bm25 import evidence_top_n, acc_at_k, map_at_k, f1_at_k

# Ensure NLTK punkt is available for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def evaluate_retrieval(data_path: Path, top_k: int = 5):
    all_acc = []
    all_map = []
    all_f1 = []

    with data_path.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for line in tqdm(lines, desc=f"Evaluating BM25 Top-{top_k}"):
        record = DatasetRecord.model_validate_json(line)
        
        # Ground truth evidence list
        gold_evidence = [e.text for e in record.evidence] if record.evidence else []
        if not gold_evidence:
            continue # Skip records with no gold evidence to evaluate against
            
        # Context to search in
        context = record.context or ""
        if not context:
            all_acc.append(0.0)
            all_map.append(0.0)
            all_f1.append(0.0)
            continue
            
        # Split context into sentences
        splited_sentences = nltk.tokenize.sent_tokenize(context)
        
        # Retrieve
        retrieved_evidence, _ = evidence_top_n(splited_sentences, record.claim, top_n=top_k)
        
        # Evaluate
        all_acc.append(acc_at_k(retrieved_evidence, gold_evidence, top_k))
        all_map.append(map_at_k(retrieved_evidence, gold_evidence, top_k))
        all_f1.append(f1_at_k(retrieved_evidence, gold_evidence, top_k))
        
    print(f"\n=== BM25 Retrieval Results (Top-{top_k}) ===")
    print(f"Evaluated {len(all_acc)} samples.")
    print(f"ACC@{top_k} : {sum(all_acc) / len(all_acc):.4f}")
    print(f"mAP@{top_k} : {sum(all_map) / len(all_map):.4f}")
    print(f"F1@{top_k}  : {sum(all_f1) / len(all_f1):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("data/processed/vifactcheck/test.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    
    evaluate_retrieval(args.gold, args.top_k)
