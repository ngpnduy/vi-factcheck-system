from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import LABELS


def classification_metrics(labels, predictions):
    if not labels:
        raise ValueError("Cannot evaluate an empty split")
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=[0, 1, 2], zero_division=0)
    return {"count": len(labels), "accuracy": float(accuracy_score(labels, predictions)), "macro_f1": float(np.mean(f1)), "per_label": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(LABELS)}, "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(), "label_order": list(LABELS)}


def retrieval_metrics(rows, rankings, ks=(5, 20)):
    eligible = [(row, ranking) for row, ranking in zip(rows, rankings) if row["label"] != 2 and row["alignment"]["status"] == "valid" and row["alignment"]["gold_sentence_ids"]]
    result = {"count": len(eligible), "alignment_coverage": sum(r["alignment"]["status"] == "valid" for r in rows) / len(rows) if rows else 0.0, "alignment_status_counts": dict(Counter(r["alignment"]["status"] for r in rows))}
    mrr = []
    for k in ks:
        hits, recalls, complete = [], [], []
        for row, ranking in eligible:
            gold, found = set(row["alignment"]["gold_sentence_ids"]), set(ranking[:k])
            hits.append(float(bool(gold & found)))
            recalls.append(len(gold & found) / len(gold))
            complete.append(float(gold <= found))
        result[f"hit@{k}"] = float(np.mean(hits)) if hits else None
        result[f"sentence_recall@{k}"] = float(np.mean(recalls)) if recalls else None
        result[f"complete_gold@{k}"] = float(np.mean(complete)) if complete else None
    for row, ranking in eligible:
        gold = set(row["alignment"]["gold_sentence_ids"])
        mrr.append(next((1 / (i + 1) for i, sentence in enumerate(ranking) if sentence in gold), 0.0))
    result["mrr"] = float(np.mean(mrr)) if mrr else None
    return result


def error_stage(row, ranking, predicted):
    if predicted == row["label"]:
        return "correct"
    if row["label"] == 2:
        return "nei_verification_error"
    if row["alignment"]["status"] != "valid":
        return "alignment_unavailable"
    gold = set(row["alignment"]["gold_sentence_ids"])
    if not gold <= set(row["candidate_ids"]):
        return "bm25_incomplete_gold"
    if not gold <= set(ranking[:5]):
        return "selection_incomplete_gold"
    return "verification_with_complete_gold"
