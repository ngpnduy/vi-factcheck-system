"""Metric computation for fact-checking classification.

Mirrors the metric used by the original ViFactCheck paper:
    sklearn classification_report → Macro-F1, Precision, Recall per class.

Usage::

    from factcheck.evaluation.metrics import compute_metrics

    result = compute_metrics(y_true=["SUPPORTED", "REFUTED"], y_pred=["SUPPORTED", "SUPPORTED"])
    print(result["macro_f1"])
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import classification_report

# Canonical label order (matches paper Table 3 column order)
LABEL_NAMES = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    label_names: list[str] = LABEL_NAMES,
) -> dict[str, Any]:
    """Compute classification metrics matching the ViFactCheck paper.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.
        label_names: Canonical label order for the report.

    Returns:
        A dict with keys: macro_f1, macro_precision, macro_recall,
        accuracy, and a per_class dict with per-label metrics.
    """
    report: dict[str, Any] = classification_report(
        y_true,
        y_pred,
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float]] = {}
    for label in label_names:
        if label in report:
            per_class[label] = {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }

    macro = report.get("macro avg", {})
    return {
        "macro_f1": round(macro.get("f1-score", 0.0), 4),
        "macro_precision": round(macro.get("precision", 0.0), 4),
        "macro_recall": round(macro.get("recall", 0.0), 4),
        "accuracy": round(report.get("accuracy", 0.0), 4),
        "per_class": per_class,
        "n_samples": len(y_true),
    }
