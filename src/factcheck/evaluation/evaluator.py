"""Standalone evaluator: compare a predictions file against processed gold data.

Usage (CLI)::

    uv run python -m factcheck.evaluation.evaluator \\
        --predictions outputs/phobert_base/test_predictions.jsonl \\
        --gold data/processed/vifactcheck/test.jsonl

Each line in the predictions file must be a JSON object with at least:
    {"id": "vifactcheck:test:123", "prediction": "SUPPORTED"}

Each line in the gold file follows the canonical DatasetRecord schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from factcheck.data.schemas import DatasetRecord, VerdictLabel
from factcheck.evaluation.metrics import compute_metrics


def _load_gold(path: Path) -> dict[str, str]:
    """Load gold labels from a processed JSONL file. Returns {id: label}."""
    gold: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = DatasetRecord.model_validate_json(line)
            gold[record.id] = record.label.value
    return gold


def _load_predictions(path: Path) -> dict[str, str]:
    """Load predictions from a JSONL file. Returns {id: prediction_label}."""
    preds: dict[str, str] = {}
    valid_labels = {v.value for v in VerdictLabel}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record_id = obj.get("id")
            pred = obj.get("prediction")
            if not record_id:
                raise ValueError(f"{path}:{line_no}: missing 'id' field")
            if pred not in valid_labels:
                raise ValueError(
                    f"{path}:{line_no}: invalid prediction {pred!r}. "
                    f"Must be one of {valid_labels}"
                )
            preds[record_id] = pred
    return preds


def evaluate(predictions_path: Path, gold_path: Path) -> dict:
    """Run evaluation and return the metrics dict."""
    gold = _load_gold(gold_path)
    preds = _load_predictions(predictions_path)

    # Only evaluate on IDs present in both files
    common_ids = sorted(set(gold) & set(preds))
    missing_in_preds = set(gold) - set(preds)
    extra_in_preds = set(preds) - set(gold)

    if missing_in_preds:
        print(f"[WARNING] {len(missing_in_preds)} gold IDs not found in predictions.")
    if extra_in_preds:
        print(f"[WARNING] {len(extra_in_preds)} prediction IDs not found in gold (ignored).")

    y_true = [gold[i] for i in common_ids]
    y_pred = [preds[i] for i in common_ids]

    return compute_metrics(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fact-checking predictions")
    parser.add_argument(
        "--predictions", type=Path, required=True,
        help="Path to predictions JSONL (each line: {id, prediction})"
    )
    parser.add_argument(
        "--gold", type=Path, required=True,
        help="Path to gold JSONL (canonical DatasetRecord format)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to save the metrics JSON result"
    )
    args = parser.parse_args()

    result = evaluate(args.predictions, args.gold)

    print("\n=== Evaluation Results ===")
    print(f"  Samples evaluated : {result['n_samples']}")
    print(f"  Macro F1          : {result['macro_f1']:.4f}")
    print(f"  Macro Precision   : {result['macro_precision']:.4f}")
    print(f"  Macro Recall      : {result['macro_recall']:.4f}")
    print(f"  Accuracy          : {result['accuracy']:.4f}")
    print("\n  Per-class:")
    for label, stats in result["per_class"].items():
        print(f"    {label:<20} P={stats['precision']:.4f}  R={stats['recall']:.4f}  F1={stats['f1']:.4f}  n={stats['support']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
