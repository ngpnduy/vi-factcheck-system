"""Evaluate M0 predictions and sentence retrieval using separate gold targets.

Verdict metrics include all three labels, including classes absent from a split.
Retrieval metrics use only SUPPORTED/REFUTED claims with fully aligned, nonempty
gold evidence. They are macro averages over those claims, not over sentences.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LABELS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")
ALIGNMENT_STATUSES = frozenset(
    {"aligned", "ambiguous", "unmatched", "partial", "no_gold", "not_applicable"}
)


class EvaluationError(ValueError):
    """Raised when incomplete or inconsistent artifacts would distort metrics."""


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    if not rows:
        raise EvaluationError(f"{name} must not be empty")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise EvaluationError(f"Each {name} row must be an object")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise EvaluationError(f"Each {name} row must have a nonempty string id")
        if row_id in result:
            raise EvaluationError(f"Duplicate {name} id: {row_id!r}")
        result[row_id] = row
    return result


def _same_ids(reference: dict, actual: dict, name: str) -> None:
    missing = sorted(reference.keys() - actual.keys())
    extra = sorted(actual.keys() - reference.keys())
    if missing or extra:
        raise EvaluationError(
            f"{name} IDs must exactly match target IDs; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )


def _label(row: dict[str, Any], name: str) -> str:
    label = row.get("label")
    if label not in LABELS:
        raise EvaluationError(f"Invalid {name} label for {row['id']!r}: {label!r}")
    return label


def _sentence_ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise EvaluationError(f"{name} must be a list of nonempty string IDs")
    if len(set(value)) != len(value):
        raise EvaluationError(f"Duplicate sentence IDs in {name}")
    return value


def verdict_metrics(
    targets: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute strict, claim-level three-class metrics.

    Probability order is always SUPPORTED, REFUTED, NOT_ENOUGH_INFO. Ties use
    the first maximum in that order, matching torch.argmax. Missing/extra or
    duplicate predictions are errors rather than silently excluded examples.
    """
    target_by_id = _index(targets, "target")
    prediction_by_id = _index(predictions, "prediction")
    _same_ids(target_by_id, prediction_by_id, "Prediction")
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for row_id, target in target_by_id.items():
        gold_label = _label(target, "target")
        prediction = prediction_by_id[row_id]
        predicted_label = _label(prediction, "prediction")
        probabilities = prediction.get("probabilities")
        if (
            not isinstance(probabilities, list)
            or len(probabilities) != len(LABELS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
                for value in probabilities
            )
        ):
            raise EvaluationError(
                f"Probabilities for {row_id!r} must contain three finite numbers in [0, 1]"
            )
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise EvaluationError(f"Probabilities for {row_id!r} must sum to 1")
        argmax = max(range(len(LABELS)), key=probabilities.__getitem__)
        if predicted_label != LABELS[argmax]:
            raise EvaluationError(f"Prediction label for {row_id!r} disagrees with argmax")
        if "sentence_ids" in prediction:
            _sentence_ids(prediction["sentence_ids"], f"prediction {row_id!r}")
        matrix[LABELS.index(gold_label)][argmax] += 1

    count = len(target_by_id)
    per_class: dict[str, dict[str, int | float]] = {}
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * true_positive / (support + predicted_count) if support + predicted_count else 0.0
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted_count,
            "one_vs_rest_accuracy": (count - false_positive - false_negative) / count,
        }
    return {
        "count": count,
        "labels": list(LABELS),
        "accuracy": sum(matrix[index][index] for index in range(len(LABELS))) / count,
        "macro_f1": sum(values["f1"] for values in per_class.values()) / len(LABELS),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_axes": {"rows": "gold", "columns": "predicted"},
        "argmax_tie_break": "first label in labels order",
    }


def retrieval_metrics(
    inputs: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    ks: Sequence[int] = (5, 20),
) -> dict[str, Any]:
    """Measure ranked sentence retrieval on fully aligned S/R claims.

    The input candidate list must already be ranked. Gold sentence IDs may be
    outside that list: these are retrieval misses and remain in the denominator.
    NEI claims never enter evidence retrieval denominators. No eligible claims
    produces null metrics, not misleading zero scores.
    """
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise EvaluationError("ks must contain positive integers")
    ks = tuple(sorted(set(ks)))
    target_by_id = _index(targets, "target")
    input_by_id = _index(inputs, "input")
    _same_ids(target_by_id, input_by_id, "Input")
    sr_count = 0
    evaluable = 0
    status_counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    aggregates = {
        k: {"hit_count": 0, "recall_sum": 0.0, "reciprocal_rank_sum": 0.0, "complete_count": 0}
        for k in ks
    }
    for row_id, target in target_by_id.items():
        label = _label(target, "target")
        status = target.get("alignment_status")
        if not isinstance(status, str) or status not in ALIGNMENT_STATUSES:
            raise EvaluationError(f"Invalid alignment_status for {row_id!r}: {status!r}")
        gold = set(_sentence_ids(target.get("gold_sentence_ids"), f"gold {row_id!r}"))
        query = input_by_id[row_id]
        candidates = query.get("candidates")
        if not isinstance(candidates, list) or any(
            not isinstance(candidate, dict) for candidate in candidates
        ):
            raise EvaluationError(f"Candidates for {row_id!r} must be a list of objects")
        ranked_ids = _sentence_ids(
            [candidate.get("sentence_id") for candidate in candidates], f"candidates {row_id!r}"
        )
        sentence_count = query.get("sentence_count")
        if (
            isinstance(sentence_count, bool)
            or not isinstance(sentence_count, int)
            or sentence_count < len(ranked_ids)
            or sentence_count < len(gold)
        ):
            raise EvaluationError(f"Invalid sentence_count for {row_id!r}")
        if len(ranked_ids) < min(max(ks), sentence_count):
            raise EvaluationError(
                f"Not enough ranked candidates for {row_id!r} to evaluate k={max(ks)}"
            )
        if label == "NOT_ENOUGH_INFO":
            excluded["not_enough_info"] += 1
            continue
        sr_count += 1
        status_counts[status] += 1
        if status != "aligned":
            excluded[f"alignment_{status}"] += 1
            continue
        if not gold:
            excluded["aligned_empty_gold"] += 1
            continue
        evaluable += 1
        for k in ks:
            ranks = [
                rank
                for rank, sentence_id in enumerate(ranked_ids[:k], start=1)
                if sentence_id in gold
            ]
            aggregates[k]["hit_count"] += int(bool(ranks))
            aggregates[k]["recall_sum"] += len(ranks) / len(gold)
            aggregates[k]["reciprocal_rank_sum"] += 1 / ranks[0] if ranks else 0.0
            aggregates[k]["complete_count"] += int(len(ranks) == len(gold))
    at_k = {}
    for k, aggregate in aggregates.items():
        at_k[str(k)] = {
            "hit": aggregate["hit_count"] / evaluable if evaluable else None,
            "macro_recall": aggregate["recall_sum"] / evaluable if evaluable else None,
            "mrr": aggregate["reciprocal_rank_sum"] / evaluable if evaluable else None,
            "complete_gold": aggregate["complete_count"] / evaluable if evaluable else None,
            "hit_count": aggregate["hit_count"],
            "complete_gold_count": aggregate["complete_count"],
        }
    return {
        "count": len(target_by_id),
        "supported_refuted_count": sr_count,
        "evaluable_count": evaluable,
        "alignment_coverage": evaluable / sr_count if sr_count else None,
        "alignment_status_counts": dict(sorted(status_counts.items())),
        "alignment_status_counts_scope": "SUPPORTED and REFUTED only",
        "excluded_by_reason": dict(sorted(excluded.items())),
        "metric_denominator": "fully aligned SUPPORTED/REFUTED claims with nonempty gold",
        "at_k": at_k,
    }


def summarize_seed_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize verdict accuracy/Macro-F1; sample std is null for a single run.

    Callers must pass comparable runs on the same fixed evaluation split. This
    helper does not substitute for paired significance tests against M1/M2.
    """
    if not reports:
        raise EvaluationError("At least one seed report is required")
    metrics = [report.get("verdict", report) for report in reports]
    counts = {report.get("count") for report in metrics}
    if len(counts) != 1 or None in counts:
        raise EvaluationError("Seed reports must have the same evaluation count")
    summary: dict[str, Any] = {"runs": len(metrics), "count_per_run": counts.pop(), "std_ddof": 1}
    for key in ("accuracy", "macro_f1"):
        values = [report.get(key) for report in metrics]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in values
        ):
            raise EvaluationError(f"Invalid {key} in seed reports")
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "values": values,
        }
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise EvaluationError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def evaluate_m0(
    predictions_path: Path,
    targets_path: Path,
    output_path: Path,
    inputs_path: Path | None = None,
) -> dict[str, Any]:
    targets = _read_jsonl(targets_path)
    report: dict[str, Any] = {
        "kind": "m0_metrics",
        "verdict": verdict_metrics(targets, _read_jsonl(predictions_path)),
    }
    if inputs_path is not None:
        report["retrieval"] = retrieval_metrics(_read_jsonl(inputs_path), targets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate M0 verdict and sentence retrieval")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Optional ranked, gold-free prepared inputs")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_m0(args.predictions, args.targets, args.output, args.inputs)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
