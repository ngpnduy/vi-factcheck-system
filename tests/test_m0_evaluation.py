from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from factcheck.evaluation.evaluate_m0 import (
    LABELS,
    EvaluationError,
    evaluate_m0,
    retrieval_metrics,
    summarize_seed_metrics,
    verdict_metrics,
)


def _target(
    row_id: str,
    label: str = "SUPPORTED",
    gold: list[str] | None = None,
    status: str = "aligned",
) -> dict:
    return {
        "id": row_id,
        "label": label,
        "gold_sentence_ids": ["s1"] if gold is None else gold,
        "alignment_status": status,
    }


def _prediction(row_id: str, label: str = "SUPPORTED") -> dict:
    probabilities = [0.0, 0.0, 0.0]
    probabilities[LABELS.index(label)] = 1.0
    return {"id": row_id, "label": label, "probabilities": probabilities}


def _input(row_id: str, ids: list[str], sentence_count: int | None = None) -> dict:
    return {
        "id": row_id,
        "candidates": [{"sentence_id": sentence_id} for sentence_id in ids],
        "sentence_count": len(ids) if sentence_count is None else sentence_count,
    }


def test_verdict_metrics_align_by_id_and_count_absent_class_in_macro_f1() -> None:
    targets = [_target("a"), _target("b", "REFUTED"), _target("c")]
    predictions = [_prediction("c"), _prediction("b", "SUPPORTED"), _prediction("a")]
    metrics = verdict_metrics(targets, predictions)
    assert metrics["labels"] == list(LABELS)
    assert metrics["confusion_matrix"] == [[2, 0, 0], [1, 0, 0], [0, 0, 0]]
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["SUPPORTED"]["f1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx(0.8 / 3)
    assert metrics["per_class"]["NOT_ENOUGH_INFO"]["support"] == 0
    assert metrics["per_class"]["NOT_ENOUGH_INFO"]["f1"] == 0.0


def test_verdict_ties_choose_first_label_in_fixed_order() -> None:
    target = [_target("a")]
    prediction = {"id": "a", "label": "SUPPORTED", "probabilities": [0.5, 0.5, 0]}
    assert verdict_metrics(target, [prediction])["accuracy"] == 1.0
    prediction["label"] = "REFUTED"
    with pytest.raises(EvaluationError, match="argmax"):
        verdict_metrics(target, [prediction])


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5, 0.5],
        [0.2, 0.2, 0.2],
        [math.nan, 0.0, 0.0],
        [math.inf, 0.0, 0.0],
        [1.1, -0.1, 0.0],
        [True, 0.0, 0.0],
        ["1.0", 0.0, 0.0],
    ],
)
def test_invalid_probabilities_fail_loudly(probabilities: list) -> None:
    prediction = {"id": "a", "label": "SUPPORTED", "probabilities": probabilities}
    with pytest.raises(EvaluationError, match="Probabilities"):
        verdict_metrics([_target("a")], [prediction])


@pytest.mark.parametrize(
    "targets,predictions,error",
    [
        ([], [], "empty"),
        ([_target("a")], [_prediction("b")], "exactly match"),
        ([_target("a"), _target("b")], [_prediction("a")], "exactly match"),
        ([_target("a")], [_prediction("a"), _prediction("b")], "exactly match"),
        ([_target("a"), _target("a")], [_prediction("a")], "Duplicate target"),
        ([_target("a")], [_prediction("a"), _prediction("a")], "Duplicate prediction"),
        ([_target("a", "bad")], [_prediction("a")], "Invalid target label"),
    ],
)
def test_incomplete_verdict_artifacts_are_rejected(
    targets: list[dict], predictions: list[dict], error: str
) -> None:
    with pytest.raises(EvaluationError, match=error):
        verdict_metrics(targets, predictions)


def test_retrieval_macro_recall_is_distinct_from_hit_and_gold_coverage() -> None:
    targets = [
        _target("a", gold=["s2", "s5"]),
        _target("b", "REFUTED", gold=["s1"]),
        _target("c", "REFUTED", gold=[], status="unmatched"),
        _target("d", gold=["s1"], status="ambiguous"),
        _target("e", "NOT_ENOUGH_INFO", gold=[], status="not_applicable"),
    ]
    inputs = [_input(target["id"], ["s1", "s2", "s3"], 5) for target in targets]
    metrics = retrieval_metrics(inputs, targets, ks=(1, 3))
    assert metrics["count"] == 5
    assert metrics["supported_refuted_count"] == 4
    assert metrics["evaluable_count"] == 2
    assert metrics["alignment_coverage"] == 0.5
    assert metrics["excluded_by_reason"] == {
        "alignment_ambiguous": 1,
        "alignment_unmatched": 1,
        "not_enough_info": 1,
    }
    assert metrics["at_k"]["1"]["hit"] == 0.5
    assert metrics["at_k"]["1"]["macro_recall"] == 0.5
    assert metrics["at_k"]["3"]["hit"] == 1.0
    assert metrics["at_k"]["3"]["macro_recall"] == 0.75
    assert metrics["at_k"]["3"]["mrr"] == 0.75
    assert metrics["at_k"]["3"]["complete_gold"] == 0.5


def test_retrieval_uses_actual_short_context_without_padding() -> None:
    metrics = retrieval_metrics([_input("a", ["s1"])], [_target("a")])
    assert metrics["at_k"]["5"]["macro_recall"] == 1.0
    assert metrics["at_k"]["20"]["complete_gold"] == 1.0


def test_no_evaluable_gold_yields_null_scores_with_explicit_denominators() -> None:
    targets = [
        _target("a", gold=[], status="no_gold"),
        _target("b", "REFUTED", gold=[], status="aligned"),
        _target("c", "NOT_ENOUGH_INFO", gold=[], status="not_applicable"),
    ]
    inputs = [_input(target["id"], ["s1"]) for target in targets]
    metrics = retrieval_metrics(inputs, targets)
    assert metrics["alignment_coverage"] == 0.0
    assert metrics["evaluable_count"] == 0
    assert metrics["at_k"]["5"]["hit"] is None
    assert metrics["excluded_by_reason"]["aligned_empty_gold"] == 1
    nei_only = retrieval_metrics(inputs[2:], targets[2:])
    assert nei_only["alignment_coverage"] is None


@pytest.mark.parametrize(
    "inputs,targets,error",
    [
        ([_input("a", ["s1"])], [_target("b")], "exactly match"),
        ([_input("a", ["s1", "s1"])], [_target("a")], "Duplicate sentence"),
        ([_input("a", ["s1"])], [_target("a", gold=["s1", "s1"])], "Duplicate sentence"),
        ([_input("a", ["s1"], 30)], [_target("a")], "Not enough ranked"),
        ([_input("a", ["s1"], 0)], [_target("a")], "Invalid sentence_count"),
        ([_input("a", ["s1"])], [_target("a", status="typo")], "alignment_status"),
    ],
)
def test_retrieval_rejects_inconsistent_artifacts(
    inputs: list[dict], targets: list[dict], error: str
) -> None:
    with pytest.raises(EvaluationError, match=error):
        retrieval_metrics(inputs, targets)


def test_seed_summary_uses_sample_standard_deviation() -> None:
    reports = [
        {"verdict": {"count": 3, "accuracy": 0.5, "macro_f1": 0.4}},
        {"verdict": {"count": 3, "accuracy": 0.7, "macro_f1": 0.6}},
    ]
    summary = summarize_seed_metrics(reports)
    assert summary["accuracy"]["mean"] == pytest.approx(0.6)
    assert summary["accuracy"]["std"] == pytest.approx(math.sqrt(0.02))
    assert summary["std_ddof"] == 1
    assert summarize_seed_metrics(reports[:1])["macro_f1"]["std"] is None


def test_file_evaluation_writes_verdict_and_retrieval_report(tmp_path: Path) -> None:
    targets_path = tmp_path / "validation.targets.jsonl"
    inputs_path = tmp_path / "validation.inputs.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    targets_path.write_text(json.dumps(_target("a")) + "\n", encoding="utf-8")
    inputs_path.write_text(json.dumps(_input("a", ["s1"])) + "\n", encoding="utf-8")
    predictions_path.write_text(json.dumps(_prediction("a")) + "\n", encoding="utf-8")
    output_path = tmp_path / "new_directory" / "metrics.json"
    metrics = evaluate_m0(predictions_path, targets_path, output_path, inputs_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == metrics
    assert metrics["verdict"]["accuracy"] == 1.0
    assert metrics["retrieval"]["evaluable_count"] == 1
