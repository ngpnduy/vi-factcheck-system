from collections import Counter
from types import SimpleNamespace

import pytest

from vifactcheck.config import TrainConfig
from vifactcheck.data import stratified_subset
from vifactcheck.pilot import require_space
from vifactcheck.reporting import paired_article_bootstrap


def test_stratified_subset_does_not_filter_unaligned_claims():
    rows = [{"id": f"{label}-{i}", "label": label, "alignment": {"status": "unmatched"}} for label in range(3) for i in range(5)]
    selected = stratified_subset(rows, 3, 42)
    assert Counter(r["label"] for r in selected) == {0: 3, 1: 3, 2: 3}
    assert selected == stratified_subset(rows, 3, 42)
    assert len({r["id"] for r in selected}) == 9
    with pytest.raises(ValueError):
        stratified_subset(rows, 6, 42)


def test_old_configuration_normalizes_to_no_subset():
    old = TrainConfig().to_dict()
    old.pop("train_per_label")
    assert TrainConfig(**old).to_dict() == TrainConfig().to_dict()
    with pytest.raises(ValueError):
        TrainConfig(smoke=True, train_per_label=300)


def prediction(claim_id, article, gold, predicted):
    return {"id": claim_id, "article_id": article, "gold_label_id": gold, "label_id": predicted}


def test_article_bootstrap_hand_calculation_and_pairing():
    left = [prediction("a1", "a", 0, 0), prediction("a2", "a", 0, 0), prediction("b", "b", 1, 1), prediction("c", "c", 2, 2)]
    right = [{**r, "label_id": 1 if r["article_id"] == "a" else r["label_id"]} for r in left]
    result = paired_article_bootstrap(left, list(reversed(right)))
    assert result["article_count"] == 3 and result["claim_count"] == 4
    assert result["metrics"]["macro_f1"]["difference"] == pytest.approx(0.5)
    assert result["metrics"]["macro_f1"]["ci95"] == pytest.approx([0, 5/9])
    assert result["metrics"]["refuted_f1"]["ci95"] == pytest.approx([0, 2/3])
    identical = paired_article_bootstrap(left, left, iterations=30)
    assert identical["metrics"]["macro_f1"]["ci95"] == [0, 0]
    assert result == paired_article_bootstrap(left, right)
    with pytest.raises(ValueError):
        paired_article_bootstrap(left, right[:-1])
    with pytest.raises(ValueError):
        paired_article_bootstrap(left + left[:1], right)
    with pytest.raises(ValueError):
        paired_article_bootstrap(left, [{**right[0], "article_id": "other"}] + right[1:])


def test_storage_gate_includes_temporary_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr("vifactcheck.pilot.shutil.disk_usage", lambda path: SimpleNamespace(free=3*1024**3))
    assert require_space(tmp_path, 1024**3) == 3*1024**3
    with pytest.raises(RuntimeError, match="Insufficient checkpoint space"):
        require_space(tmp_path, 1024**3 + 1)


def test_report_generation_contains_all_models_and_single_seed_limits(tmp_path):
    from vifactcheck import LABELS
    from vifactcheck.io import write_json, write_jsonl
    from vifactcheck.metrics import classification_metrics
    from vifactcheck.reporting import write_pilot_report
    rows = [{**prediction(str(i), f"article-{i//3}", i % 3, i % 3), "label": LABELS[i % 3], "claim": "Test claim.", "selected_sentences": [{"text": "Test evidence.", "weight": 1.0}]} for i in range(1447)]
    classification = classification_metrics([r["gold_label_id"] for r in rows], [r["label_id"] for r in rows])
    retrieval = {"count": 539, "alignment_status_counts": {"valid": 857, "unmatched": 579, "ambiguous": 11}, "mrr": 1.0}
    for k in (5, 20):
        for key in ("hit", "sentence_recall", "complete_gold"):
            retrieval[f"{key}@{k}"] = 1.0
    metrics = {"classification": classification, "retrieval": retrieval, "cost": {"prepared_input_seconds_per_claim": 0.1}, "error_stages": {"correct": 1447}, "truncation": {"claim_truncated_count": 0, "claim_count": 1447, "candidate_sentence_truncated_count": 0, "candidate_pair_count": 1447, "selected_sentence_truncated_count": 0, "selected_pair_count": 1447}}
    write_json(tmp_path/"pilot_spec.json", {"train_evidence_supervised_count": 331, "subset_article_overlap": {"train-test": 20}})
    write_json(tmp_path/"experiment.json", {"subset_ids": {"train": list(map(str, range(900)))}, "data_manifest": {"splits": {"test": {"valid_count": 1447}}, "assets": {"dataset_revision": "fixture", "model_revision": "fixture"}}, "hardware": {"gpu": "test fixture"}})
    write_json(tmp_path/"summary.json", {m: {"metrics": metrics} for m in ("M0", "M1", "M2")})
    for model in ("m0", "m1", "m2"):
        write_json(tmp_path/f"{model}_test_metrics.json", metrics)
        write_jsonl(tmp_path/f"{model}_test_predictions.jsonl", rows)
    for phase in ("reranker", "warmup", "m0", "m1", "m2"):
        write_json(tmp_path/f"{phase}_history.json", [{"training_seconds": 1, "peak_training_cuda": {"allocated_mib": 1, "reserved_mib": 2}}])
    report = write_pilot_report(tmp_path)
    assert len(report["models"]) == 3
    text = (tmp_path/"REPORT.md").read_text(encoding="utf8")
    assert "single-seed pilot" in text
    assert "do not measure variability across training seeds" in text
    assert "| M2 |" in text and "Test claim." in text and "Test evidence." in text
    assert (tmp_path/"results.csv").exists() and (tmp_path/"paired_bootstrap.json").exists()
