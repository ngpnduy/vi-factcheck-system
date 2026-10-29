from pathlib import Path

import pytest

from vifactcheck.full import (checkpoint_cleanup, exclusive_run, seal_training,
                             select_trial, training_job, verify_completed_training,
                             verify_initialization)
from vifactcheck.full_reporting import mean_std, write_full_report
from vifactcheck.io import read_json, write_json, write_jsonl


def test_tuning_ties_use_search_order_and_never_test_scores():
    trials = [{"learning_rate": 1e-5, "validation_macro_f1": .5, "test_macro_f1": .1},
              {"learning_rate": 2e-5, "validation_macro_f1": .5, "test_macro_f1": .9}]
    assert select_trial(trials, "learning_rate") == 1e-5


def test_run_lock_rejects_duplicate_writer_and_releases(tmp_path):
    with exclusive_run(tmp_path):
        with pytest.raises((RuntimeError, BlockingIOError)):
            with exclusive_run(tmp_path):
                pass
    with exclusive_run(tmp_path):
        pass


@pytest.mark.parametrize("unchanged_component", [None, "verifier", "reranker"])
def test_joint_checks_use_full_checkpoint_names_and_require_both_encoders(monkeypatch, unchanged_component):
    from copy import deepcopy
    import torch
    from torch import nn
    from vifactcheck.full import joint_checks

    system = nn.Module()
    for component in ("verifier", "reranker"):
        pair = nn.Module()
        pair.encoder = nn.Module()
        pair.encoder.encoder = nn.Module()
        pair.encoder.encoder.layer = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.attention = nn.Module()
            setattr(layer.attention, "self", nn.Module())
            getattr(layer.attention, "self").query = nn.Linear(2, 2, bias=False)
            pair.encoder.encoder.layer.append(layer)
        setattr(system, component, pair)
    warmup = deepcopy(system.state_dict())
    with torch.no_grad():
        for name, parameter in system.named_parameters():
            if not unchanged_component or not name.startswith(f"{unchanged_component}."):
                parameter.add_(1)
    monkeypatch.setattr("vifactcheck.full.system_from_checkpoint", lambda *a: system)
    monkeypatch.setattr("vifactcheck.full.isolated_joint_gradient", lambda *a: {"gradient_fixture": True})
    monkeypatch.setattr("vifactcheck.full.load_checkpoint", lambda *a: {"model": warmup})
    monkeypatch.setattr("vifactcheck.full.release_gpu", lambda: None)

    if unchanged_component:
        with pytest.raises(AssertionError, match="both encoders"):
            joint_checks("m2.pt", "warmup.pt", {})
    else:
        checks = joint_checks("m2.pt", "warmup.pt", {})
        changed = checks["first_last_attention_query_changed"]
        assert len(changed) == 4 and all(changed.values())
        assert "verifier.encoder.layer.0.attention.self.query.weight" in changed
        assert checks["gradient_fixture"]


def test_completed_job_skips_training_but_rejects_corruption(tmp_path, monkeypatch):
    from vifactcheck.config import TrainConfig
    write_json(tmp_path / "summary.json", {"M0": {"checkpoint": str(tmp_path / "m0_best.pt")}})
    write_json(tmp_path / "experiment.json", {"config": "fixture"})
    for name in ("m0_dev_metrics.json", "m0_checks.json"):
        write_json(tmp_path / name, {})
    write_jsonl(tmp_path / "m0_dev_predictions.jsonl", [])
    (tmp_path / "m0_best.pt").write_bytes(b"trusted fixture weights")
    (tmp_path / "m0_last.pt").write_bytes(b"completed resume fixture")
    seal_training(tmp_path, {"M0": {}})
    monkeypatch.setattr("vifactcheck.full.run_experiment", lambda *a, **k: pytest.fail("Completed job was retrained"))
    assert "M0" in training_job(TrainConfig(device="cpu"), tmp_path, ("M0",), None, [])
    assert not (tmp_path / "m0_last.pt").exists()
    (tmp_path / "m0_best.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        verify_completed_training(tmp_path)


def test_cleanup_requires_ledger_and_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="ledger"):
        checkpoint_cleanup(tmp_path, ["m0_last.pt"])
    write_json(tmp_path / "training_complete.json", {})
    with pytest.raises(ValueError, match="basename"):
        checkpoint_cleanup(tmp_path, ["../unrelated.pt"])


def test_initialization_and_three_seed_statistics():
    summary = {m: {"checks": {"verifier_initial_sha256": "initial", "shared_warmup_checkpoint_sha256": "warm", "reranker_unchanged": m == "M1"}} for m in ("M0", "M1", "M2")}
    verify_initialization(summary)
    summary["M2"]["checks"]["shared_warmup_checkpoint_sha256"] = "different"
    with pytest.raises(AssertionError, match="initialization checkpoint"):
        verify_initialization(summary)
    result = mean_std([.4, .5, .6])
    assert result["mean"] == pytest.approx(.5)
    assert result["sample_std"] == pytest.approx(.1)
    with pytest.raises(ValueError):
        mean_std([.4, .5])


@pytest.mark.parametrize("include_ablations", [False, True])
def test_full_report_requires_all_complete_paired_models(tmp_path, monkeypatch, include_ablations):
    from vifactcheck import LABELS
    from vifactcheck.config import ALL_MODELS, STANDARD_MODELS, comparison_pairs
    from vifactcheck.metrics import classification_metrics
    models = ALL_MODELS if include_ablations else STANDARD_MODELS
    monkeypatch.setattr("vifactcheck.full_reporting.EXPECTED_COUNTS", {"train": 6, "dev": 3, "test": 3})
    write_json(tmp_path / "full_plan.json", {"data_checksums": {"train": "fixture"}, "models": list(models)})
    write_json(tmp_path / "selected_hyperparameters.json", {"learning_rate": 1e-5, "evidence_lambda": .1})
    predictions = [{"id": str(i), "article_id": str(i), "gold_label_id": i, "label_id": i} for i in range(3)]
    classification = classification_metrics([0, 1, 2], [0, 1, 2])
    retrieval = {"count": 2, "alignment_coverage": 1.0}
    retrieval.update({k: 1.0 for k in ("hit@5", "hit@20", "sentence_recall@5", "sentence_recall@20", "mrr", "complete_gold@5", "complete_gold@20")})
    metric = {"classification": classification, "retrieval": retrieval, "cost": {"prepared_input_seconds_per_claim": .1}, "error_stages": {}, "truncation": {}}
    for seed in (42, 13, 2026):
        folder = tmp_path / "seeds" / f"seed{seed}"
        write_json(folder / "experiment.json", {"subset_ids": {"train": list(range(6))}, "config": {"seed": seed, "smoke": False, "train_per_label": None}})
        write_json(folder / "summary.json", {m: {"metrics": metric} for m in models})
        for mode in (m.lower() for m in models):
            write_json(folder / f"{mode}_test_metrics.json", metric)
            write_jsonl(folder / f"{mode}_test_predictions.jsonl", predictions)
    result = write_full_report(tmp_path)
    assert len(result["per_seed"]) == 3 * len(models)
    assert result["models"]["M2"]["test_macro_f1"] == {"mean": 1.0, "sample_std": 0.0}
    assert len(read_json(tmp_path / "paired_bootstrap.json")) == 3 * len(comparison_pairs(models))
    assert "not a vote ensemble" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    metadata_file = tmp_path / "seeds/seed2026/experiment.json"
    original_metadata = read_json(metadata_file)
    write_json(metadata_file, {**original_metadata, "config": {**original_metadata["config"], "reranker_architecture": "listwise"}})
    with pytest.raises(ValueError, match="mix reranker"):
        write_full_report(tmp_path)
    write_json(metadata_file, original_metadata)
    write_jsonl(tmp_path / "seeds/seed2026/m2_test_predictions.jsonl", predictions[:-1])
    with pytest.raises(ValueError, match="complete"):
        write_full_report(tmp_path)
