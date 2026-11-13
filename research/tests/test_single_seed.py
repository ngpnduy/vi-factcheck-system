import argparse
import importlib.util
from pathlib import Path

import pytest

from vifactcheck.config import ALL_MODELS, TrainConfig
from vifactcheck.io import read_json, write_json, write_jsonl
from vifactcheck.metrics import classification_metrics


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_single_seed.py"
    spec = importlib.util.spec_from_file_location("single_seed_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_seed_report_keeps_all_five_and_conditional_uncertainty(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "EXPECTED_COUNTS", {"test": 3})
    metric = {"classification": classification_metrics([0, 1, 2], [0, 1, 2]),
              "retrieval": {"count": 2, "mrr": 1.0}, "cost": {"peak_cuda_allocated_mib": 100.0}}
    monkeypatch.setattr(runner, "verify_completed_training", lambda directory: {
        m: {"metrics": metric} for m in ALL_MODELS})
    models = tmp_path / "models"
    prediction = [{"id": str(i), "article_id": str(i), "gold_label_id": i, "label_id": i} for i in range(3)]
    for mode in ALL_MODELS:
        write_json(models / f"{mode.lower()}_test_metrics.json", metric)
        write_jsonl(models / f"{mode.lower()}_test_predictions.jsonl", prediction)
    report = runner.write_report(tmp_path, TrainConfig(), models)
    assert [r["model"] for r in report["models"]] == list(ALL_MODELS)
    assert len(report["paired_article_bootstrap"]) == 4
    assert "between-training-seed" in report["limitation"]
    assert all(r["count"] == 2 for r in report["models"])
    assert "Single-seed" in (tmp_path / "REPORT.md").read_text()
    # A complete classification count cannot hide a truncated prediction set.
    write_jsonl(models / "a2_test_predictions.jsonl", prediction[:2])
    with pytest.raises(ValueError, match="complete official"):
        runner.write_report(tmp_path, TrainConfig(), models)


def test_single_seed_failure_is_visible_and_changed_configuration_rejected(runner, monkeypatch, tmp_path):
    data = tmp_path / "data"
    write_json(data / "manifest.json", {"prepared_file_sha256": {}})
    args = argparse.Namespace(output=str(tmp_path / "output"), status_dir=str(tmp_path / "status"),
                              data=str(data), model_path=str(tmp_path / "model"), segmenter_path=str(tmp_path / "segmenter"),
                              seed=42, reranker_architecture="listwise")
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "require_space", lambda *a: 10 * 1024**3)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        runner.run(args)
    assert read_json(Path(args.status_dir) / "status.json")["status"] == "failed"
    assert read_json(Path(args.output) / "failure.json")["exception"] == "RuntimeError"
    args.seed = 13
    with pytest.raises(ValueError, match="Source, data or settings changed"):
        runner.run(args)
