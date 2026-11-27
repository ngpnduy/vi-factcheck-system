import argparse

import pytest

from vifactcheck.cli import main
from vifactcheck.config import ALL_MODELS, TrainConfig, comparison_pairs, reranker_options


def test_slide_configuration_legacy_defaults_and_invalid_attention():
    old = TrainConfig().to_dict()
    for name in reranker_options(argparse.Namespace()):
        old.pop(name)
    assert TrainConfig(**old).reranker_architecture == "pairwise"
    assert comparison_pairs(ALL_MODELS) == (("A1", "M0"), ("M1", "A1"), ("M2", "M1"), ("M2", "A2"))
    assert comparison_pairs(("M0", "M1", "M2")) == (("M1", "M0"), ("M2", "M1"))
    with pytest.raises(ValueError, match="architecture"):
        TrainConfig(reranker_architecture="unrecorded")
    with pytest.raises(ValueError, match="attention"):
        TrainConfig(listwise_heads=0)


def test_training_cli_routes_ablations_and_recorded_architecture(monkeypatch, tmp_path):
    seen = {}
    def capture(config, output, models, **kwargs):
        seen.update(config=config, output=output, models=models, kwargs=kwargs)
        return {"status": "captured"}
    monkeypatch.setattr("vifactcheck.training.run_experiment", capture)
    main(["train", "--device", "cpu", "--models", "A1", "A2", "--reranker-architecture", "listwise",
          "--listwise-heads", "2", "--initialization-from", str(tmp_path / "parent"), "--output", str(tmp_path / "new")])
    assert seen["models"] == ["A1", "A2"]
    assert seen["config"].reranker_architecture == "listwise"
    assert seen["config"].listwise_heads == 2
    assert seen["kwargs"]["initialization_from"] == str(tmp_path / "parent")


def test_full_cli_routes_fifteen_model_protocol(monkeypatch, tmp_path):
    seen = {}
    def capture(args):
        seen.update(vars(args))
        return {"status": "captured"}
    monkeypatch.setattr("vifactcheck.full.run_full", capture)
    main(["full", "--include-ablations", "--reranker-architecture", "listwise", "--output", str(tmp_path)])
    assert seen["include_ablations"] and seen["reranker_architecture"] == "listwise"


def test_smoke_cli_accepts_all_slide_models(monkeypatch, tmp_path):
    seen = {}
    def capture(config, output, models, **kwargs):
        seen.update(models=models, config=config)
        return {"status": "captured"}
    monkeypatch.setattr("vifactcheck.cli.smoke_run", capture)
    main(["smoke", "--models", *ALL_MODELS, "--device", "cpu", "--output", str(tmp_path)])
    assert seen["models"] == list(ALL_MODELS)
    assert seen["config"].smoke


def test_compare_rejects_mixed_architectures(tmp_path):
    from vifactcheck.cli import compare
    from vifactcheck.io import write_json
    directories = []
    for architecture, mode in (("pairwise", "M0"), ("listwise", "M1")):
        directory = tmp_path / architecture
        directories.append(str(directory))
        write_json(directory / "experiment.json", {"config": TrainConfig(reranker_architecture=architecture).to_dict()})
        write_json(directory / "summary.json", {mode: {"metrics": {"classification": {"macro_f1": 0.5}}}})
    with pytest.raises(ValueError, match="combine pairwise/list-aware"):
        compare(argparse.Namespace(runs=directories, output=str(tmp_path / "mixed.json")))
