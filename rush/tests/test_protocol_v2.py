from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from vifactcheck.config import TrainConfig
from vifactcheck.io import read_json, write_json, write_jsonl
from vifactcheck.model import VerificationSystem
from vifactcheck.training import (copy_checkpoint, fingerprint, load_checkpoint,
                                  phase, resume_checkpoint, save_checkpoint,
                                  seed_everything, system_from_checkpoint)


class Tokens:
    def encode(self, claim, sentence):
        return [0, int(sentence) + 4, 2], {"claim_truncated": False, "sentence_truncated": False}

    def inputs(self, claim, sentence, device):
        ids, metadata = self.encode(claim, sentence)
        return {"input_ids": torch.tensor([ids], device=device)}, metadata


class Encoder(nn.Module):
    def __init__(self, path, classes, checkpointing):
        super().__init__()
        self.encoder = nn.Embedding(16, 8)
        self.head = nn.Linear(8, classes)

    def encode(self, inputs):
        return self.encoder(inputs["input_ids"]).mean(1).squeeze(0)

    def forward(self, inputs):
        return self.head(self.encode(inputs).unsqueeze(0)).squeeze(0)


@pytest.fixture
def workload(monkeypatch):
    monkeypatch.setattr("vifactcheck.model.PairEncoder", Encoder)
    monkeypatch.setattr("vifactcheck.model.PairTokenizer", lambda *args: Tokens())
    cfg = TrainConfig(device="cpu", learning_rate=.01, accumulation_steps=2, reranker_epochs=1)
    rows = [{"id": str(i), "article_id": str(i), "claim_segmented": "claim", "label": i % 3,
             "candidate_ids": [0, 1], "sentences": [{"id": 0, "segmented": "0"}, {"id": 1, "segmented": "1"}],
             "alignment": {"status": "valid", "gold_sentence_ids": [0]}} for i in range(6)]
    return cfg, rows


def validation(score):
    return {"classification": {"macro_f1": score}, "retrieval": {"mrr": .8}}


def warmup(workload, directory, monkeypatch, scores):
    cfg, rows = workload
    values = iter(scores)
    monkeypatch.setattr("vifactcheck.training.evaluate", lambda *a, **k: validation(next(values)))
    seed_everything(cfg.seed)
    system = VerificationSystem(cfg, "M1", Tokens())
    best = phase(system, rows, rows, directory, "warmup", "warmup", 3)
    end = directory / "warmup_end.pt"
    copy_checkpoint(directory / "warmup_last.pt", end)
    return best, end


@pytest.mark.parametrize("mode", ["M1", "M2", "A2"])
@pytest.mark.parametrize("scores,expected", [
    ([.90, .80, .70, .85, .86], 1),
    ([.70, .90, .80, .85, .86], 2),
    ([.70, .80, .90, .85, .86], 3),
    ([.70, .80, .75, .90, .85], 4),
    ([.70, .80, .75, .85, .90], 5),
    ([.90, .80, .75, .90, .90], 1),
    ([.70, .80, .75, .90, .90], 4),
])
def test_all_five_epochs_branch_from_last_and_ties_keep_earlier(workload, monkeypatch, tmp_path, mode, scores, expected):
    cfg, rows = workload
    best, end = warmup(workload, tmp_path / "parent", monkeypatch, scores[:3])
    expected_model = deepcopy(load_checkpoint(end)["model"])
    values = iter(scores[3:])
    monkeypatch.setattr("vifactcheck.training.evaluate", lambda *a, **k: validation(next(values)))
    system = VerificationSystem(cfg, mode, Tokens())
    original_forward = system.forward
    first_states = []
    def capture(*a, **k):
        if not first_states:
            first_states.append({key: torch.equal(value, expected_model[key]) for key, value in system.state_dict().items()})
        return original_forward(*a, **k)
    monkeypatch.setattr(system, "forward", capture)
    selected_path = phase(system, rows, rows, tmp_path / mode, mode.lower(), "continuation", 2,
                          source=end, selection_source=best)
    assert all(first_states[0].values()), "Branch must use epoch three, even if warmup best is earlier"
    selected = load_checkpoint(selected_path)
    terminal = load_checkpoint(tmp_path / mode / f"{mode.lower()}_last.pt")
    assert selected["mode"] == mode
    assert selected["config"]["protocol_version"] == 2
    assert selected["selected_global_epoch"] == expected
    assert selected["selected_stage"] == ("warmup" if expected <= 3 else "continuation")
    assert terminal["completed_epochs"] == 2 and terminal["global_epoch"] == 5
    assert [e["global_epoch"] for e in terminal["history"]] == [1, 2, 3, 4, 5]
    assert [e["validation"]["classification"]["macro_f1"] for e in terminal["history"]] == scores
    assert len(read_json(tmp_path / mode / f"{mode.lower()}_history.json")) == 5
    assert selected["best_score"] == max(scores)


def test_continuation_cannot_branch_from_earlier_warmup_best(workload, monkeypatch, tmp_path):
    cfg, rows = workload
    best, _ = warmup(workload, tmp_path / "parent", monkeypatch, [.9, .8, .7])
    system = VerificationSystem(cfg, "M2", Tokens())
    with pytest.raises(ValueError, match="final warmup"):
        phase(system, rows, rows, tmp_path / "invalid", "m2", "continuation", 2, source=best, selection_source=best)


def test_restart_between_last_save_and_best_link_recovers_new_winner(workload, monkeypatch, tmp_path):
    import vifactcheck.training as training
    cfg, rows = workload
    best, end = warmup(workload, tmp_path / "parent", monkeypatch, [.7, .8, .75])
    monkeypatch.setattr(training, "evaluate", lambda *a, **k: validation(.9))
    original_copy = training.copy_checkpoint
    monkeypatch.setattr(training, "copy_checkpoint", lambda *a: (_ for _ in ()).throw(RuntimeError("power loss after last save")))
    system = VerificationSystem(cfg, "M2", Tokens())
    with pytest.raises(RuntimeError, match="power loss"):
        phase(system, rows, rows, tmp_path / "branch", "m2", "continuation", 2, source=end, selection_source=best)
    assert load_checkpoint(tmp_path / "branch/m2_last.pt")["global_epoch"] == 4
    assert load_checkpoint(tmp_path / "branch/m2_best.pt")["selected_global_epoch"] == 2
    monkeypatch.setattr(training, "copy_checkpoint", original_copy)
    monkeypatch.setattr(training, "evaluate", lambda *a, **k: validation(.85))
    selected = phase(VerificationSystem(cfg, "M2", Tokens()), rows, rows, tmp_path / "branch", "m2", "continuation", 2,
                     source=end, selection_source=best)
    assert load_checkpoint(selected)["selected_global_epoch"] == 4
    assert load_checkpoint(selected)["best_score"] == .9


def test_cross_warmup_selection_and_optimizer_rng_survive_resume(workload, monkeypatch, tmp_path):
    cfg, rows = workload
    best, end = warmup(workload, tmp_path / "parent", monkeypatch, [.9, .8, .7])
    for name in ("whole", "interrupted"):
        calls = []
        def evaluate(*a, **k):
            calls.append(1)
            if name == "interrupted" and len(calls) == 2:
                raise RuntimeError("simulated restart after epoch four")
            return validation(.85 if len(calls) == 1 else .86)
        monkeypatch.setattr("vifactcheck.training.evaluate", evaluate)
        system = VerificationSystem(cfg, "M2", Tokens())
        if name == "interrupted":
            with pytest.raises(RuntimeError, match="simulated restart"):
                phase(system, rows, rows, tmp_path / name, "m2", "continuation", 2, source=end, selection_source=best)
        else:
            phase(system, rows, rows, tmp_path / name, "m2", "continuation", 2, source=end, selection_source=best)
    monkeypatch.setattr("vifactcheck.training.evaluate", lambda *a, **k: validation(.86))
    resumed = VerificationSystem(cfg, "M2", Tokens())
    phase(resumed, rows, rows, tmp_path / "interrupted", "m2", "continuation", 2, source=end, selection_source=best)
    left, right = (load_checkpoint(tmp_path / name / "m2_last.pt") for name in ("whole", "interrupted"))
    assert all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"])
    assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    assert left["selected_global_epoch"] == right["selected_global_epoch"] == 1
    assert left["best_score"] == right["best_score"] == .9
    assert left["optimizer"]["param_groups"] == right["optimizer"]["param_groups"]


def test_legacy_checkpoint_loads_for_inference_but_not_resume(workload, monkeypatch, tmp_path):
    cfg, rows = workload
    monkeypatch.setattr("vifactcheck.training.evaluate", lambda *a, **k: validation(.7))
    original = VerificationSystem(cfg, "M0", Tokens())
    phase(original, rows, rows, tmp_path, "m0", "m0", 1)
    state = load_checkpoint(tmp_path / "m0_last.pt")
    state.pop("protocol_version")
    state["config"].pop("protocol_version")
    legacy = tmp_path / "legacy" / "m0_last.pt"
    legacy.parent.mkdir()
    save_checkpoint(legacy, state)
    loaded = system_from_checkpoint(legacy)
    assert loaded.config.protocol_version == 1
    assert loaded.prediction(rows[0])["probabilities"] == original.prediction(rows[0])["probabilities"]
    with pytest.raises(ValueError, match="v1"):
        resume_checkpoint(legacy)


def test_full_job_validates_terminal_state_when_warmup_wins(tmp_path, monkeypatch):
    from vifactcheck.full import training_job
    result = {"M2": {"checks": {"reranker_unchanged": True, "trained_reranker_unchanged": False}}}
    write_json(tmp_path / "experiment.json", {"protocol_version": 2})
    write_json(tmp_path / "summary.json", result)
    for name in ("m2_dev_metrics.json", "m2_checks.json"):
        write_json(tmp_path / name, {})
    write_jsonl(tmp_path / "m2_dev_predictions.jsonl", [])
    for name in ("m2_best.pt", "m2_last.pt", "warmup_end.pt"):
        (tmp_path / name).write_bytes(b"fixture")
    monkeypatch.setattr("vifactcheck.full.run_experiment", lambda *a, **k: result)
    paths = []
    monkeypatch.setattr("vifactcheck.full.joint_checks", lambda checkpoint, parent, row: paths.append((checkpoint.name, parent.name)) or {"passed": True})
    training_job(TrainConfig(device="cpu"), tmp_path, ("M2",), None, [{"candidate_ids": [0, 1]}])
    assert paths == [("m2_last.pt", "warmup_end.pt")]
    assert read_json(tmp_path / "m2_checks.json")["reranker_unchanged"]


def test_full_start_requires_35_gib_without_deleting_existing_files(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vifactcheck.full import ExperimentSpaceError, require_full_start_space
    keep = tmp_path / "old.pt"
    keep.write_bytes(b"keep")
    monkeypatch.setattr("vifactcheck.full.shutil.disk_usage", lambda p: SimpleNamespace(free=34 * 1024**3))
    with pytest.raises(ExperimentSpaceError, match="35 GiB"):
        require_full_start_space(tmp_path)
    assert keep.read_bytes() == b"keep"
    monkeypatch.setattr("vifactcheck.full.shutil.disk_usage", lambda p: SimpleNamespace(free=35 * 1024**3))
    assert require_full_start_space(tmp_path) == 35 * 1024**3


def test_every_checkpoint_write_checks_space_before_touching_existing_file(tmp_path, monkeypatch):
    from types import SimpleNamespace
    target = tmp_path / "best.pt"
    target.write_bytes(b"existing checkpoint remains intact")
    monkeypatch.setattr("vifactcheck.training.shutil.disk_usage", lambda p: SimpleNamespace(free=2 * 1024**3))
    with pytest.raises(RuntimeError, match="checkpoint space"):
        save_checkpoint(target, {"model": {"weight": torch.ones(8)}})
    assert target.read_bytes() == b"existing checkpoint remains intact"
    assert not target.with_suffix(".tmp").exists()


def test_lambda_trials_use_the_common_final_warmup(workload, monkeypatch, tmp_path):
    from vifactcheck.full import lambda_job
    cfg, rows = workload
    calls = []
    monkeypatch.setattr("vifactcheck.full.training_job", lambda *args, **kwargs: calls.append((args, kwargs)) or {"M2": {}})
    parent = tmp_path / "selected_lr"
    for value in (.1, .5, 1.):
        current = TrainConfig(**{**cfg.to_dict(), "evidence_lambda": value})
        lambda_job(current, tmp_path / str(value), parent, None, rows, rows)
    assert [args[0].evidence_lambda for args, kwargs in calls] == [.1, .5, 1.]
    assert all(args[2] == ("M2",) and kwargs["initialization_from"] == parent for args, kwargs in calls)


def test_v2_three_seed_report_records_warmup_winner_and_costs(tmp_path, monkeypatch):
    from vifactcheck.config import ALL_MODELS
    from vifactcheck.full_reporting import write_full_report
    from vifactcheck.metrics import classification_metrics
    monkeypatch.setattr("vifactcheck.full_reporting.EXPECTED_COUNTS", {"train": 6, "dev": 3, "test": 3})
    cfg = TrainConfig(reranker_architecture="listwise")
    write_json(tmp_path / "full_plan.json", {"protocol_version": 2, "config_defaults": cfg.to_dict(),
                                            "models": list(ALL_MODELS), "data_checksums": {}})
    write_json(tmp_path / "selected_hyperparameters.json", {"learning_rate": cfg.learning_rate, "evidence_lambda": .5})
    classification = classification_metrics([0, 1, 2], [0, 1, 2])
    retrieval = {"count": 2, "alignment_coverage": 1.0,
                 **{key: 1.0 for key in ("hit@5", "hit@20", "sentence_recall@5", "sentence_recall@20", "mrr", "complete_gold@5", "complete_gold@20")}}
    metric = {"classification": classification, "retrieval": retrieval,
              "cost": {"prepared_input_seconds_per_claim": .1}, "error_stages": {}, "truncation": {}}
    predictions = [{"id": str(i), "article_id": str(i), "gold_label_id": i, "label_id": i} for i in range(3)]
    for seed in (42, 13, 2026):
        folder = tmp_path / "seeds" / f"seed{seed}"
        current = TrainConfig(**{**cfg.to_dict(), "seed": seed})
        write_json(folder / "experiment.json", {"protocol_version": 2, "config": current.to_dict(), "subset_ids": {"train": list(range(6))}})
        write_json(folder / "summary.json", {mode: {
            "metrics": metric,
            "selection": {"protocol_version": 2, "selected_global_epoch": 2, "selected_stage": "warmup" if mode in ("M1", "M2", "A2") else mode.lower(), "selected_validation_macro_f1": 1.0},
            "training_cost": {"training_seconds": 10., "peak_training_allocated_mib": 100., "peak_training_reserved_mib": 120.}}
            for mode in ALL_MODELS})
        for mode in ALL_MODELS:
            write_json(folder / f"{mode.lower()}_test_metrics.json", metric)
            write_jsonl(folder / f"{mode.lower()}_test_predictions.jsonl", predictions)
    report = write_full_report(tmp_path)
    assert report["protocol_version"] == 2 and len(report["per_seed"]) == 15
    assert all(row["selected_global_epoch"] == 2 for row in report["per_seed"])
    assert report["models"]["M2"]["training_seconds"] == {"mean": 10., "sample_std": 0.}
    assert len(read_json(tmp_path / "paired_bootstrap.json")) == 12
    assert "final epoch-three" in (tmp_path / "REPORT.md").read_text()
    path = tmp_path / "seeds/seed13/experiment.json"
    metadata = read_json(path)
    metadata["protocol_version"] = 1
    write_json(path, metadata)
    with pytest.raises(ValueError, match="legacy"):
        write_full_report(tmp_path)
