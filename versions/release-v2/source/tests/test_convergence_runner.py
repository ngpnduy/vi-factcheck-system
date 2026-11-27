import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vifactcheck import convergence as diagnostic
from vifactcheck.config import DATASET_ID, DATASET_REVISION, MODEL_ID, MODEL_REVISION, TrainConfig
from vifactcheck.io import read_json, sha256, write_json, write_jsonl
from vifactcheck.model import VerificationSystem
from vifactcheck.training import copy_checkpoint, fingerprint, load_checkpoint, phase, seed_everything


def rows(count=6, prefix="claim"):
    return [{"id": f"{prefix}:{i}", "article_id": f"article:{i}", "label": i % 3,
             "claim_segmented": "claim", "candidate_ids": [0, 1],
             "sentences": [{"id": j, "segmented": str(j)} for j in (0, 1)],
             "alignment": {"status": "valid", "gold_sentence_ids": [0]}} for i in range(count)]


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
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(8, classes)

    def encode(self, inputs):
        return self.dropout(self.encoder(inputs["input_ids"]).mean(1).squeeze(0))

    def forward(self, inputs):
        return self.head(self.encode(inputs).unsqueeze(0)).squeeze(0)


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr("vifactcheck.model.PairEncoder", Encoder)
    monkeypatch.setattr("vifactcheck.model.PairTokenizer", lambda *args: Tokens())
    config = TrainConfig(device="cpu", learning_rate=.01, accumulation_steps=2,
                         reranker_epochs=1, warmup_epochs=3, continuation_epochs=2)
    return config, rows()


def test_eval_loss_uses_all_claim_denominator_and_nei_has_no_bce():
    class Known(nn.Module):
        config = SimpleNamespace(evidence_lambda=.1)

        def forward(self, row, training_stage=None):
            probabilities = torch.full((3,), .1)
            probabilities[row["label"]] = .8
            return {"log_probabilities": probabilities.log(), "evidence_loss": torch.tensor(.4) if row["label"] != 2 else None}
    result = diagnostic.loss_summary(Known(), rows(3), "continuation")
    assert result["eligible_evidence_count"] == 2
    assert result["verification_nll"] == pytest.approx(-math.log(.8))
    assert result["eligible_evidence_bce"] == pytest.approx(.4)
    assert result["evidence_weighted_contribution"] == pytest.approx(.1 * .8 / 3)
    assert result["objective_mean"] == pytest.approx(-math.log(.8) + .1 * .8 / 3)


def test_observer_restores_random_state_and_mode(monkeypatch):
    system = nn.Linear(1, 1)
    system.train()
    def consuming(*args):
        random.random(); np.random.rand(); torch.rand(1)
        args[0].eval()
        return {"verification_nll": 1.0}
    monkeypatch.setattr(diagnostic, "loss_summary", consuming)
    seed_everything(42)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    seed_everything(42)
    diagnostic.make_observer(rows())(system, rows(), rows(), "warmup", {})
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    assert system.training


def test_longer_horizon_preserves_original_warmup_weights(tiny, tmp_path):
    config, data = tiny
    seed_everything(42)
    reference = VerificationSystem(config, "M1", Tokens())
    phase(reference, data, data, tmp_path / "five", "warmup", "warmup", 3)
    original_hash = fingerprint(reference)
    long_config = TrainConfig(**{**config.to_dict(), "continuation_epochs": 12})
    seed_everything(42)
    extended = VerificationSystem(long_config, "M1", Tokens())
    phase(extended, data, data, tmp_path / "fifteen", "warmup", "warmup", 3,
          epoch_observer=diagnostic.make_observer(data), diagnostic_id=diagnostic.DIAGNOSTIC_ID, lr_horizon_epochs=5)
    assert fingerprint(extended) == original_hash
    record = read_json(tmp_path / "fifteen/warmup_history.json")[-1]
    assert record["loss_components"]["eligible_evidence_count"] == 0
    assert record["diagnostics"]["dev"]["verification_nll"] is not None


def test_extended_resume_matches_continuous_and_recovers_missing_journal(tiny, tmp_path):
    config, data = tiny
    config.continuation_epochs = 12
    opts = dict(epoch_observer=diagnostic.make_observer(data), diagnostic_id=diagnostic.DIAGNOSTIC_ID, lr_horizon_epochs=5)
    seed_everything(42)
    parent = VerificationSystem(config, "M1", Tokens())
    best = phase(parent, data, data, tmp_path / "parent", "warmup", "warmup", 3, **opts)
    end = tmp_path / "parent/warmup_end.pt"
    copy_checkpoint(tmp_path / "parent/warmup_last.pt", end)
    outputs = {}
    for name in ("continuous", "resumed"):
        def progress(event):
            if name == "resumed" and event["event"] == "epoch_complete" and event["global_epoch"] == 5:
                raise InterruptedError("simulated restart")
        system = VerificationSystem(config, "M2", Tokens())
        if name == "resumed":
            with pytest.raises(InterruptedError):
                phase(system, data, data, tmp_path / name, "m2", "continuation", 12, source=end,
                      selection_source=best, progress=progress, **opts)
        phase(system, data, data, tmp_path / name, "m2", "continuation", 12, source=end, selection_source=best, **opts)
        outputs[name] = load_checkpoint(tmp_path / name / "m2_last.pt")
    assert outputs['resumed']['selected_global_epoch'] == outputs['continuous']['selected_global_epoch']
    assert all(torch.equal(outputs['resumed']['model'][k], v) for k, v in outputs['continuous']['model'].items())
    journal = tmp_path / "resumed/m2_history.json"
    journal.unlink()
    phase(system, data, data, tmp_path / "resumed", "m2", "continuation", 12, source=end, selection_source=best, **opts)
    assert len(read_json(journal)) == 15
    diagnostic.monitor_logs(tmp_path / "resumed")
    assert len((tmp_path / "resumed/diagnostics.jsonl").read_text().splitlines()) == 15
    with pytest.raises(ValueError, match="LR horizon"):
        phase(system, data, data, tmp_path / "resumed", "m2", "continuation", 12, source=end, selection_source=best,
              **{**opts, "lr_horizon_epochs": 15})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    model = cache / "phobert"
    model.mkdir(parents=True)
    write_json(model / "config.json", {"model_type": "roberta", "vocab_size": 64001})
    for name in ("vocab.txt", "bpe.codes", "pytorch_model.bin"):
        (model / name).write_text('fixture only')
    files = [{"path": "phobert/" + p.name, "sha256": sha256(p)} for p in model.iterdir()]
    write_json(cache / "assets.json", {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "files": files})
    data = tmp_path / "data"
    data.mkdir()
    write_jsonl(data / "train.jsonl", rows(24, "train"))
    write_jsonl(data / "dev.jsonl", rows(12, "dev"))
    # Intentionally no test file: the runner must not require or read one.
    write_json(data / "manifest.json", {"assets": {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION},
                                       "prepared_file_sha256": {s: sha256(data / f'{s}.jsonl') for s in ('train', 'dev')}})
    monkeypatch.setattr(diagnostic.shutil, "disk_usage", lambda p: SimpleNamespace(free=40 * 1024**3))
    args = diagnostic.parser().parse_args(["--data", str(data), "--model-path", str(model), "--output", str(tmp_path / "run"),
                                         "--device", "cpu", "--smoke", "--check-only"])
    return args


def test_check_only_is_read_only_and_never_needs_test(prepared):
    result = diagnostic.run(prepared)
    assert result['status'] == 'ready' and result['writes'] is False
    assert result['train_count'] == 24 and result['dev_count'] == 12
    assert not Path(prepared.output).exists()


def test_preflight_checks_counts_assets_and_revision(prepared):
    prepared.smoke = False
    with pytest.raises(ValueError, match='claim counts'):
        diagnostic.preflight(prepared)
    prepared.smoke = True
    model = Path(prepared.model_path) / 'vocab.txt'
    original = model.read_bytes()
    model.write_text('corruption')
    with pytest.raises(ValueError, match='hash'):
        diagnostic.preflight(prepared)
    model.write_bytes(original)
    manifest_path = Path(prepared.data) / 'manifest.json'
    manifest = read_json(manifest_path)
    manifest['assets']['dataset_revision'] = 'wrong'
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='dataset revision'):
        diagnostic.preflight(prepared)


def test_ownership_refusal_does_not_write_failure(prepared):
    output = Path(prepared.output)
    write_json(output / 'convergence_plan.json', {'other': 'study'})
    before = (output / 'convergence_plan.json').read_bytes()
    with pytest.raises(ValueError, match='different'):
        diagnostic.main(["--data", prepared.data, "--model-path", prepared.model_path, "--output", str(output),
                         "--device", "cpu", "--smoke"])
    assert (output / 'convergence_plan.json').read_bytes() == before
    assert not (output / 'failure.json').exists()


def test_storage_gate_creates_nothing(prepared, monkeypatch):
    monkeypatch.setattr(diagnostic.shutil, 'disk_usage', lambda p: SimpleNamespace(free=6 * 1024**3))
    with pytest.raises(RuntimeError, match='35 GiB'):
        diagnostic.run(prepared)
    assert not Path(prepared.output).exists()


def test_runner_completes_and_reloads_without_training(tiny, prepared, monkeypatch):
    prepared.check_only = False
    prepared.total_epochs, prepared.warmup_epochs, prepared.reranker_epochs = 3, 1, 1
    first = diagnostic.run(prepared)
    output = Path(prepared.output)
    assert first['test_evaluated'] is False
    assert len(read_json(output / 'm2_history.json')) == 3
    assert len((output / 'diagnostics.jsonl').read_text().splitlines()) == 4
    # Complete phases must reload rather than train or evaluate observations again.
    monkeypatch.setattr(diagnostic, 'loss_summary', lambda *a: pytest.fail('Diagnostics repeated on completed phases'))
    second = diagnostic.run(prepared)
    assert first['best_sha256'] == second['best_sha256']
    assert (output / 'source_snapshot/src/vifactcheck/training.py').exists()
