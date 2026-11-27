from pathlib import Path
import pytest
import torch
from torch import nn

from vifactcheck.config import TrainConfig
from vifactcheck.model import PairTokenizer, VerificationSystem
from vifactcheck.training import fingerprint, load_checkpoint, phase, seed_everything


class FakeTokenizer:
    def encode(self, claim, sentence):
        return [0, int(sentence.split()[-1]) + 4, 2], {"claim_truncated": False, "sentence_truncated": False}

    def inputs(self, claim, sentence, device):
        ids, metadata = self.encode(claim, sentence)
        return {"input_ids": torch.tensor([ids], device=device)}, metadata


class TinyEncoder(nn.Module):
    def __init__(self, path, classes, checkpointing):
        super().__init__()
        self.encoder = nn.Embedding(32, 8)
        self.head = nn.Linear(8, classes)

    def forward(self, inputs):
        return self.head(self.encoder(inputs["input_ids"]).mean(1)).squeeze(0)


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr("vifactcheck.model.PairEncoder", TinyEncoder)
    config = TrainConfig(device="cpu", accumulation_steps=2, learning_rate=0.01, warmup_epochs=1, continuation_epochs=2)
    rows = [{"id": str(i), "claim_segmented": "claim", "label": i % 3, "candidate_ids": [0, 1], "sentences": [{"id": 0, "segmented": "sentence 0"}, {"id": 1, "segmented": "sentence 1"}], "alignment": {"status": "valid", "gold_sentence_ids": [0]}, "article_id": str(i)} for i in range(6)]
    return config, rows


def test_pair_token_limits():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [5] * int(text)
        def num_special_tokens_to_add(self, pair=True):
            return 4
        def build_inputs_with_special_tokens(self, claim, evidence):
            return [0] + claim + [2, 2] + evidence + [2]
    tokenizer = PairTokenizer.__new__(PairTokenizer)
    tokenizer.tokenizer, tokenizer.max_length, tokenizer.claim_max_tokens = Tokenizer(), 256, 96
    ids, metadata = tokenizer.encode("200", "300")
    assert len(ids) == 256
    assert ids[97:99] == [2, 2]
    assert metadata["claim_truncated"] and metadata["sentence_truncated"]


def test_m1_frozen_m2_joint_and_nei(tiny):
    config, rows = tiny
    seed_everything(42)
    m1 = VerificationSystem(config, "M1", FakeTokenizer())
    initial = fingerprint(m1.reranker)
    m1.configure_training("continuation")
    output = m1(rows[0], "continuation")
    (-output["log_probabilities"][0]).backward()
    assert all(p.grad is None for p in m1.reranker.parameters())
    assert fingerprint(m1.reranker) == initial
    seed_everything(42)
    m2 = VerificationSystem(config, "M2", FakeTokenizer())
    assert fingerprint(m1.verifier) == fingerprint(m2.verifier)
    m2.configure_training("continuation")
    output = m2(rows[0], "continuation")
    # Isolate verification loss, so evidence loss cannot explain the gradient.
    (-output["log_probabilities"][0]).backward()
    assert m2.reranker.head.weight.grad.norm() > 0
    assert m2.reranker.encoder.weight.grad.norm() > 0
    assert m2(rows[2], "continuation")["evidence_loss"] is None


def test_common_warmup_branches_restore_optimizer(tiny, tmp_path):
    config, rows = tiny
    seed_everything(42)
    warm = VerificationSystem(config, "M1", FakeTokenizer())
    source = phase(warm, rows, rows, tmp_path / "warm", "warmup", "warmup", 1)
    shared = load_checkpoint(source)["model"]
    m1 = VerificationSystem(config, "M1", FakeTokenizer())
    m2 = VerificationSystem(config, "M2", FakeTokenizer())
    m1.load_state_dict(shared)
    m2.load_state_dict(shared)
    assert fingerprint(m1.verifier) == fingerprint(m2.verifier)
    assert fingerprint(m1.reranker) == fingerprint(m2.reranker)
    frozen = fingerprint(m1.reranker)
    phase(m1, rows, rows, tmp_path / "m1", "m1", "continuation", 1, source)
    phase(m2, rows, rows, tmp_path / "m2", "m2", "continuation", 1, source)
    assert fingerprint(m1.reranker) == frozen
    assert fingerprint(m2.reranker) != frozen


def test_interrupted_resume_matches_uninterrupted_training(tiny, tmp_path, monkeypatch):
    import vifactcheck.training as training
    config, rows = tiny
    seed_everything(42)
    whole = VerificationSystem(config, "M0", FakeTokenizer())
    phase(whole, rows, rows, tmp_path / "whole", "m0", "m0", 2)
    seed_everything(42)
    interrupted = VerificationSystem(config, "M0", FakeTokenizer())
    original, calls = training.evaluate, 0
    def crash_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "evaluate", crash_on_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        phase(interrupted, rows, rows, tmp_path / "resumed", "m0", "m0", 2)
    monkeypatch.setattr(training, "evaluate", original)
    restarted = VerificationSystem(config, "M0", FakeTokenizer())
    phase(restarted, rows, rows, tmp_path / "resumed", "m0", "m0", 2)
    left = load_checkpoint(tmp_path / "whole/m0_last.pt")
    right = load_checkpoint(tmp_path / "resumed/m0_last.pt")
    assert all(torch.equal(left["model"][key], right["model"][key]) for key in left["model"])
    assert left["completed_epochs"] == right["completed_epochs"] == 2


def test_completed_m2_phase_resumes_without_training_or_validation(tiny, tmp_path, monkeypatch):
    config, rows = tiny
    seed_everything(42)
    trained = VerificationSystem(config, "M2", FakeTokenizer())
    best = phase(trained, rows, rows, tmp_path, "m2", "continuation", 2)
    expected = load_checkpoint(best)["model"]
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.pt")}
    restarted = VerificationSystem(config, "M2", FakeTokenizer())
    monkeypatch.setattr(restarted, "forward", lambda *a, **k: pytest.fail("Completed M2 phase was retrained"))
    monkeypatch.setattr("vifactcheck.training.evaluate", lambda *a, **k: pytest.fail("Completed phase was revalidated"))
    assert phase(restarted, rows, rows, tmp_path, "m2", "continuation", 2) == best
    assert all(torch.equal(restarted.state_dict()[key], expected[key]) for key in expected)
    assert before == {path.name: path.read_bytes() for path in tmp_path.glob("*.pt")}
