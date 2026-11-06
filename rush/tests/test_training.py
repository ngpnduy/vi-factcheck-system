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

    def encode(self, inputs):
        return self.encoder(inputs["input_ids"]).mean(1).squeeze(0)

    def forward(self, inputs):
        return self.head(self.encode(inputs).unsqueeze(0)).squeeze(0)


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
    wrong_mode = VerificationSystem(config, "A2", FakeTokenizer())
    with pytest.raises(ValueError, match="different configuration/data/source"):
        phase(wrong_mode, rows, rows, tmp_path, "m2", "continuation", 2)


@pytest.mark.parametrize("architecture", ["pairwise", "listwise"])
def test_a1_shares_m1_selection_but_uses_uniform_probabilities(tiny, architecture):
    config, rows = tiny
    config.reranker_architecture = architecture
    config.listwise_heads, config.listwise_feedforward, config.listwise_dropout = 2, 16, 0.0
    seed_everything(42)
    m1 = VerificationSystem(config, "M1", FakeTokenizer())
    a1 = VerificationSystem(config, "A1", FakeTokenizer())
    a1.load_state_dict(m1.state_dict())
    m1.eval()
    a1.eval()
    expected, uniform = m1(rows[0]), a1(rows[0])
    assert expected["ranking"] == uniform["ranking"]
    assert expected["selected"] == uniform["selected"]
    assert torch.allclose(uniform["weights"], torch.tensor([0.5, 0.5]))
    logits = [a1.verifier(a1.tokenizer.inputs(rows[0]["claim_segmented"],
              rows[0]["sentences"][i]["segmented"], "cpu")[0]) for i in uniform["selected"]]
    assert torch.allclose(uniform["log_probabilities"].exp(), torch.stack(logits).softmax(-1).mean(0))
    a1.configure_training("a1")
    (-a1(rows[0], "a1")["log_probabilities"][0]).backward()
    assert all(p.grad is None and not p.requires_grad for p in a1.reranker.parameters())
    assert a1.verifier.head.weight.grad.norm() > 0


@pytest.mark.parametrize("architecture", ["pairwise", "listwise"])
def test_a2_blocks_only_verification_and_keeps_evidence_gradients(tiny, architecture):
    from vifactcheck.training import isolated_stop_gradient
    config, rows = tiny
    config.reranker_architecture = architecture
    config.listwise_heads, config.listwise_feedforward, config.listwise_dropout = 2, 16, 0.0
    seed_everything(42)
    a2 = VerificationSystem(config, "A2", FakeTokenizer())
    joint = VerificationSystem(config, "M2", FakeTokenizer())
    joint.load_state_dict(a2.state_dict())
    a2.configure_training("continuation")
    joint.configure_training("continuation")
    result = a2(rows[0], "continuation")
    control = joint(rows[0], "continuation")
    assert torch.allclose(result["log_probabilities"], control["log_probabilities"])
    loss = -result["log_probabilities"][0]
    assert torch.autograd.grad(loss, a2.reranker.head.weight, allow_unused=True, retain_graph=True)[0] is None
    assert torch.autograd.grad(result["evidence_loss"], a2.reranker.head.weight, retain_graph=True)[0].norm() > 0
    if architecture == "listwise":
        attention_weight = a2.reranker.contextualizer.layers[0].self_attn.in_proj_weight
        assert torch.autograd.grad(loss, attention_weight, allow_unused=True, retain_graph=True)[0] is None
        assert torch.autograd.grad(result["evidence_loss"], attention_weight)[0].norm() > 0
    check = isolated_stop_gradient(a2, rows[0])
    assert check["verification_only_reranker_head_gradient_norm"] == 0
    assert check["evidence_only_reranker_head_gradient_norm"] > 0
    assert check["all_reranker_parameters_trainable"] and check["all_verifier_parameters_trainable"]
    nei = a2(rows[2], "continuation")
    assert nei["evidence_loss"] is None
    assert torch.autograd.grad(-nei["log_probabilities"][2], a2.reranker.head.weight, allow_unused=True)[0] is None


def test_listwise_attention_exchanges_information_and_is_permutation_equivariant():
    from vifactcheck.model import CandidateAttention
    config = TrainConfig(device="cpu", reranker_architecture="listwise", listwise_heads=2,
                         listwise_feedforward=16, listwise_dropout=0.0)
    seed_everything(42)
    attention = CandidateAttention(8, config).eval()
    candidates = torch.randn(3, 8, requires_grad=True)
    output = attention(candidates)
    gradient = torch.autograd.grad(output[0, 0], candidates)[0]
    assert gradient[1:].abs().sum() > 0
    permutation = torch.tensor([2, 0, 1])
    assert torch.allclose(attention(candidates[permutation]), output[permutation], atol=1e-6)
    assert attention(candidates[:1]).shape == (1, 8)
    with pytest.raises(ValueError, match="divisible"):
        CandidateAttention(7, config)


@pytest.mark.parametrize("architecture", ["pairwise", "listwise"])
def test_all_five_models_share_initialization_resume_and_external_parents(tiny, tmp_path, monkeypatch, architecture):
    from vifactcheck.config import ALL_MODELS
    from vifactcheck.full import verify_initialization
    from vifactcheck.io import read_json, write_json
    from vifactcheck.training import resume_checkpoint, run_experiment
    import vifactcheck.training as training
    config, rows = tiny
    config.reranker_epochs = 1
    config.reranker_architecture = architecture
    config.listwise_heads, config.listwise_feedforward, config.listwise_dropout = 2, 16, 0.0
    config.data_path = str(tmp_path / "data")
    write_json(Path(config.data_path) / "manifest.json", {"fixture": "same-official-splits"})
    monkeypatch.setattr(training, "training_rows", lambda _: (rows, rows))
    monkeypatch.setattr(training, "PairTokenizer", lambda *a: FakeTokenizer())
    monkeypatch.setattr("vifactcheck.model.PairTokenizer", lambda *a: FakeTokenizer())
    source = tmp_path / "all"
    results = run_experiment(config, source, ALL_MODELS)
    verify_initialization(results)
    assert set(results) == set(ALL_MODELS)
    assert len(read_json(source / "a1_history.json")) == len(read_json(source / "m0_history.json")) == 3
    assert len(read_json(source / "a2_history.json")) == 3
    assert results["A2"]["checks"]["verification_only_reranker_head_gradient_norm"] == 0
    for name in ("a1", "a2"):
        before = (source / f"{name}_last.pt").read_bytes()
        resume_checkpoint(source / f"{name}_last.pt")
        assert (source / f"{name}_last.pt").read_bytes() == before
    original_phase = training.phase
    def no_repeated_preparation(*args, **kwargs):
        if args[4] in ("reranker", "warmup"):
            pytest.fail("Matching parent preparation must not be retrained")
        return original_phase(*args, **kwargs)
    monkeypatch.setattr(training, "phase", no_repeated_preparation)
    from vifactcheck.io import sha256
    parent_hashes = {name: sha256(source / name) for name in ("reranker_best.pt", "warmup_best.pt")}
    extended = run_experiment(config, tmp_path / "ablations", ("A1", "A2"), initialization_from=source)
    assert extended["A2"]["checks"]["shared_warmup_checkpoint_sha256"] == results["M2"]["checks"]["shared_warmup_checkpoint_sha256"]
    assert extended["A1"]["checks"]["reranker_final_sha256"] == results["M1"]["checks"]["reranker_final_sha256"]
    assert parent_hashes == {name: sha256(source / name) for name in parent_hashes}
    original_load = training.load_checkpoint
    for field, value in (("data_signature", "different-claims"), ("mode", "M0"), ("stage", "continuation")):
        def wrong_parent_checkpoint(path):
            checkpoint = original_load(path)
            if Path(path) == source / "reranker_best.pt":
                checkpoint[field] = value
            return checkpoint
        with monkeypatch.context() as scoped:
            scoped.setattr(training, "load_checkpoint", wrong_parent_checkpoint)
            with pytest.raises(ValueError, match="different data"):
                run_experiment(config, tmp_path / ("wrong-" + field), ("A1", "A2"), initialization_from=source)
    with pytest.raises(ValueError, match="different training settings or architecture"):
        other = TrainConfig(**{**config.to_dict(), "reranker_architecture": "pairwise" if architecture == "listwise" else "listwise"})
        run_experiment(other, tmp_path / "wrong", ("A1", "A2"), initialization_from=source)
