"""M0 claim-level supervision and tokenization; no pretrained model downloads."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from factcheck.models.verifier import ID_TO_LABEL, LABEL_TO_ID, M0Verifier, uniform_log_pool
from factcheck.training.m0_dataset import M0Collator, M0Dataset


class FakeTokenizer:
    """Whitespace test tokenizer with the four special tokens used by RoBERTa."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self.encoded_texts = []

    def num_special_tokens_to_add(self, pair=True):
        return 4 if pair else 2

    def encode(self, text, add_special_tokens=False):
        self.encoded_texts.append(text)
        return list(range(4, 4 + len(text.split())))

    def build_inputs_with_special_tokens(self, claim_ids, evidence_ids):
        return [0, *claim_ids, 2, 2, *evidence_ids, 2]

    def pad(self, encoded_pairs, padding=True, return_tensors="pt"):
        max_length = max(len(item["input_ids"]) for item in encoded_pairs)
        return {
            field: torch.tensor([
                item[field] + [1 if field == "input_ids" else 0] * (max_length - len(item[field]))
                for item in encoded_pairs
            ])
            for field in ("input_ids", "attention_mask")
        }


def input_row(claim_id="claim-1", count=3):
    return {
        "id": claim_id,
        "claim": "Hà Nội là thủ đô Việt Nam",
        "context_id": "context-1",
        "sentence_count": count,
        "candidates": [
            {
                "sentence_id": f"s{index}",
                "index": index,
                "text": f"Câu bằng chứng {index}",
                "start": index * 20,
                "end": index * 20 + 18,
                "score": 10.0 - index,
            }
            for index in range(count)
        ],
    }


def test_uniform_pool_is_mean_probability_and_loss_is_per_claim():
    logits = torch.tensor([
        [4.0, 1.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 3.0],
    ], requires_grad=True)
    bag_index = torch.tensor([0, 0, 1])
    log_probs = uniform_log_pool(logits, bag_index, 2)
    pair_probs = logits.softmax(dim=-1)
    expected = torch.stack([pair_probs[:2].mean(dim=0), pair_probs[2]])
    assert torch.allclose(log_probs.exp(), expected)
    assert not torch.allclose(log_probs[0].exp(), logits[:2].mean(dim=0).softmax(dim=-1))
    labels = torch.tensor([0, 2])
    loss = F.nll_loss(log_probs, labels)
    expected_loss = -(expected[0, 0].log() + expected[1, 2].log()) / 2
    assert torch.allclose(loss, expected_loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad.abs().sum(dim=-1) > 0).all()
    permutation = torch.tensor([2, 1, 0])
    assert torch.allclose(
        uniform_log_pool(logits[permutation], bag_index[permutation], 2), log_probs
    )


def test_pool_stability_and_empty_bags():
    logits = torch.tensor([[10000.0, -10000.0, 0.0]])
    log_probs = uniform_log_pool(logits, torch.tensor([0]), 1)
    assert torch.isfinite(log_probs).all()
    assert F.nll_loss(log_probs, torch.tensor([1])).item() == 20000.0
    with pytest.raises(ValueError, match="no evidence"):
        uniform_log_pool(logits, torch.tensor([0]), 2)
    with pytest.raises(ValueError, match="out-of-range"):
        uniform_log_pool(logits, torch.tensor([-1]), 1)


def test_dataset_target_join_and_gold_isolation():
    inputs = [input_row("a", count=7), input_row("b", count=1)]
    targets = [
        {"id": "b", "label": "NOT_ENOUGH_INFO", "gold_sentence_ids": ["secret-b"]},
        {"id": "a", "label": "REFUTED", "evidence": "SECRET GOLD"},
    ]
    dataset = M0Dataset(inputs, targets)
    assert len(dataset) == 2
    assert [dataset[index]["label"] for index in range(2)] == [1, 2]
    assert len(dataset[0]["candidates"]) == 5
    assert "SECRET" not in str(dataset.rows)
    assert len(inputs[0]["candidates"]) == 7
    inference = M0Dataset(inputs)
    assert "label" not in inference[0]
    with pytest.raises(ValueError, match="exactly"):
        M0Dataset([{**inputs[0], "evidence": "gold"}])
    with pytest.raises(ValueError, match="duplicate input"):
        M0Dataset([inputs[0], inputs[0]])
    with pytest.raises(ValueError, match="duplicate target"):
        M0Dataset(inputs, targets + [targets[0]])
    with pytest.raises(ValueError, match="missing target"):
        M0Dataset(inputs, targets[:1])
    with pytest.raises(ValueError, match="absent from inputs"):
        M0Dataset(inputs, targets + [{"id": "c", "label": "SUPPORTED"}])
    broken = copy.deepcopy(inputs)
    broken[0]["candidates"][0]["score"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        M0Dataset(broken)
    broken = copy.deepcopy(inputs)
    broken[0]["candidates"] = broken[0]["candidates"][:2]
    with pytest.raises(ValueError, match="insufficient candidates"):
        M0Dataset(broken)
    broken = copy.deepcopy(inputs)
    broken[0]["candidates"][0]["score"] = 1.0
    with pytest.raises(ValueError, match="ordering"):
        M0Dataset(broken)


def test_collator_caps_counts_and_segmentation_cache():
    tokenizer = FakeTokenizer()
    segmented = []

    def segment(text):
        segmented.append(text)
        return text.replace("Hà Nội", "Hà_Nội").replace("Việt Nam", "Việt_Nam")

    collator = M0Collator(tokenizer, segment, max_length=12, claim_max_tokens=3)
    rows = [input_row("a", 2), input_row("b", 1)]
    rows[0]["candidates"][0]["text"] = " ".join(["dài"] * 20)
    rows[1]["claim"] = rows[0]["claim"]
    rows[1]["candidates"][0]["text"] = rows[0]["candidates"][0]["text"]
    dataset = M0Dataset(rows, [
        {"id": "a", "label": "SUPPORTED"}, {"id": "b", "label": "REFUTED"},
    ])
    batch = collator(dataset.rows)
    assert batch["input_ids"].shape == (3, 12)
    assert batch["attention_mask"].sum(dim=-1).tolist() == [12, 11, 12]
    assert batch["bag_index"].tolist() == [0, 0, 1]
    assert batch["num_claims"] == 2
    assert batch["labels"].tolist() == [0, 1]
    assert batch["sentence_ids"] == [["s0", "s1"], ["s0"]]
    assert batch["truncation"] == {"claims": 2, "evidence": 2, "pairs": 3}
    assert segmented.count(rows[0]["claim"]) == 1
    assert "Hà_Nội là thủ đô Việt_Nam" in tokenizer.encoded_texts
    assert batch["input_ids"][1, -1] == 1
    assert batch["attention_mask"][1, -1] == 0
    inference_batch = collator(M0Dataset(rows).rows)
    assert "labels" not in inference_batch
    with pytest.raises(ValueError, match="at least one token"):
        M0Collator(tokenizer, segment, max_length=100, claim_max_tokens=96)


class TinyPairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 3)

    def forward(self, input_ids, attention_mask):
        token_logits = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1)
        return SimpleNamespace(logits=(token_logits * weights).sum(1) / weights.sum(1))


def test_verifier_padding_invariance_and_tiny_claim_overfit():
    torch.manual_seed(42)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        verifier = M0Verifier(TinyPairModel())
        labels = torch.tensor([index % 3 for index in range(16)])
        bag = [index for index in range(16) for _ in range(1 + index % 3)]
        pairs = torch.tensor([[int(labels[index]) + 2] for index in bag])
        batch = {
            "input_ids": pairs,
            "attention_mask": torch.ones_like(pairs),
            "bag_index": torch.tensor(bag),
            "num_claims": 16,
            "labels": labels,
        }
        original = verifier(**batch)["probs"]
        padded = {
            **batch,
            "input_ids": F.pad(pairs, (0, 3), value=7),
            "attention_mask": F.pad(torch.ones_like(pairs), (0, 3), value=0),
        }
        assert torch.allclose(original, verifier(**padded)["probs"])
        optimizer = torch.optim.Adam(verifier.parameters(), lr=0.1)
        initial_loss = verifier(**batch)["loss"].item()
        for _ in range(80):
            optimizer.zero_grad()
            output = verifier(**batch)
            output["loss"].backward()
            optimizer.step()
        output = verifier(**batch)
        assert output["loss"].item() < initial_loss * 0.05
        assert torch.equal(output["probs"].argmax(dim=-1), labels)
    finally:
        torch.set_num_threads(previous_threads)


def test_tiny_local_huggingface_model_save_reload(tmp_path):
    from transformers import RobertaConfig, RobertaForSequenceClassification

    config = RobertaConfig(
        vocab_size=32,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=16,
        max_position_embeddings=64,
        num_labels=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    verifier = M0Verifier(RobertaForSequenceClassification(config)).eval()
    batch = {
        "input_ids": torch.tensor([[0, 4, 2, 2, 5, 2], [0, 6, 2, 2, 7, 2]]),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
        "bag_index": torch.tensor([0, 0]),
        "num_claims": 1,
    }
    with torch.no_grad():
        expected = verifier(**batch)["probs"]
    verifier.save_pretrained(tmp_path)
    restored = M0Verifier.from_pretrained(tmp_path, local_files_only=True).eval()
    assert restored.pair_model.config.label2id == LABEL_TO_ID
    assert restored.pair_model.config.id2label == ID_TO_LABEL
    with torch.no_grad():
        assert torch.allclose(restored(**batch)["probs"], expected, atol=1e-7)
