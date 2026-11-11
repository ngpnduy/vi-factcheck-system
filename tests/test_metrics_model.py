import pytest
import torch

from vifactcheck.metrics import classification_metrics, retrieval_metrics
from vifactcheck.model import aggregate, evidence_loss, stable_topk


def test_uniform_and_weighted_aggregation():
    logits = torch.tensor([[2.0, 0.0, -1.0], [-1.0, 3.0, 0.0]])
    log_probs, weights = aggregate(logits)
    assert torch.allclose(log_probs.exp(), logits.softmax(-1).mean(0))
    assert torch.allclose(weights, torch.tensor([0.5, 0.5]))
    scores = torch.tensor([0.0, 2.0], requires_grad=True)
    log_probs, weights = aggregate(logits, scores)
    assert torch.allclose(log_probs.exp(), (logits.softmax(-1) * weights[:, None]).sum(0))
    (-log_probs[0]).backward()
    assert scores.grad.norm() > 0


def test_topk_ties_and_gradient_path():
    scores = torch.tensor([1.0, 1.0, -2.0], requires_grad=True)
    chosen, order = stable_topk(scores, [3, 1, 2], 2)
    assert chosen == [1, 0] and order == [1, 0, 2]
    scores[chosen].sum().backward()
    assert scores.grad.tolist() == [1.0, 1.0, 0.0]


def test_balanced_evidence_loss_and_nei():
    row = {"label": 1, "alignment": {"status": "valid", "gold_sentence_ids": [0]}}
    scores = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    loss = evidence_loss(scores, row, [0, 1, 2])
    expected = 0.5 * (torch.nn.functional.softplus(-scores[0]) + torch.nn.functional.softplus(scores[1:]).mean())
    assert torch.allclose(loss, expected)
    loss.backward()
    assert (scores.grad != 0).all()
    row["label"] = 2
    assert evidence_loss(scores, row, [0, 1, 2]) is None


def test_hand_computed_retrieval_denominators():
    rows = [
        {"label": 0, "alignment": {"status": "valid", "gold_sentence_ids": [1, 2]}},
        {"label": 1, "alignment": {"status": "valid", "gold_sentence_ids": [4]}},
        {"label": 2, "alignment": {"status": "valid", "gold_sentence_ids": [0]}},
        {"label": 0, "alignment": {"status": "unmatched", "gold_sentence_ids": []}},
    ]
    metrics = retrieval_metrics(rows, [[0, 1, 2], [8, 9], [0], [0]], ks=(2, 5))
    assert metrics["count"] == 2
    assert metrics["hit@2"] == 0.5
    assert metrics["sentence_recall@2"] == 0.25
    assert metrics["complete_gold@2"] == 0
    assert metrics["complete_gold@5"] == 0.5
    assert metrics["mrr"] == 0.25


def test_three_label_metrics_even_with_absent_labels():
    metrics = classification_metrics([0, 0], [0, 0])
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert len(metrics["confusion_matrix"]) == 3
