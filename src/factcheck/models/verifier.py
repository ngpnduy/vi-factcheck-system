"""M0: classify each claim/sentence pair and average verdict probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LABEL_TO_ID = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_INFO": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def uniform_log_pool(pair_logits: Tensor, bag_index: Tensor, num_claims: int) -> Tensor:
    """Return log(mean(softmax(pair logits))) for each nonempty claim bag.

    Each real sentence appears once. There are no padded sentences to contribute
    probabilities or alter the denominator. Computing in log space keeps the
    claim-level NLL stable even when the correct class has tiny probability.
    """
    if pair_logits.ndim != 2 or pair_logits.shape[1] != len(LABEL_TO_ID):
        raise ValueError("pair_logits must have shape (number_of_pairs, 3)")
    if bag_index.ndim != 1 or bag_index.shape[0] != pair_logits.shape[0]:
        raise ValueError("bag_index must contain one claim index per pair")
    if bag_index.dtype != torch.long:
        raise ValueError("bag_index must be a torch.long tensor")
    if num_claims < 1 or pair_logits.shape[0] == 0:
        raise ValueError("at least one claim and one real pair are required")
    if bag_index.device != pair_logits.device:
        raise ValueError("bag_index and pair_logits must be on the same device")
    if (bag_index < 0).any() or (bag_index >= num_claims).any():
        raise ValueError("bag_index contains an out-of-range claim index")

    # Pool in float32 under mixed precision; retain float64 for numerical tests.
    logits = (
        pair_logits.float()
        if pair_logits.dtype in {torch.float16, torch.bfloat16}
        else pair_logits
    )
    pair_log_probs = F.log_softmax(logits, dim=-1)
    claim_log_probs = []
    for index in range(num_claims):
        selected = pair_log_probs[bag_index == index]
        if selected.shape[0] == 0:
            raise ValueError(f"claim {index} has no evidence candidates")
        log_count = selected.new_tensor(selected.shape[0]).log()
        claim_log_probs.append(torch.logsumexp(selected, dim=0) - log_count)
    return torch.stack(claim_log_probs)


class M0Verifier(nn.Module):
    """An injectable pair classifier with uniform, differentiable bag pooling."""

    def __init__(self, pair_model: nn.Module) -> None:
        super().__init__()
        if hasattr(pair_model, "config"):
            if pair_model.config.num_labels != len(LABEL_TO_ID):
                raise ValueError("M0 requires exactly three verdict classes")
            pair_model.config.label2id = dict(LABEL_TO_ID)
            pair_model.config.id2label = dict(ID_TO_LABEL)
        self.pair_model = pair_model

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path, **kwargs: Any) -> M0Verifier:
        """Load a Hugging Face classifier; importing this module never downloads it."""
        from transformers import AutoModelForSequenceClassification

        pair_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(LABEL_TO_ID),
            label2id=LABEL_TO_ID,
            id2label=ID_TO_LABEL,
            **kwargs,
        )
        return cls(pair_model)

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        """The pooling layer has no parameters; save the underlying pair model."""
        self.pair_model.save_pretrained(output_dir, **kwargs)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        bag_index: Tensor,
        num_claims: int,
        labels: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        pair_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            pair_inputs["token_type_ids"] = token_type_ids
        pair_output = self.pair_model(**pair_inputs)
        pair_logits = pair_output.logits
        log_probs = uniform_log_pool(pair_logits, bag_index, num_claims)
        output = {
            "pair_logits": pair_logits,
            "log_probs": log_probs,
            "probs": log_probs.exp(),
        }
        if labels is not None:
            if labels.dtype != torch.long or labels.shape != (num_claims,):
                raise ValueError("labels must be a torch.long tensor with one label per claim")
            if (labels < 0).any() or (labels >= len(LABEL_TO_ID)).any():
                raise ValueError("labels must use SUPPORTED=0, REFUTED=1, NOT_ENOUGH_INFO=2")
            # Exactly one loss per claim, irrespective of its number of sentences.
            output["loss"] = F.nll_loss(log_probs, labels)
        return output
