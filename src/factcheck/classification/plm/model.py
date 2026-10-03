"""PLM classifier model.

Directly ported from the original ViFactCheck plm_training.py:
    - PhoBERT (or any AutoModel) backbone
    - Dropout(0.3) on [CLS] pooled output
    - Linear head → num_classes

Works with any HuggingFace AutoModel that returns pooled_output as second
element when return_dict=False (e.g. PhoBERT, XLM-R, mBERT, ViBERT).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel


class PLMClassifier(nn.Module):
    """[PLM backbone] → Dropout(0.3) → Linear(hidden, num_classes).

    Hyperparameters from original paper:
        dropout = 0.3
    """

    def __init__(self, backbone: PreTrainedModel, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=0.3)  # from original paper
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, seq_len]
        attention_mask: torch.Tensor,  # [B, seq_len]
    ) -> torch.Tensor:                 # [B, num_classes] logits
        _, pooled = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return self.classifier(self.dropout(pooled))  # [B, num_classes]

    @classmethod
    def from_pretrained(cls, model_name: str, num_classes: int = 3) -> PLMClassifier:
        """Convenience constructor from a HuggingFace model name."""
        backbone = AutoModel.from_pretrained(model_name)
        return cls(backbone, num_classes=num_classes)
