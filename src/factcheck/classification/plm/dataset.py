"""PyTorch Dataset for PLM fact-checking classification.

Reads from the canonical processed JSONL files (DatasetRecord schema).
Each sample pairs (claim, evidence_text) as sentence pair input to the PLM,
exactly as done in the original ViFactCheck paper's plm_training.py.

Two input modes (controlled by `input_mode`):
    "gold_evidence"  – claim + first gold evidence sentence (paper Table 3, col 1)
    "full_context"   – claim + full article context (paper Table 3, col 2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from factcheck.data.schemas import DatasetRecord, VerdictLabel

# Label string → integer index (matches original paper: 0=SUP, 1=REF, 2=NEI)
LABEL2IDX: dict[str, int] = {
    VerdictLabel.SUPPORTED: 0,
    VerdictLabel.REFUTED: 1,
    VerdictLabel.NOT_ENOUGH_INFO: 2,
}
IDX2LABEL: dict[int, str] = {v: k for k, v in LABEL2IDX.items()}

InputMode = Literal["gold_evidence", "full_context"]


def _load_records(path: Path) -> list[DatasetRecord]:
    """Load canonical JSONL into a list of DatasetRecord."""
    records: list[DatasetRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(DatasetRecord.model_validate_json(line))
    return records


def _get_second_text(record: DatasetRecord, mode: InputMode) -> str:
    """Extract the second sentence for the sentence-pair input."""
    if mode == "gold_evidence":
        if record.evidence:
            return record.evidence[0].text
        # Fallback: if no gold evidence, use context
        return record.context or ""
    else:  # full_context
        return record.context or ""


class FactCheckDataset(Dataset):
    """Sentence-pair dataset for claim verification with a PLM.

    Each item returns:
        input_ids      : [max_length]
        attention_mask : [max_length]
        label          : scalar int64
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        input_mode: InputMode = "gold_evidence",
    ) -> None:
        self.records = _load_records(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        text_a = record.claim
        text_b = _get_second_text(record, self.input_mode)

        encoding = self.tokenizer.encode_plus(
            text_a,
            text_pair=text_b,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),       # [max_length]
            "attention_mask": encoding["attention_mask"].flatten(),  # [max_length]
            "label": torch.tensor(LABEL2IDX[record.label.value], dtype=torch.long),
        }

    @property
    def ids(self) -> list[str]:
        """Return record IDs in dataset order (for predictions output)."""
        return [r.id for r in self.records]
