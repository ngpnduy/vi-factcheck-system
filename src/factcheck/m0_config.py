"""Explicit, serializable protocol for the M0 experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class M0Config:
    model_name: str = "vinai/phobert-base-v2"
    model_revision: str = "86cd7fd4c148980922ac11a2cf5e257f2ba639e1"
    model_cache: str = "outputs/models/huggingface"
    word_segmenter_dir: str = "outputs/models/vncorenlp"
    data_dir: str = "data/interim/vifactcheck/m0"
    output_dir: str = "outputs/m0"
    protocol_status: str = "development_pending_m1_learning_rate"
    scope: str = "within_context"
    pooling: str = "mean_probabilities"
    top_k: int = 5
    max_length: int = 256
    claim_max_tokens: int = 96
    seeds: tuple[int, ...] = (13, 42, 2026)
    warmup_epochs: int = 3
    continuation_epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    device: str = "auto"
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        if self.scope != "within_context" or self.pooling != "mean_probabilities":
            raise ValueError("M0 requires within_context retrieval and mean_probabilities")
        if self.top_k != 5 or self.max_length != 256 or self.claim_max_tokens != 96:
            raise ValueError("M0 protocol requires top_k=5, max_length=256, claim_max_tokens=96")
        for name in ("batch_size", "gradient_accumulation_steps", "cpu_threads", "warmup_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.continuation_epochs) is not int or self.continuation_epochs < 0:
            raise ValueError("continuation_epochs must be a nonnegative integer")
        if not self.seeds or any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not 0 < self.learning_rate < 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Invalid learning_rate or warmup_ratio")
        if not self.max_grad_norm > 0 or not self.weight_decay >= 0:
            raise ValueError("Invalid max_grad_norm or weight_decay")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if len(self.model_revision) != 40:
            raise ValueError("Pin model_revision to a full Hugging Face commit SHA")

    @classmethod
    def load(cls, path: Path) -> M0Config:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Config must be a YAML mapping")
        if "seeds" in values:
            values["seeds"] = tuple(values["seeds"])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
