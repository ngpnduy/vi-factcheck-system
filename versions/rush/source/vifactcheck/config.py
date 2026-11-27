from dataclasses import asdict, dataclass

DATASET_ID = "tranthaihoa/vifactcheck"
DATASET_REVISION = "2df623632299a3459ad95feeff9ac67d7205a42a"
MODEL_ID = "vinai/phobert-base"
MODEL_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
EXPECTED_COUNTS = {"train": 5062, "dev": 723, "test": 1447}


@dataclass
class TrainConfig:
    seed: int = 42
    learning_rate: float = 2e-5
    evidence_lambda: float = 0.5
    reranker_epochs: int = 3
    warmup_epochs: int = 3
    continuation_epochs: int = 2
    accumulation_steps: int = 8
    max_length: int = 256
    claim_max_tokens: int = 96
    candidate_count: int = 20
    evidence_count: int = 5
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    model_path: str = ".cache/phobert"
    data_path: str = "data/prepared"
    smoke: bool = False
    train_per_label: int | None = None
    optimizer: str = "adafactor"

    def __post_init__(self):
        if self.device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if not 0 < self.evidence_count <= self.candidate_count:
            raise ValueError("evidence_count must be between 1 and candidate_count")
        if not 4 < self.max_length <= 256 or not 0 < self.claim_max_tokens <= self.max_length - 4:
            raise ValueError("Invalid PhoBERT pair/claim token budget")
        if self.accumulation_steps < 1 or self.learning_rate <= 0 or self.evidence_lambda < 0:
            raise ValueError("Invalid optimization settings")
        if min(self.reranker_epochs, self.warmup_epochs, self.continuation_epochs) < 1:
            raise ValueError("Each training phase needs at least one epoch")
        if self.train_per_label is not None and (self.train_per_label < 1 or self.smoke):
            raise ValueError("A training subset needs a positive per-label count and cannot be smoke")

    def to_dict(self):
        return asdict(self)
