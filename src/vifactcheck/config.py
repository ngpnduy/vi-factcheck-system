from dataclasses import asdict, dataclass

DATASET_ID = "tranthaihoa/vifactcheck"
DATASET_REVISION = "2df623632299a3459ad95feeff9ac67d7205a42a"
MODEL_ID = "vinai/phobert-base"
MODEL_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
EXPECTED_COUNTS = {"train": 5062, "dev": 723, "test": 1447}
STANDARD_MODELS = ("M0", "M1", "M2")
ALL_MODELS = ("M0", "M1", "A1", "M2", "A2")
PROTOCOL_VERSION = 2


def comparison_pairs(models):
    """Contrasts that isolate retrieval, weighting, and verification feedback."""
    pairs = (("A1", "M0"), ("M1", "A1"), ("M2", "M1"), ("M2", "A2"))
    if not {"A1", "A2"}.issubset(models):
        pairs = (("M1", "M0"), ("M2", "M1")) + pairs[:2] + pairs[3:]
    return tuple((a, b) for a, b in pairs if a in models and b in models)


@dataclass
class TrainConfig:
    protocol_version: int = PROTOCOL_VERSION
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
    reranker_architecture: str = "pairwise"
    listwise_layers: int = 1
    listwise_heads: int = 8
    listwise_feedforward: int = 1536
    listwise_dropout: float = 0.1
    backbone: str | None = None

    def __post_init__(self):
        if self.protocol_version not in (1, PROTOCOL_VERSION):
            raise ValueError("Unsupported training protocol version")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if self.backbone not in (None, "mbert", "xlmr"):
            raise ValueError("Unsupported benchmark backbone")
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
        if self.reranker_architecture not in ("pairwise", "listwise"):
            raise ValueError("reranker_architecture must be pairwise or listwise")
        if min(self.listwise_layers, self.listwise_heads, self.listwise_feedforward) < 1 or not 0 <= self.listwise_dropout < 1:
            raise ValueError("Invalid list-aware attention settings")

    def to_dict(self):
        result = asdict(self)
        if self.backbone is None:
            del result["backbone"]
        return result

    @classmethod
    def from_saved(cls, value):
        """Unversioned checkpoints retain their historical inference protocol."""
        return cls(**{"protocol_version": 1, **value})


def reranker_options(args):
    names = ("reranker_architecture", "listwise_layers", "listwise_heads", "listwise_feedforward", "listwise_dropout")
    return {name: getattr(args, name, TrainConfig.__dataclass_fields__[name].default) for name in names}
