"""Canonical processed-data schema."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

SCHEMA_VERSION = "1.0.0"


class VerdictLabel(StrEnum):
    """The only ground-truth labels in processed fact-checking datasets."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class EvidenceRecord(BaseModel):
    """A gold evidence snippet associated with one claim."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_title: str | None = None
    source_url: HttpUrl | None = None
    position: str | None = None

    @field_validator("evidence_id", "text")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class DatasetRecord(BaseModel):
    """The schema written to every file under data/processed/."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    context: str | None = None
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    label: VerdictLabel
    dataset: str = Field(min_length=1)
    split: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "claim", "dataset")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("context")
    @classmethod
    def strip_optional_context(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("split")
    @classmethod
    def validate_split(cls, value: str) -> str:
        if value not in {"train", "validation", "test"}:
            raise ValueError("must be one of: train, validation, test")
        return value
