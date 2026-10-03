"""Gold-free retrieved inputs, separate targets, and variable-size claim bags."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from factcheck.models.verifier import LABEL_TO_ID

Rows = str | Path | Sequence[Mapping[str, Any]]
INPUT_FIELDS = {"id", "claim", "context_id", "candidates", "sentence_count"}
CANDIDATE_FIELDS = {"sentence_id", "index", "text", "start", "end", "score"}


def _rows(source: Rows) -> list[dict[str, Any]]:
    if isinstance(source, str | Path):
        result = []
        with Path(source).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{source}:{line_number}: expected a JSON object")
                result.append(row)
        return result
    return [dict(row) for row in source]


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


class M0Dataset(Dataset):
    """Join claim labels by ID without admitting gold evidence into model inputs.

    Candidate order is the BM25 ranking saved by retrieval. Only the first top_k
    sentences are used. All input rows must have exactly the documented input
    fields, so an accidental gold-bearing input file fails immediately.
    """

    def __init__(self, inputs: Rows, targets: Rows | None = None, top_k: int = 5) -> None:
        _integer(top_k, "top_k", minimum=1)
        label_by_id = {}
        if targets is not None:
            for target in _rows(targets):
                target_id = _required_text(target.get("id"), "target id")
                if target_id in label_by_id:
                    raise ValueError(f"duplicate target ID: {target_id}")
                label = target.get("label")
                if not isinstance(label, str) or label not in LABEL_TO_ID:
                    raise ValueError(f"invalid target label for {target_id}: {label!r}")
                label_by_id[target_id] = LABEL_TO_ID[label]

        self.rows = []
        input_ids = set()
        for row in _rows(inputs):
            if set(row) != INPUT_FIELDS:
                raise ValueError(
                    "input fields must be exactly "
                    f"{sorted(INPUT_FIELDS)}; got {sorted(row)}"
                )
            claim_id = _required_text(row["id"], "id")
            if claim_id in input_ids:
                raise ValueError(f"duplicate input ID: {claim_id}")
            input_ids.add(claim_id)
            _required_text(row["claim"], "claim")
            _required_text(row["context_id"], "context_id")
            sentence_count = _integer(row["sentence_count"], "sentence_count", minimum=1)
            candidates = row["candidates"]
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"{claim_id}: candidates must be a nonempty list")
            if len(candidates) > sentence_count:
                raise ValueError(f"{claim_id}: more candidates than context sentences")
            if len(candidates) < min(top_k, sentence_count):
                raise ValueError(f"{claim_id}: insufficient candidates for top_k={top_k}")
            seen_sentences, seen_indexes = set(), set()
            previous_order = None
            for candidate in candidates:
                if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
                    raise ValueError(f"{claim_id}: unexpected candidate fields")
                sentence_id = _required_text(candidate["sentence_id"], "sentence_id")
                _required_text(candidate["text"], "candidate text")
                index = _integer(candidate["index"], "sentence index")
                start = _integer(candidate["start"], "sentence start")
                end = _integer(candidate["end"], "sentence end", minimum=1)
                if index >= sentence_count or end <= start:
                    raise ValueError(f"{claim_id}: invalid sentence index or character span")
                if sentence_id in seen_sentences or index in seen_indexes:
                    raise ValueError(f"{claim_id}: duplicate candidate")
                seen_sentences.add(sentence_id)
                seen_indexes.add(index)
                score = candidate["score"]
                if isinstance(score, bool) or not isinstance(score, int | float):
                    raise ValueError(f"{claim_id}: candidate score must be numeric")
                if not math.isfinite(score):
                    raise ValueError(f"{claim_id}: candidate score must be finite")
                order = (-score, index)
                if previous_order is not None and order < previous_order:
                    raise ValueError(f"{claim_id}: candidates violate BM25 score/index ordering")
                previous_order = order
            selected = {**row, "candidates": [dict(item) for item in candidates[:top_k]]}
            if targets is not None:
                if claim_id not in label_by_id:
                    raise ValueError(f"missing target for input ID: {claim_id}")
                selected["label"] = label_by_id[claim_id]
            self.rows.append(selected)
        if not self.rows:
            raise ValueError("M0 dataset must not be empty")
        if targets is not None and input_ids != set(label_by_id):
            raise ValueError("targets contain IDs absent from inputs")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class M0Collator:
    """Word-segment and tokenize each pair, retaining only real sentences.

    ``segment`` is required: production callers must supply a Vietnamese word
    segmenter suitable for PhoBERT. Tests can inject a deterministic substitute.
    The claim cap counts subword tokens before special tokens. The pair cap
    includes all special tokens. Evidence uses the remaining budget.
    """

    def __init__(
        self,
        tokenizer: Any,
        segment: Callable[[str], str],
        *,
        max_length: int = 256,
        claim_max_tokens: int = 96,
        segment_cache_size: int = 10000,
    ) -> None:
        _integer(max_length, "max_length", minimum=1)
        _integer(claim_max_tokens, "claim_max_tokens", minimum=1)
        _integer(segment_cache_size, "segment_cache_size")
        if not callable(segment):
            raise TypeError("a Vietnamese word-segmentation callable is required")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.claim_max_tokens = claim_max_tokens
        self.special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
        if claim_max_tokens + self.special_tokens >= max_length:
            raise ValueError("pair budget must leave at least one token for evidence")
        self.segment = lru_cache(maxsize=segment_cache_size)(segment)

    def _encode(self, text: str) -> list[int]:
        segmented = self.segment(text)
        _required_text(segmented, "segmented text")
        ids = self.tokenizer.encode(segmented, add_special_tokens=False)
        if not ids:
            raise ValueError("nonempty text produced no tokenizer tokens")
        return ids

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        labeled = ["label" in row for row in rows]
        if any(labeled) and not all(labeled):
            raise ValueError("batch mixes labeled and unlabeled claims")
        encoded_pairs, bag_index, sentence_ids = [], [], []
        truncation = {"claims": 0, "evidence": 0, "pairs": 0}
        uses_token_types = "token_type_ids" in self.tokenizer.model_input_names
        for claim_index, row in enumerate(rows):
            candidates = row["candidates"]
            if not candidates:
                raise ValueError(f"{row['id']}: cannot collate an empty claim bag")
            claim_ids = self._encode(row["claim"])
            claim_truncated = len(claim_ids) > self.claim_max_tokens
            truncation["claims"] += int(claim_truncated)
            claim_ids = claim_ids[:self.claim_max_tokens]
            evidence_budget = self.max_length - self.special_tokens - len(claim_ids)
            selected_ids = []
            for candidate in candidates:
                evidence_ids = self._encode(candidate["text"])
                evidence_truncated = len(evidence_ids) > evidence_budget
                truncation["evidence"] += int(evidence_truncated)
                truncation["pairs"] += int(claim_truncated or evidence_truncated)
                evidence_ids = evidence_ids[:evidence_budget]
                pair_ids = self.tokenizer.build_inputs_with_special_tokens(claim_ids, evidence_ids)
                if len(pair_ids) > self.max_length:
                    raise ValueError("tokenizer special-token count exceeds configured pair budget")
                encoded = {"input_ids": pair_ids, "attention_mask": [1] * len(pair_ids)}
                if uses_token_types:
                    encoded["token_type_ids"] = self.tokenizer.create_token_type_ids_from_sequences(
                        claim_ids, evidence_ids
                    )
                encoded_pairs.append(encoded)
                bag_index.append(claim_index)
                selected_ids.append(candidate["sentence_id"])
            sentence_ids.append(selected_ids)
        batch = dict(self.tokenizer.pad(encoded_pairs, padding=True, return_tensors="pt"))
        batch.update(
            bag_index=torch.tensor(bag_index, dtype=torch.long),
            num_claims=len(rows),
            ids=[row["id"] for row in rows],
            sentence_ids=sentence_ids,
            truncation=truncation,
        )
        if all(labeled):
            batch["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        return batch
