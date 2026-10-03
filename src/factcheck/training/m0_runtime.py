"""Shared reproducible loading, inference, and artifact utilities for M0."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from factcheck.data.vifactcheck import _read_jsonl, _sha256, _write_jsonl
from factcheck.m0_config import M0Config
from factcheck.models.verifier import ID_TO_LABEL, M0Verifier


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def seed_everything(seed: int, cpu_threads: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def model_batch(batch: dict, device: torch.device, *, labels: bool = False) -> dict:
    names = {"input_ids", "attention_mask", "token_type_ids", "bag_index", "num_claims"}
    if labels:
        names.add("labels")
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items() if key in names}


def predict_batches(model: M0Verifier, loader: DataLoader, device: torch.device,
                    *, seed: int, model_id: str) -> tuple[list[dict], dict]:
    model.eval()
    rows = []
    truncation = {"claims": 0, "evidence": 0, "pairs": 0}
    pair_count = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            result = model(**model_batch(batch, device))
            probabilities = result["probs"].cpu().tolist()
            for claim_id, sentence_ids, probs in zip(
                    batch["ids"], batch["sentence_ids"], probabilities, strict=True):
                rows.append({"id": claim_id, "sentence_ids": sentence_ids,
                             "probabilities": probs,
                             "label": ID_TO_LABEL[int(np.argmax(probs))],
                             "seed": seed, "model": model_id})
            pair_count += len(batch["bag_index"])
            for key in truncation:
                truncation[key] += batch["truncation"][key]
    synchronize(device)
    elapsed = time.perf_counter() - started
    return rows, {"seconds": elapsed, "claims": len(rows), "pairs": pair_count,
                  "seconds_per_claim": elapsed / len(rows), "truncation": truncation,
                  "timing_scope": "collation, segmentation/tokenization, and model inference; "
                                  "excludes model loading and BM25 preparation"}


def verify_prepared(data_dir: Path, splits: tuple[str, ...]) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["scope"] != "within_context" or manifest["candidate_k"] < 20:
        raise ValueError("Prepared data does not implement the M0 retrieval protocol")
    expected_bm25 = {"method": "lucene", "k1": 1.5, "b": 0.75,
                     "tokenizer": "unicode_word_v1", "statistics": "per_context",
                     "ties": "sentence_index_ascending"}
    if any(manifest["bm25"].get(key) != value for key, value in expected_bm25.items()):
        raise ValueError("Prepared BM25 settings differ from the configured M0 protocol")
    for split in splits:
        if split not in manifest["splits"]:
            raise ValueError(f"Split {split} is not in the prepared manifest")
        for suffix in ("inputs", "targets"):
            name = f"{split}.{suffix}.jsonl"
            if _sha256(data_dir / name) != manifest["outputs"][name]["sha256"]:
                raise ValueError(f"Prepared artifact checksum mismatch: {name}")
    return manifest


def load_components(config: M0Config, checkpoint: Path | None = None):
    from transformers import AutoTokenizer

    source = str(checkpoint) if checkpoint is not None else config.model_name
    kwargs = {"local_files_only": True}
    if checkpoint is None:
        kwargs.update(revision=config.model_revision, cache_dir=config.model_cache)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False, **kwargs)
    model = M0Verifier.from_pretrained(source, **kwargs)
    return model, tokenizer


def runtime_metadata() -> dict:
    packages = {}
    for name in ("torch", "transformers", "numpy", "bm25s", "safetensors"):
        packages[name] = importlib.metadata.version(name)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "git_revision": revision, "git_dirty": dirty}


def save_predictions(path: Path, predictions: list[dict]) -> None:
    _write_jsonl(path, predictions)


def read_rows(path: Path) -> list[dict]:
    return _read_jsonl(path)
