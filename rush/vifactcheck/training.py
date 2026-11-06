"""Explicit claim-level training and resumable experiment phases."""
from collections import Counter
import gc
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from transformers.optimization import Adafactor

from .config import ALL_MODELS, PROTOCOL_VERSION, TrainConfig
from .data import has_evidence_supervision, load_split, smoke_subset, stratified_subset
from .io import read_json, sha256, write_json, write_jsonl
from .metrics import classification_metrics, error_stage, retrieval_metrics
from .model import PairTokenizer, VerificationSystem, evidence_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path):
    # Only locally generated trusted checkpoints. mmap avoids another full CPU
    # copy of the two encoders on the 16 GB laptop.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def fingerprint(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, payload):
    path = Path(path)
    model_bytes = sum(value.numel() * value.element_size() for value in payload.get("model", {}).values())
    free = shutil.disk_usage(path.parent).free
    required = 2 * 1024**3 + int(model_bytes * 1.15)
    if free < required:
        raise RuntimeError(f"Insufficient checkpoint space: {free / 1024**3:.2f} GiB free; require 2 GiB reserve plus space for the temporary checkpoint")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_v2(checkpoint):
    if checkpoint.get("protocol_version", 1) != PROTOCOL_VERSION:
        raise ValueError("Protocol v1 checkpoints support prediction/evaluation only; start a new protocol v2 run")


def checkpoint_selection(path):
    """Selection provenance lives outside the hashed checkpoint itself."""
    state = load_checkpoint(path)
    checksum = sha256(path)
    result = {key: state[key] for key in ("selected_global_epoch", "selected_stage", "selected_validation_macro_f1")}
    result.update(checkpoint_sha256=checksum,
                  initialization_checkpoint_sha256=state.get("source_checkpoint_sha256"),
                  selected_source_checkpoint_sha256=state.get("selected_source_checkpoint_sha256") or checksum,
                  protocol_version=state["protocol_version"])
    return result


def copy_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def make_optimizer(system, config):
    # Stable parameter ordering allows the common verifier optimizer state to
    # load into both branches. Frozen parameters have no gradients and acquire
    # no optimizer state; M2 can then start updating the same reranker entries.
    return Adafactor(list(system.parameters()), lr=config.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False, beta1=None, weight_decay=0.0, clip_threshold=1.0)


def evaluate(system, rows, output=None, progress=None):
    if not rows:
        raise ValueError("Evaluation split is empty")
    if system.config.device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    predictions = []
    for row in rows:
        prediction = system.prediction(row)
        prediction["gold_label_id"] = row["label"]
        prediction["article_id"] = row["article_id"]
        prediction["alignment"] = row["alignment"]
        prediction["claim"] = row.get("claim", row["claim_segmented"])
        prediction["error_stage"] = error_stage(row, prediction["ranking"], prediction["label_id"])
        predictions.append(prediction)
        if progress and (len(predictions) % 25 == 0 or len(predictions) == len(rows)):
            progress({"event": "evaluation", "completed_claims": len(predictions), "total_claims": len(rows)})
    if system.config.device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    rankings = [p["ranking"] for p in predictions]
    selected_metadata = [meta for p in predictions for meta in p["truncation"]]
    candidate_metadata = [system.tokenizer.encode(r["claim_segmented"], r["sentences"][i]["segmented"])[1] for r in rows for i in r["candidate_ids"][:system.config.candidate_count]]
    elapsed = time.perf_counter() - started
    metrics = {"classification": classification_metrics([r["label"] for r in rows], [p["label_id"] for p in predictions]), "retrieval": retrieval_metrics(rows, rankings), "bm25_retrieval": retrieval_metrics(rows, [r["candidate_ids"] for r in rows]), "error_stages": dict(Counter(p["error_stage"] for p in predictions)), "truncation": {"claim_count": len(rows), "claim_truncated_count": sum(any(m["claim_truncated"] for m in p["truncation"]) for p in predictions), "candidate_pair_count": len(candidate_metadata), "candidate_sentence_truncated_count": sum(m["sentence_truncated"] for m in candidate_metadata), "selected_pair_count": len(selected_metadata), "selected_sentence_truncated_count": sum(m["sentence_truncated"] for m in selected_metadata)}, "cost": {"evaluation_seconds": elapsed, "seconds_per_claim": elapsed / len(rows), "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if system.config.device == "cuda" else None, "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if system.config.device == "cuda" else None}}
    metrics["cost"]["prepared_input_inference_seconds"] = inference_seconds
    metrics["cost"]["prepared_input_seconds_per_claim"] = inference_seconds / len(rows)
    if output:
        write_json(str(output) + "_metrics.json", metrics)
        write_jsonl(str(output) + "_predictions.jsonl", predictions)
    return metrics


def training_rows(config):
    train, dev = load_split(config.data_path, "train"), load_split(config.data_path, "dev")
    if config.smoke:
        train, dev = smoke_subset(train, 8, config.seed), smoke_subset(dev, 4, config.seed)
    elif config.train_per_label is not None:
        train = stratified_subset(train, config.train_per_label, config.seed)
    return train, dev


def evaluate_ranker(system, rows, progress=None):
    from .model import stable_topk
    system.eval()
    rankings = []
    with torch.no_grad():
        for row in rows:
            candidates, scores = system.rank_scores(row, gradients=False)
            _, ordered = stable_topk(scores, candidates, len(candidates))
            rankings.append([candidates[i] for i in ordered])
            if progress and (len(rankings) % 25 == 0 or len(rankings) == len(rows)):
                progress({"event": "evaluation", "completed_claims": len(rankings), "total_claims": len(rows)})
    return {"retrieval": retrieval_metrics(rows, rankings)}


def phase(system, train, dev, directory, name, stage, epochs, source=None, progress=None, selection_source=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    last_path, best_path = directory / f"{name}_last.pt", directory / f"{name}_best.pt"
    config = system.config
    if config.protocol_version != PROTOCOL_VERSION:
        raise ValueError("Training requires protocol v2; legacy checkpoints are inference-only")
    system.configure_training(stage)
    optimizer = make_optimizer(system, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config.device == "cuda" and config.mixed_precision)
    completed, best_score, history = 0, -math.inf, []
    data_signature = hashlib.sha256("\n".join(r["id"] for r in train + dev).encode()).hexdigest()
    source_hash = sha256(source) if source else None
    selection_hash = sha256(selection_source) if selection_source else None
    selected = {}
    checkpoint = None
    if last_path.exists():
        checkpoint = load_checkpoint(last_path)
        require_v2(checkpoint)
        if checkpoint["mode"] != system.mode or TrainConfig(**checkpoint["config"]).to_dict() != config.to_dict() or checkpoint["data_signature"] != data_signature or checkpoint["source_checkpoint_sha256"] != source_hash:
            raise ValueError("Existing run has different configuration/data/source; choose a new output directory")
        completed, best_score, history = checkpoint["completed_epochs"], checkpoint["best_score"], checkpoint["history"]
        if checkpoint.get("selection_baseline_sha256") != selection_hash:
            raise ValueError("Existing run has a different warmup selection baseline")
        selected = {key: checkpoint[key] for key in ("selected_global_epoch", "selected_stage", "selected_validation_macro_f1", "selected_source_checkpoint_sha256")}
        # Recover a power loss between committing last.pt and linking best.pt.
        # An earlier lost winner cannot be invented from a later worse state.
        valid_best = False
        if best_path.exists():
            try:
                saved_best = load_checkpoint(best_path)
                valid_best = (saved_best.get("protocol_version") == PROTOCOL_VERSION and saved_best["mode"] == system.mode
                              and saved_best["selected_global_epoch"] == selected["selected_global_epoch"]
                              and saved_best["best_score"] == best_score)
                del saved_best
            except (OSError, RuntimeError, ValueError, KeyError, EOFError):
                valid_best = False
        if not valid_best:
            if selected["selected_global_epoch"] == checkpoint["global_epoch"]:
                copy_checkpoint(last_path, best_path)
            elif selected["selected_stage"] == "warmup" and selection_source:
                baseline = load_checkpoint(selection_source)
                require_v2(baseline)
                baseline.update(mode=system.mode, config=config.to_dict(), **selected,
                                source_checkpoint_sha256=source_hash,
                                source_checkpoint=str(Path(source).resolve()),
                                selection_baseline_sha256=selection_hash,
                                selection_source_checkpoint=str(Path(selection_source).resolve()))
                save_checkpoint(best_path, baseline)
                del baseline
            else:
                raise ValueError("Best checkpoint is missing or inconsistent; audit saved files before resuming")
    elif source:
        checkpoint = load_checkpoint(source)
        require_v2(checkpoint)
        if checkpoint["data_signature"] != data_signature:
            raise ValueError("Initialization checkpoint has different data")
        saved_config = TrainConfig.from_saved(checkpoint["config"]).to_dict()
        current_config = config.to_dict()
        for key in ("evidence_lambda", "device"):
            saved_config.pop(key)
            current_config.pop(key)
        for key in ("model_path", "data_path"):
            saved_config[key] = str(Path(saved_config[key]).resolve())
            current_config[key] = str(Path(current_config[key]).resolve())
        if saved_config != current_config:
            raise ValueError("Initialization checkpoint has different training settings or architecture")
        if stage == "continuation":
            if checkpoint["stage"] != "warmup" or checkpoint["completed_epochs"] != config.warmup_epochs:
                raise ValueError("Continuation must start from the final warmup epoch")
            history = list(checkpoint["history"])
    if checkpoint:
        system.load_state_dict(checkpoint["model"])
        if checkpoint["stage"] == stage or (checkpoint["stage"] == "warmup" and stage == "continuation"):
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        del checkpoint
    if stage == "continuation" and selection_source and not last_path.exists():
        baseline = load_checkpoint(selection_source)
        require_v2(baseline)
        if baseline["stage"] != "warmup" or baseline["data_signature"] != data_signature:
            raise ValueError("Warmup selection baseline has different data or phase")
        best_score = baseline["best_score"]
        selected = {"selected_global_epoch": baseline["selected_global_epoch"], "selected_stage": "warmup",
                    "selected_validation_macro_f1": best_score, "selected_source_checkpoint_sha256": selection_hash}
        # The selected weights can precede branching. Give them the target mode
        # and configuration so M2/A2 inference remains correctly identified.
        baseline.update(mode=system.mode, config=config.to_dict(), **selected,
                        source_checkpoint_sha256=source_hash,
                        source_checkpoint=str(Path(source).resolve()),
                        selection_baseline_sha256=selection_hash,
                        selection_source_checkpoint=str(Path(selection_source).resolve()))
        save_checkpoint(best_path, baseline)
        del baseline
    system.configure_training(stage)
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(completed, epochs):
        system.configure_training(stage)
        order = list(range(len(train)))
        # Local RNG gives the same sample order across M0/M1/M2, regardless of
        # dropout calls or extra reranker forward passes.
        order_epoch = epoch + (config.warmup_epochs if stage == "continuation" else 0)
        random.Random(config.seed + order_epoch).shuffle(order)
        if stage == "reranker":
            from .data import has_evidence_supervision
            order = [i for i in order if has_evidence_supervision(train[i])]
        if not order:
            raise ValueError("No eligible examples for this training phase")
        started, total, evidence_count, optimizer_steps = time.perf_counter(), 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        epoch_metadata = {"phase": name, "epoch": epoch + 1, "total_epochs": epochs,
                          "global_epoch": order_epoch + 1,
                          "total_verifier_epochs": config.warmup_epochs + config.continuation_epochs if stage != "reranker" else None}
        if progress:
            progress({**epoch_metadata, "event": "training", "completed_claims": 0, "total_claims": len(order)})
        for group_start in range(0, len(order), config.accumulation_steps):
            group = order[group_start:group_start + config.accumulation_steps]
            # Warm up to the configured explicit LR over the first 10% of
            # optimizer updates of this phase; no relative Adafactor schedule.
            schedule_epochs = config.reranker_epochs if stage == "reranker" else config.warmup_epochs + config.continuation_epochs
            total_updates = math.ceil(len(order) / config.accumulation_steps) * schedule_epochs
            update = order_epoch * math.ceil(len(order) / config.accumulation_steps) + group_start // config.accumulation_steps + 1
            lr = config.learning_rate * min(1.0, update / max(1, math.ceil(total_updates * 0.1)))
            for params in optimizer.param_groups:
                params["lr"] = lr
            for index in group:
                row = train[index]
                if stage == "reranker":
                    candidates, scores = system.rank_scores(row, gradients=True)
                    loss = evidence_loss(scores, row, candidates)
                    evidence_count += 1
                else:
                    result = system(row, training_stage=stage)
                    loss = -result["log_probabilities"][row["label"]]
                    if result["evidence_loss"] is not None:
                        loss = loss + config.evidence_lambda * result["evidence_loss"]
                        evidence_count += 1
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at {row['id']}")
                total += float(loss.detach())
                scaler.scale(loss / len(group)).backward()
                del loss
                if stage == "reranker":
                    del scores
                else:
                    del result
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if progress:
                progress({**epoch_metadata, "event": "training", "completed_claims": min(group_start + len(group), len(order)), "total_claims": len(order), "epoch_training_seconds": time.perf_counter() - started})
            print(f"{name} epoch {epoch + 1}/{epochs}: {min(group_start + len(group), len(order))}/{len(order)} claims", flush=True)
        training_seconds = time.perf_counter() - started
        training_memory = {"allocated_mib": torch.cuda.max_memory_allocated() / 1024**2, "reserved_mib": torch.cuda.max_memory_reserved() / 1024**2} if config.device == "cuda" else None
        validation_progress = (lambda event: progress({**event, **epoch_metadata, "event": "validation"})) if progress else None
        validation = evaluate_ranker(system, dev, validation_progress) if stage == "reranker" else evaluate(system, dev, progress=validation_progress)
        score = validation["retrieval"]["mrr"] if stage == "reranker" else validation["classification"]["macro_f1"]
        if score is None:
            raise ValueError("Validation has no aligned evidence for reranker selection")
        global_epoch = order_epoch + 1
        history.append({"epoch": epoch + 1, "global_epoch": global_epoch, "stage": stage, "phase": name, "average_loss": total / len(order), "evidence_supervised_claims": evidence_count, "optimizer_steps": optimizer_steps, "training_seconds": training_seconds, "peak_training_cuda": training_memory, "validation": validation})
        improved = score > best_score
        best_score = max(score, best_score)
        if improved:
            selected = {"selected_global_epoch": global_epoch, "selected_stage": stage,
                        "selected_validation_macro_f1": score if stage != "reranker" else None,
                        "selected_source_checkpoint_sha256": None}
        payload = {"format_version": 2, "protocol_version": PROTOCOL_VERSION, "model": system.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "config": config.to_dict(), "mode": system.mode, "phase": name, "stage": stage, "completed_epochs": epoch + 1, "global_epoch": global_epoch, "best_score": best_score, "history": history, "data_signature": data_signature, "source_checkpoint_sha256": source_hash,
                   "source_checkpoint": str(Path(source).resolve()) if source else None,
                   "selection_baseline_sha256": selection_hash,
                   "selection_source_checkpoint": str(Path(selection_source).resolve()) if selection_source else None, **selected}
        if progress:
            progress({**epoch_metadata, "event": "checkpoint", "checkpoint_model_bytes": sum(t.numel() * t.element_size() for t in payload["model"].values())})
        save_checkpoint(last_path, payload)
        del payload
        if improved:
            copy_checkpoint(last_path, best_path)
        write_json(directory / f"{name}_history.json", history)
        print(f"{name}: validation {'MRR' if stage == 'reranker' else 'Macro-F1'}={score:.4f}", flush=True)
    if not best_path.exists():
        raise RuntimeError("Best checkpoint missing")
    payload = load_checkpoint(best_path)
    system.load_state_dict(payload["model"])
    del payload, optimizer
    write_json(directory / f"{name}_selection.json", checkpoint_selection(best_path))
    return best_path


def training_cost(config, output, mode, parent=None):
    """Include shared phases for model cost comparisons, without double warmup."""
    output, parent = Path(output), Path(parent) if parent else Path(output)
    files = [(output / f"{mode.lower()}_history.json", "continuation" if mode in ("M1", "M2", "A2") else None)]
    if mode != "M0":
        files.insert(0, (parent / "reranker_history.json", None))
    if mode in ("M1", "M2", "A2"):
        files.insert(1, (parent / "warmup_history.json", None))
    records = [e for path, stage in files for e in read_json(path) if stage is None or e["stage"] == stage]
    return {"training_seconds": sum(e["training_seconds"] for e in records),
            "peak_training_allocated_mib": max((e.get("peak_training_cuda") or {}).get("allocated_mib", 0) for e in records) if config.device == "cuda" else None,
            "peak_training_reserved_mib": max((e.get("peak_training_cuda") or {}).get("reserved_mib", 0) for e in records) if config.device == "cuda" else None,
            "note": "Includes shared preparation for per-model comparison; do not sum model costs for workflow time"}


def isolated_joint_gradient(system, row):
    state = rng_state()
    system.configure_training("continuation")
    output = system(row, training_stage="continuation")
    loss = -output["log_probabilities"][row["label"]]
    gradient = torch.autograd.grad(loss, system.reranker.head.weight)[0]
    norm = float(gradient.norm())
    restore_rng(state)
    system.zero_grad(set_to_none=True)
    if norm <= 0 or not math.isfinite(norm):
        raise AssertionError("Verification loss failed to reach the reranker")
    return {"verification_only_reranker_head_gradient_norm": norm, "all_reranker_parameters_trainable": all(p.requires_grad for p in system.reranker.parameters()), "all_verifier_parameters_trainable": all(p.requires_grad for p in system.verifier.parameters())}


def isolated_stop_gradient(system, row):
    """A2 blocks verification feedback while retaining evidence supervision."""
    if system.mode != "A2" or not has_evidence_supervision(row):
        raise ValueError("A2 gradient validation requires an evidence-supervised claim")
    state = rng_state()
    try:
        system.configure_training("continuation")
        output = system(row, training_stage="continuation")
        loss = -output["log_probabilities"][row["label"]]
        verification_gradient = torch.autograd.grad(loss, system.reranker.head.weight,
                                                    allow_unused=True, retain_graph=True)[0]
        if verification_gradient is not None:
            raise AssertionError("A2 verification loss must be disconnected from the reranker")
        evidence_gradient = torch.autograd.grad(output["evidence_loss"], system.reranker.head.weight)[0]
        norm = float(evidence_gradient.norm())
        if norm <= 0 or not math.isfinite(norm):
            raise AssertionError("A2 evidence loss failed to reach the reranker")
        return {"verification_only_reranker_head_gradient_norm": 0.0,
                "evidence_only_reranker_head_gradient_norm": norm,
                "all_reranker_parameters_trainable": all(p.requires_grad for p in system.reranker.parameters()),
                "all_verifier_parameters_trainable": all(p.requires_grad for p in system.verifier.parameters())}
    finally:
        restore_rng(state)
        system.zero_grad(set_to_none=True)


def run_experiment(config, output, models=("M0", "M1", "M2"), progress=None, initialization_from=None):
    if config.protocol_version != PROTOCOL_VERSION:
        raise ValueError("Training requires protocol v2; use a new output directory")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    if not models or any(m not in ALL_MODELS for m in models) or len(set(models)) != len(models):
        raise ValueError(f"Choose distinct configurations from {ALL_MODELS}")
    train, dev = training_rows(config)
    selected_ids = {"train": [r["id"] for r in train], "dev": [r["id"] for r in dev]}
    metadata = {"config": config.to_dict(), "models": list(models), "data_manifest": read_json(Path(config.data_path) / "manifest.json"), "subset_ids": selected_ids, "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "rank-bm25", "scikit-learn")}, "hardware": {"device": config.device, "gpu": torch.cuda.get_device_name() if config.device == "cuda" else None}, "purpose": "pipeline validation only; scores are not research results" if config.smoke else "full training"}
    parent_directory = Path(initialization_from).resolve() if initialization_from else None
    if parent_directory:
        parent = read_json(parent_directory / "experiment.json")
        if parent.get("protocol_version", parent["config"].get("protocol_version", 1)) != PROTOCOL_VERSION:
            raise ValueError("Protocol v1 initialization cannot be reused in protocol v2")
        if parent["subset_ids"] != selected_ids or parent["data_manifest"] != metadata["data_manifest"]:
            raise ValueError("Initialization parent uses different data or splits")
        parent_config = TrainConfig(**parent["config"]).to_dict()
        current_config = config.to_dict()
        for key in ("model_path", "data_path"):
            parent_config[key] = str(Path(parent_config[key]).resolve())
            current_config[key] = str(Path(current_config[key]).resolve())
        # Lambda is unused in reranker pretraining and common verifier warmup.
        for key in ("evidence_lambda", "device"):
            parent_config.pop(key)
            current_config.pop(key)
        if parent_config != current_config:
            raise ValueError("Initialization parent uses different training settings or architecture")
        sources = {"reranker": parent_directory / "reranker_best.pt"}
        if any(mode in models for mode in ("M1", "M2", "A2")):
            sources["warmup"] = parent_directory / "warmup_end.pt"
            sources["warmup_best"] = parent_directory / "warmup_best.pt"
        metadata["initialization_sources"] = {name: {"path": str(path), "sha256": sha256(path)}
                                              for name, path in sources.items()}
        signature = hashlib.sha256("\n".join(r["id"] for r in train + dev).encode()).hexdigest()
        ledger_path = parent_directory / "training_complete.json"
        parent_artifacts = read_json(ledger_path)["artifacts"] if ledger_path.exists() else {}
        for name, path in sources.items():
            state = load_checkpoint(path)
            require_v2(state)
            expected_stage = "warmup" if name.startswith("warmup") else name
            if state["data_signature"] != signature or state["mode"] != "M1" or state["stage"] != expected_stage:
                raise ValueError("Initialization checkpoint has different data, mode, or phase")
            if name == "warmup" and state["completed_epochs"] != config.warmup_epochs:
                raise ValueError("Initialization must use the final warmup epoch")
            expected_hash = parent_artifacts.get(path.name)
            if expected_hash and expected_hash != metadata["initialization_sources"][name]["sha256"]:
                raise ValueError("Initialization checkpoint changed since its completion ledger")
            if name.startswith("warmup") and state["source_checkpoint_sha256"] != metadata["initialization_sources"]["reranker"]["sha256"]:
                raise ValueError("Warmup does not reference the selected reranker checkpoint")
            del state
    config_path = output / "experiment.json"
    metadata["protocol_version"] = PROTOCOL_VERSION
    metadata["checkpoint_selection"] = "dev_macro_f1_over_all_verifier_epochs; ties favor earlier global epoch"
    if config.train_per_label is not None:
        metadata["purpose"] = "reduced-data single-seed pilot; not full proposal reproduction"
    if config_path.exists():
        previous = read_json(config_path)
        if TrainConfig(**previous["config"]).to_dict() != metadata["config"] or previous["subset_ids"] != selected_ids or previous["data_manifest"] != metadata["data_manifest"] or previous.get("models") != list(models) or previous.get("initialization_sources") != metadata.get("initialization_sources"):
            raise ValueError("Output directory belongs to different experiment")
    write_json(config_path, metadata)
    seed_everything(config.seed)
    tokenizer = PairTokenizer(config.model_path, config.max_length, config.claim_max_tokens)
    results = {}
    # Each construction starts from identical verifier initialization. The warm
    # checkpoint contains both models plus the verifier optimizer and RNG state.
    if any(m != "M0" for m in models):
        system = VerificationSystem(config, "M1", tokenizer)
        initial_verifier_hash = fingerprint(system.verifier)
        if parent_directory:
            parent_checks = read_json(parent_directory / "summary.json")["M1"]["checks"]
            if parent_checks["verifier_initial_sha256"] != initial_verifier_hash:
                raise ValueError("Initialization parent did not use the same initial verifier")
            reranker_best = sources["reranker"]
        else:
            reranker_best = phase(system, train, dev, output, "reranker", "reranker", config.reranker_epochs, progress=progress)
        if any(m in models for m in ("M1", "M2", "A2")):
            if parent_directory:
                warmup_best, warmup_end = sources["warmup_best"], sources["warmup"]
            else:
                warmup_best = phase(system, train, dev, output, "warmup", "warmup", config.warmup_epochs, source=reranker_best, progress=progress)
                warmup_end = output / "warmup_end.pt"
                if warmup_end.exists():
                    if sha256(warmup_end) != sha256(output / "warmup_last.pt"):
                        raise ValueError("Pinned final warmup checkpoint changed")
                else:
                    copy_checkpoint(output / "warmup_last.pt", warmup_end)
            warmup_hash = sha256(warmup_end)
        del system
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
        for mode in (m for m in models if m in ("M1", "M2", "A2")):
            seed_everything(config.seed)
            system = VerificationSystem(config, mode, tokenizer)
            checkpoint = load_checkpoint(warmup_end)
            system.load_state_dict(checkpoint["model"])
            del checkpoint
            start_rank_hash = fingerprint(system.reranker)
            checks = {"shared_warmup_checkpoint_sha256": warmup_hash, "verifier_initial_sha256": initial_verifier_hash,
                      "reranker_initial_sha256": start_rank_hash}
            if mode == "M2" and config.train_per_label is not None:
                from .pilot import backbone_probes
                initial_probes = backbone_probes(system)
            if mode == "M2" and (config.smoke or config.train_per_label is not None):
                checks.update(isolated_joint_gradient(system, next(r for r in train if len(r["candidate_ids"]) > 1)))
            if mode == "A2":
                checks.update(isolated_stop_gradient(system, next(r for r in train if has_evidence_supervision(r))))
            best = phase(system, train, dev, output, mode.lower(), "continuation", config.continuation_epochs,
                         source=warmup_end, progress=progress, selection_source=warmup_best)
            selected_rank_hash = fingerprint(system.reranker)
            terminal = load_checkpoint(output / f"{mode.lower()}_last.pt")
            system.load_state_dict(terminal["model"])
            del terminal
            end_rank_hash = fingerprint(system.reranker)
            if mode == "M2" and config.train_per_label is not None:
                end_probes = backbone_probes(system)
                checks["first_last_attention_query_changed"] = {key: initial_probes[key] != end_probes[key] for key in initial_probes}
                if not all(checks["first_last_attention_query_changed"].values()):
                    raise AssertionError("M2 did not update first/last attention queries in both encoders")
            checks["reranker_unchanged"] = start_rank_hash == selected_rank_hash
            checks["reranker_final_sha256"] = selected_rank_hash
            checks["trained_reranker_unchanged"] = start_rank_hash == end_rank_hash
            checks["trained_reranker_final_sha256"] = end_rank_hash
            if mode == "M1" and not checks["reranker_unchanged"]:
                raise AssertionError("M1 changed frozen reranker parameters")
            if mode in ("M2", "A2") and checks["trained_reranker_unchanged"]:
                raise AssertionError(f"{mode} failed to update its reranker")
            selected_state = load_checkpoint(best)
            system.load_state_dict(selected_state["model"])
            del selected_state
            metrics = evaluate(system, dev, output / f"{mode.lower()}_dev", progress=(lambda event: progress({**event, "phase": f"{mode.lower()}_dev"})) if progress else None)
            write_json(output / f"{mode.lower()}_checks.json", checks)
            results[mode] = {"checkpoint": str(best), "metrics": metrics, "checks": checks,
                             "selection": read_json(output / f"{mode.lower()}_selection.json")}
            del system
            gc.collect()
            if config.device == "cuda":
                torch.cuda.empty_cache()
    if "A1" in models:
        seed_everything(config.seed)
        system = VerificationSystem(config, "A1", tokenizer)
        source = load_checkpoint(reranker_best)
        system.load_state_dict(source["model"])
        del source
        if fingerprint(system.verifier) != initial_verifier_hash:
            raise AssertionError("A1 verifier must start from the common initial weights")
        start_hash = fingerprint(system.reranker)
        best = phase(system, train, dev, output, "a1", "a1", config.warmup_epochs + config.continuation_epochs,
                     source=reranker_best, progress=progress)
        final_hash = fingerprint(system.reranker)
        if final_hash != start_hash:
            raise AssertionError("A1 changed its frozen reranker")
        if "M1" in results and results["M1"]["checks"]["reranker_final_sha256"] != start_hash:
            raise AssertionError("A1 and M1 must use identical sentence selection")
        metrics = evaluate(system, dev, output / "a1_dev",
                           progress=(lambda event: progress({**event, "phase": "a1_dev"})) if progress else None)
        checks = {"verifier_initial_sha256": initial_verifier_hash, "reranker_unchanged": True,
                  "reranker_initial_sha256": start_hash, "reranker_final_sha256": final_hash,
                  "selection_source_checkpoint_sha256": sha256(reranker_best), "aggregation": "uniform"}
        results["A1"] = {"checkpoint": str(best), "metrics": metrics, "checks": checks, "selection": read_json(output / "a1_selection.json")}
        write_json(output / "a1_checks.json", checks)
        del system
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
    if "M0" in models:
        seed_everything(config.seed)
        system = VerificationSystem(config, "M0", tokenizer)
        initial_hash = fingerprint(system.verifier)
        if "initial_verifier_hash" in locals() and initial_hash != initial_verifier_hash:
            raise AssertionError("M0 verifier initialization differs from M1/M2")
        best = phase(system, train, dev, output, "m0", "m0", config.warmup_epochs + config.continuation_epochs, progress=progress)
        metrics = evaluate(system, dev, output / "m0_dev", progress=(lambda event: progress({**event, "phase": "m0_dev"})) if progress else None)
        results["M0"] = {"checkpoint": str(best), "metrics": metrics, "checks": {"verifier_initial_sha256": initial_hash}}
        write_json(output / "m0_checks.json", results["M0"]["checks"])
        results["M0"]["selection"] = read_json(output / "m0_selection.json")
        del system
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
    for mode, result in results.items():
        result["training_cost"] = training_cost(config, output, mode, parent_directory)
    write_json(output / "summary.json", results)
    return results


def system_from_checkpoint(path, device=None):
    checkpoint = load_checkpoint(path)
    config = TrainConfig.from_saved(checkpoint["config"])
    if device:
        config.device = device
    system = VerificationSystem(config, checkpoint["mode"])
    system.load_state_dict(checkpoint["model"])
    system.eval()
    del checkpoint
    return system


def resume_checkpoint(path):
    checkpoint = load_checkpoint(path)
    require_v2(checkpoint)
    config = TrainConfig(**checkpoint["config"])
    name, stage, source_hash, mode = checkpoint["phase"], checkpoint["stage"], checkpoint["source_checkpoint_sha256"], checkpoint["mode"]
    recorded_source = checkpoint.get("source_checkpoint")
    selection_source = checkpoint.get("selection_source_checkpoint")
    del checkpoint
    directory = Path(path).parent
    if Path(path).name != f"{name}_last.pt":
        raise ValueError("Resume requires the phase's *_last.pt checkpoint")
    source = (Path(recorded_source) if recorded_source else directory / ("reranker_best.pt" if stage in ("warmup", "a1") else "warmup_best.pt")) if source_hash else None
    epochs = config.reranker_epochs if stage == "reranker" else config.warmup_epochs if stage == "warmup" else config.continuation_epochs if stage == "continuation" else config.warmup_epochs + config.continuation_epochs
    train, dev = training_rows(config)
    system = VerificationSystem(config, mode)
    best = phase(system, train, dev, directory, name, stage, epochs, source=source, selection_source=selection_source)
    evaluate(system, dev, directory / f"{name}_resumed_dev")
    return best
