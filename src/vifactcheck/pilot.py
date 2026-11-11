"""Deadline-aware single-seed pilot, including final evaluation and reporting."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import gc
import os
from pathlib import Path
import shutil
import time

import torch

from .config import EXPECTED_COUNTS, TrainConfig
from .data import has_evidence_supervision, load_split
from .io import read_json, sha256, write_json, write_jsonl
from .training import (evaluate, make_optimizer, run_experiment, seed_everything,
                       system_from_checkpoint, training_rows)


def utc_now():
    return datetime.now(timezone.utc)


@contextmanager
def keep_awake():
    """Temporary per-thread sleep inhibition; never alter power settings."""
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if os.name == "nt":
            kernel.SetThreadExecutionState(0x80000000)


def require_space(directory, pending_bytes=0):
    free = shutil.disk_usage(directory).free
    if free < 2 * 1024**3 + pending_bytes:
        raise RuntimeError(f"Insufficient checkpoint space: {free/1024**3:.2f} GiB free; require 2 GiB reserve plus {pending_bytes/1024**3:.2f} GiB for the next write")
    return free


def release_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def backbone_probes(system):
    import hashlib
    probes = {}
    for component in ("verifier", "reranker"):
        pairs = [(name, p) for name, p in getattr(system, component).encoder.named_parameters() if name.endswith("attention.self.query.weight")]
        if len(pairs) < 2:
            raise RuntimeError("Expected first/last attention query matrices in PhoBERT")
        for name, parameter in (pairs[0], pairs[-1]):
            probes[f"{component}.{name}"] = hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return probes


def profile_long_claims(config, train, progress):
    from .model import PairTokenizer, VerificationSystem
    seed_everything(config.seed)
    tokenizer = PairTokenizer(config.model_path, config.max_length, config.claim_max_tokens)
    workloads = []
    for row in train:
        lengths = [len(tokenizer.encode(row["claim_segmented"], row["sentences"][i]["segmented"])[0]) for i in row["candidate_ids"]]
        workloads.append((sum(lengths) + sum(sorted(lengths, reverse=True)[:config.evidence_count]), row["id"], row))
    selected = [value[2] for value in sorted(workloads, key=lambda value: (-value[0], value[1]))[:8]]
    system = VerificationSystem(config, "M2", tokenizer)
    system.configure_training("continuation")
    optimizer = make_optimizer(system, config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for i, row in enumerate(selected):
        progress({"phase": "resource_check", "event": "profiling", "completed_claims": i, "total_claims": len(selected)})
        output = system(row, training_stage="continuation")
        loss = -output["log_probabilities"][row["label"]]
        if output["evidence_loss"] is not None:
            loss = loss + config.evidence_lambda * output["evidence_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss in full-model resource check")
        scaler.scale(loss / len(selected)).backward()
        del output, loss
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    result = {"status": "passed", "claim_ids": [r["id"] for r in selected], "purpose": "eight largest tokenized candidate workloads; weights discarded", "seconds": time.perf_counter() - started, "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2, "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2, "all_encoder_parameters_trainable": all(p.requires_grad for p in system.parameters())}
    del system, optimizer, scaler, tokenizer
    release_gpu()
    return result


def run_pilot(args):
    from .cli import predict_record
    from .reporting import write_pilot_report
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(seed=42, learning_rate=2e-5, evidence_lambda=0.5,
                         train_per_label=300, data_path=str(Path(args.data).resolve()),
                         model_path=str(Path(args.model_path).resolve()))
    if not torch.cuda.is_available():
        raise RuntimeError("The laptop pilot requires CUDA; no automatic CPU fallback")
    train, dev = training_rows(config)
    manifest = read_json(Path(config.data_path)/"manifest.json")
    if {s: manifest["splits"][s]["valid_count"] for s in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise ValueError("Pilot requires the complete, pinned official prepared splits")
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    test = load_split(config.data_path, "test")
    article_sets = {"train_subset": {r["article_id"] for r in train}, "dev": {r["article_id"] for r in dev}, "test": {r["article_id"] for r in test}}
    spec = {"study_type": "reduced-data single-seed pilot", "config": config.to_dict(), "train_ids": [r["id"] for r in train], "train_labels": {str(k): v for k, v in Counter(r["label"] for r in train).items()}, "train_evidence_supervised_count": sum(has_evidence_supervision(r) for r in train), "train_alignment_status_counts": dict(Counter(r["alignment"]["status"] for r in train)), "full_split_checksums": manifest["prepared_file_sha256"], "source_code_sha256": source_hashes, "subset_article_overlap": {"train-dev": len(article_sets["train_subset"] & article_sets["dev"]), "train-test": len(article_sets["train_subset"] & article_sets["test"]), "dev-test": len(article_sets["dev"] & article_sets["test"])}, "hyperparameters": "fixed before training; no tuning; test not used for selection"}
    spec_path, status_path = output/"pilot_spec.json", output/"status.json"
    if spec_path.exists():
        previous = read_json(spec_path)
        if previous != spec:
            raise ValueError("Pilot configuration, code, data, or subset changed; use a new output directory")
    else:
        write_json(spec_path, spec)
    status = read_json(status_path) if status_path.exists() else {}
    if (output/"completion.json").exists():
        # A crash during post-completion cleanup must not rerun training after
        # completed parents have already been removed.
        ledger = read_json(output/"completion.json")
        for model, record in ledger["models"].items():
            if sha256(output/f"{model.lower()}_best.pt") != record["checkpoint_sha256"]:
                raise ValueError("Completed pilot model checkpoint changed")
        status.update({"status": "complete", "phase": "complete", "updated_utc": utc_now().isoformat(), "report": str(output/"REPORT.md")})
        write_json(status_path, status)
        return {"status": "complete", "report": str(output/"REPORT.md")}
    if status.get("status") == "complete":
        ledger = read_json(output/"completion.json")
        for model, record in ledger["models"].items():
            if sha256(output/f"{model.lower()}_best.pt") != record["checkpoint_sha256"]:
                raise ValueError("Completed pilot model checkpoint changed")
        return {"status": "complete", "report": str(output/"REPORT.md")}
    if "deadline_utc" in status:
        deadline = datetime.fromisoformat(status["deadline_utc"])
    else:
        deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")) if args.deadline_utc else utc_now() + timedelta(hours=args.hours)
    if deadline.tzinfo is None:
        raise ValueError("Pilot deadline must include a UTC timezone")
    status.update({"status": "running", "pid": os.getpid(), "started_utc": status.get("started_utc", utc_now().isoformat()), "deadline_utc": deadline.isoformat(), "train_count": len(train), "validation_count": len(dev), "test_count": len(test), "report": str(output/"REPORT.md")})
    write_json(status_path, status)
    if not (output/"REPORT.md").exists():
        (output/"REPORT.md").write_text("# M0/M1/M2 pilot — experiment in progress\n\n900 training claims; seed 42; full validation and test splits. Pilot test results are not available yet. The runner will replace this draft after all three configurations complete. See status.json and console.log for progress.\n", encoding="utf-8")

    def progress(event):
        if utc_now() >= deadline:
            raise TimeoutError("Six-hour pilot deadline reached; complete epoch checkpoints are retained. No training limits were silently changed.")
        pending = int(event.get("checkpoint_model_bytes", 0) * 1.1)
        free = require_space(output, pending)
        status.update(event)
        status.update({"status": "running", "updated_utc": utc_now().isoformat(), "free_disk_gib": free/1024**3})
        write_json(status_path, status)

    started = time.perf_counter()
    try:
        with keep_awake():
            progress({"phase": "preflight", "event": "checking"})
            model_asset = next(a for a in manifest["assets"]["files"] if a["path"] == "phobert/pytorch_model.bin")
            if sha256(Path(config.model_path)/"pytorch_model.bin") != model_asset["sha256"]:
                raise ValueError("PhoBERT weights differ from the pinned model checksum")
            if not (output/"resource_check.json").exists():
                write_json(output/"resource_check.json", profile_long_claims(config, train, progress))
            training_ledger = output/"training_complete.json"
            if training_ledger.exists():
                completed = read_json(training_ledger)
                for model, checksum in completed.items():
                    if sha256(output/f"{model.lower()}_best.pt") != checksum:
                        raise ValueError("A completed training checkpoint changed")
            else:
                results = run_experiment(config, output, progress=progress)
                write_json(training_ledger, {m: sha256(result["checkpoint"]) for m, result in results.items()})
            results = read_json(output/"summary.json")
            hashes = [results[m]["checks"]["verifier_initial_sha256"] for m in ("M0", "M1", "M2")]
            if len(set(hashes)) != 1:
                raise AssertionError("Pilot models did not share verifier initialization")
            if results["M1"]["checks"]["shared_warmup_checkpoint_sha256"] != results["M2"]["checks"]["shared_warmup_checkpoint_sha256"]:
                raise AssertionError("M1/M2 did not share the same initialization checkpoint")
            completion = {}
            for model in ("M0", "M1", "M2"):
                model_name = model.lower()
                checksum = sha256(output/f"{model_name}_best.pt")
                marker = output/f"{model_name}_test_complete.json"
                if marker.exists():
                    record = read_json(marker)
                    if record["checkpoint_sha256"] != checksum or any(sha256(output/name) != value for name, value in record["artifact_sha256"].items()):
                        raise ValueError("Completed test evaluation changed")
                else:
                    progress({"phase": f"{model_name}_test", "event": "loading"})
                    system = system_from_checkpoint(output/f"{model_name}_best.pt", "cuda")
                    torch.cuda.reset_peak_memory_stats()
                    metrics = evaluate(system, test, output/f"{model_name}_test", progress=lambda event: progress({**event, "phase": f"{model_name}_test"}))
                    prepared = system.prediction(dev[0])
                    raw = predict_record(system, dev[0]["claim"], dev[0]["context"])
                    difference = max(abs(raw["probabilities"][key]-prepared["probabilities"][key]) for key in raw["probabilities"])
                    if difference > 1e-6:
                        raise AssertionError("Reload/raw-input prediction consistency check failed")
                    write_json(output/f"{model_name}_example_prediction.json", raw)
                    files = [f"{model_name}_test_metrics.json", f"{model_name}_test_predictions.jsonl", f"{model_name}_example_prediction.json"]
                    record = {"checkpoint_sha256": checksum, "test_count": metrics["classification"]["count"], "raw_input_max_probability_difference": difference, "artifact_sha256": {name: sha256(output/name) for name in files}}
                    write_json(marker, record)
                    del system
                    release_gpu()
                completion[model] = record
            progress({"phase": "report", "event": "analysis"})
            write_pilot_report(output)
            elapsed = time.perf_counter()-started
            write_json(output/"completion.json", {"status": "complete", "models": completion, "this_session_seconds": elapsed, "started_utc": status["started_utc"], "completed_utc": utc_now().isoformat(), "purpose": spec["study_type"]})
            # Only this new pilot directory is eligible for cleanup. Final best
            # models are retained; completed workflow resume uses its ledgers.
            removed = []
            for name in ("reranker_last.pt", "reranker_best.pt", "warmup_last.pt", "warmup_best.pt", "m0_last.pt", "m1_last.pt", "m2_last.pt"):
                target = (output/name).resolve()
                if target.parent != output:
                    raise ValueError("Checkpoint cleanup escaped the pilot directory")
                if target.exists():
                    removed.append({"file": name, "sha256": sha256(target)})
                    target.unlink()
            write_json(output/"checkpoint_cleanup.json", {"removed": removed, "retained": ["m0_best.pt", "m1_best.pt", "m2_best.pt"]})
            status.update({"status": "complete", "phase": "complete", "event": "complete", "updated_utc": utc_now().isoformat(), "this_session_seconds": elapsed})
            write_json(status_path, status)
            return {"status": "complete", "report": str(output/"REPORT.md"), "results": str(output/"results.csv"), "this_session_seconds": elapsed}
    except BaseException as error:
        failure = {"status": "failed", "exception": type(error).__name__, "message": str(error), "phase": status.get("phase"), "updated_utc": utc_now().isoformat()}
        if isinstance(error, torch.cuda.OutOfMemoryError):
            failure["gpu_memory"] = {"allocated_mib": torch.cuda.memory_allocated()/1024**2, "reserved_mib": torch.cuda.memory_reserved()/1024**2}
            failure["architecture_changed"] = False
        status.update(failure)
        write_json(status_path, status)
        write_json(output/"failure.json", failure)
        raise
