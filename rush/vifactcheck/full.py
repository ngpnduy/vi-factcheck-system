"""Resumable full-data tuning, three-seed training and final evaluation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil

import torch

from .config import DATASET_REVISION, EXPECTED_COUNTS, MODEL_REVISION, TrainConfig
from .data import has_evidence_supervision, load_split
from .io import read_json, read_jsonl, sha256, write_json
from .model import PairTokenizer, VerificationSystem
from .pilot import backbone_probes, keep_awake, profile_long_claims, release_gpu, require_space
from .training import (evaluate, fingerprint, isolated_joint_gradient, load_checkpoint,
                       phase, run_experiment, seed_everything, system_from_checkpoint)

SEEDS = (42, 13, 2026)
MODELS = ("M0", "M1", "M2")
LEARNING_RATES = (1e-5, 2e-5)
LAMBDAS = (0.1, 0.5, 1.0)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_run(directory):
    """An OS lock prevents two processes from writing the same experiment."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".runner.lock").open("a+b", buffering=0) as stream:
        # Inspect size without reading a byte another Windows writer may lock.
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("This full experiment already has an active runner") from error
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def select_trial(trials, parameter):
    # Python max retains the first search-order entry on exact ties.
    return max(trials, key=lambda row: row["validation_macro_f1"])[parameter]


def checkpoint_cleanup(directory, names):
    """Remove only explicitly named superseded files in a completed new job."""
    directory = Path(directory).resolve()
    if not (directory / "training_complete.json").is_file():
        raise ValueError("Checkpoint cleanup requires a completed training ledger")
    removed = []
    for name in names:
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Cleanup must use a checkpoint basename")
        target = (directory / name).resolve()
        if target.parent != directory:
            raise ValueError("Checkpoint cleanup escaped the job directory")
        if target.is_file():
            removed.append({"file": name, "sha256": sha256(target)})
            target.unlink()
    if removed:
        previous = read_json(directory / "checkpoint_cleanup.json") if (directory / "checkpoint_cleanup.json").exists() else []
        write_json(directory / "checkpoint_cleanup.json", previous + removed)


def verify_completed_training(directory):
    directory = Path(directory)
    ledger = read_json(directory / "training_complete.json")
    for name, expected in ledger["artifacts"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Completed training artifact changed: {directory / name}")
    return read_json(directory / "summary.json")


def seal_training(directory, results, parents=()):
    directory = Path(directory)
    names = ["summary.json", "experiment.json"]
    for model in results:
        names += [f"{model.lower()}_best.pt", f"{model.lower()}_dev_metrics.json",
                  f"{model.lower()}_dev_predictions.jsonl", f"{model.lower()}_checks.json"]
    names += list(parents)
    write_json(directory / "training_complete.json", {"completed_utc": now(), "artifacts": {name: sha256(directory / name) for name in names}})


def joint_checks(checkpoint_path, warmup_path, row):
    system = system_from_checkpoint(checkpoint_path, "cuda")
    checks = isolated_joint_gradient(system, row)
    end = backbone_probes(system)
    warmup = load_checkpoint(warmup_path)
    start = {}
    for probe_name in end:
        # Probe names are relative to PairEncoder.encoder; the full checkpoint
        # also includes that enclosing module's encoder attribute.
        component, encoder_name = probe_name.split(".", 1)
        checkpoint_key = f"{component}.encoder.{encoder_name}"
        start[probe_name] = hashlib.sha256(warmup["model"][checkpoint_key].contiguous().numpy().tobytes()).hexdigest()
    del warmup
    checks["first_last_attention_query_changed"] = {key: start[key] != end[key] for key in end}
    if not all(checks["first_last_attention_query_changed"].values()):
        raise AssertionError("M2 did not update both encoders' first/last attention query matrices")
    del system
    release_gpu()
    return checks


def training_job(config, directory, models, progress, train):
    directory = Path(directory)
    if (directory / "training_complete.json").exists():
        result = verify_completed_training(directory)
        checkpoint_cleanup(directory, [f"{name}_last.pt" for name in ("reranker", "warmup", "m0", "m1", "m2")])
        return result
    result = run_experiment(config, directory, models, progress=progress)
    if "M2" in result:
        checks = joint_checks(directory / "m2_best.pt", directory / "warmup_best.pt", next(r for r in train if len(r["candidate_ids"]) > 1))
        result["M2"]["checks"].update(checks)
        if result["M2"]["checks"]["reranker_unchanged"]:
            raise AssertionError("M2 reranker did not change")
        write_json(directory / "m2_checks.json", result["M2"]["checks"])
        write_json(directory / "summary.json", result)
    parents = ("reranker_best.pt", "warmup_best.pt") if "M1" in result else ()
    seal_training(directory, result, parents)
    # Keep the parent best checkpoints used by resume/provenance. Last states
    # are discarded only after the completed-job ledger is durable.
    checkpoint_cleanup(directory, [f"{name}_last.pt" for name in ("reranker", "warmup", "m0", "m1", "m2")])
    return result


def lambda_job(config, directory, parent_directory, progress, train, dev):
    directory, parent_directory = Path(directory), Path(parent_directory)
    if (directory / "training_complete.json").exists():
        result = verify_completed_training(directory)
        checkpoint_cleanup(directory, ["m2_last.pt"])
        return result
    directory.mkdir(parents=True, exist_ok=True)
    source = parent_directory / "warmup_best.pt"
    parent = read_json(parent_directory / "summary.json")["M1"]
    seed_everything(config.seed)
    system = VerificationSystem(config, "M2", PairTokenizer(config.model_path, config.max_length, config.claim_max_tokens))
    state = load_checkpoint(source)
    system.load_state_dict(state["model"])
    del state
    start_hash = fingerprint(system.reranker)
    checkpoint = phase(system, train, dev, directory, "m2", "continuation", config.continuation_epochs, source=source, progress=progress)
    checks = {"verifier_initial_sha256": parent["checks"]["verifier_initial_sha256"],
              "shared_warmup_checkpoint_sha256": sha256(source),
              "reranker_unchanged": start_hash == fingerprint(system.reranker)}
    if checks["reranker_unchanged"]:
        raise AssertionError("M2 reranker did not change during lambda tuning")
    metrics = evaluate(system, dev, directory / "m2_dev", progress=lambda event: progress({**event, "phase": "m2_dev"}))
    del system
    release_gpu()
    checks.update(joint_checks(checkpoint, source, next(r for r in train if len(r["candidate_ids"]) > 1)))
    result = {"M2": {"checkpoint": str(checkpoint), "metrics": metrics, "checks": checks}}
    metadata = read_json(parent_directory / "experiment.json")
    metadata.update({"config": config.to_dict(), "models": ["M2"], "purpose": "full-data validation-only lambda tuning",
                     "shared_initialization_from": str(source), "shared_initialization_sha256": sha256(source)})
    write_json(directory / "experiment.json", metadata)
    write_json(directory / "summary.json", result)
    write_json(directory / "m2_checks.json", checks)
    seal_training(directory, result)
    checkpoint_cleanup(directory, ["m2_last.pt"])
    return result


def collect_seed42(directory, sources, config):
    """Reuse seed-42 tuning winners; do not retrain selected checkpoints."""
    directory = Path(directory)
    if (directory / "training_complete.json").exists():
        return verify_completed_training(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for mode, source_dir in sources.items():
        source_dir = Path(source_dir)
        result[mode] = read_json(source_dir / "summary.json")[mode]
        for suffix in ("best.pt", "dev_metrics.json", "dev_predictions.jsonl", "checks.json"):
            name = f"{mode.lower()}_{suffix}"
            target = directory / name
            if target.exists():
                if sha256(target) != sha256(source_dir / name):
                    raise ValueError("Collected checkpoint/artifact differs from its tuning winner")
                continue
            if suffix == "best.pt":
                try:
                    os.link(source_dir / name, target)
                except OSError:
                    shutil.copy2(source_dir / name, target)
            else:
                shutil.copy2(source_dir / name, target)
        result[mode] = {**result[mode], "checkpoint": str(directory / f"{mode.lower()}_best.pt"), "training_source": str(source_dir)}
    metadata = read_json(Path(sources["M0"]) / "experiment.json")
    metadata.update({"config": config.to_dict(), "models": list(MODELS), "purpose": "full-data selected seed-42 models; tuning winners reused",
                     "training_sources": {m: str(p) for m, p in sources.items()}, "evidence_lambda_applies_to": ["M2"]})
    write_json(directory / "experiment.json", metadata)
    write_json(directory / "summary.json", result)
    seal_training(directory, result)
    return result


def verify_initialization(summary):
    if len({summary[m]["checks"]["verifier_initial_sha256"] for m in MODELS}) != 1:
        raise AssertionError("The three models did not share initial verifier weights")
    if summary["M1"]["checks"]["shared_warmup_checkpoint_sha256"] != summary["M2"]["checks"]["shared_warmup_checkpoint_sha256"]:
        raise AssertionError("M1 and M2 did not share their initialization checkpoint")
    if not summary["M1"]["checks"]["reranker_unchanged"]:
        raise AssertionError("M1 changed its frozen reranker")


def run_full(args):
    output = Path(args.output).resolve()
    with exclusive_run(output):
        return _run_full(args, output)


def _run_full(args, output):
    from .assets import ensure_assets
    from .cli import predict_record
    from .full_reporting import write_full_report
    mirror = Path(args.status_dir).resolve() if args.status_dir else output
    mirror.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(model_path=str(Path(args.model_path).resolve()), data_path=str(Path(args.data).resolve()))
    manifest = read_json(Path(config.data_path) / "manifest.json")
    plan = {"format_version": 1, "study": "full-data M0/M1/M2 experiment", "seeds": list(SEEDS), "models": list(MODELS),
            "learning_rates": list(LEARNING_RATES), "evidence_lambdas": list(LAMBDAS), "tuning_seed": 42, "selection_split": "dev",
            "config_defaults": config.to_dict(), "data_checksums": manifest["prepared_file_sha256"],
            "source_code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            "segmenter_path": str(Path(args.segmenter_path).resolve()),
            "output": str(output), "lambda_initialization": "same selected-LR warmup checkpoint for all three lambda trials",
            "seed42_reuse": "selected M1/M2 tuning checkpoints are reused; M0 trains independently from identical initial weights",
            "test_protocol": "all training and settings finalized before nine test evaluations; test never selects hyperparameters",
            "checkpoint_policy": "completed jobs keep their best checkpoints and required parent best checkpoints; incomplete jobs retain resume states",
            "excluded": ["A1", "A2", "D0", "manual contradiction annotation"]}
    plan_path = output / "full_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Full experiment code/configuration/data changed; use a new output folder")
    write_json(plan_path, plan)
    snapshot = output / "source_snapshot" / "vifactcheck"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, checksum in plan["source_code_sha256"].items():
        target = snapshot / name
        if not target.exists():
            shutil.copy2(Path(__file__).parent / name, target)
        if sha256(target) != checksum:
            raise ValueError("Frozen source snapshot changed")
    write_json(mirror / "full_plan.json", plan)
    write_json(mirror / "LOCATION.json", {"checkpoint_and_result_directory": str(output), "status_directory": str(mirror)})
    status = read_json(output / "status.json") if (output / "status.json").exists() else {}
    if (output / "completion.json").exists():
        completed = read_json(output / "completion.json")
        for seed in SEEDS:
            location = output / "seeds" / f"seed{seed}"
            verify_completed_training(location)
            for mode in MODELS:
                record = read_json(location / f"{mode.lower()}_test_complete.json")
                if sha256(location / f"{mode.lower()}_best.pt") != record["checkpoint_sha256"] or any(sha256(location / name) != checksum for name, checksum in record["artifact_sha256"].items()):
                    raise ValueError("Completed test artifact changed")
        for name, checksum in completed["report_checksums"].items():
            if sha256(output / name) != checksum:
                raise ValueError("Completed full-experiment report changed")
        status.update({"status": "complete", "phase": "complete", "event": "complete", "updated_utc": now()})
        write_json(output / "status.json", status)
        write_json(mirror / "status.json", status)
        if mirror != output:
            for name in ("REPORT.md", "results.csv", "comparison.json", "paired_bootstrap.json", "completion.json", "selected_hyperparameters.json"):
                shutil.copy2(output / name, mirror / name)
        return completed
    status.update({"status": "running", "pid": os.getpid(), "started_utc": status.get("started_utc", now()), "updated_utc": now(),
                   "output": str(output), "train_count": EXPECTED_COUNTS["train"], "validation_count": EXPECTED_COUNTS["dev"],
                   "test_count": EXPECTED_COUNTS["test"], "total_training_tasks": 8, "completed_training_tasks": 0})
    status["initial_estimated_total_hours"] = [60, 80]

    def progress(event):
        if (mirror / "STOP_REQUESTED").exists() or (output / "STOP_REQUESTED").exists():
            raise InterruptedError("Stop requested; resume restarts any incomplete epoch from the last complete epoch")
        pending = int(event.get("checkpoint_model_bytes", 0) * 1.15)
        free = require_space(output, pending)
        status.update(event)
        status.update({"status": "running", "updated_utc": now(), "free_disk_gib": free / 1024**3})
        write_json(output / "status.json", status)
        if mirror != output:
            write_json(mirror / "status.json", status)

    def task(name, cfg):
        for key in ("epoch", "total_epochs", "completed_claims", "total_claims", "epoch_training_seconds", "checkpoint_model_bytes"):
            status.pop(key, None)
        progress({"task": name, "seed": cfg.seed, "learning_rate": cfg.learning_rate, "evidence_lambda": cfg.evidence_lambda,
                  "phase": "starting", "event": "checking"})
        return lambda event: progress({**event, "task": name})

    def finish_task():
        status["completed_training_tasks"] += 1
        progress({"event": "training_task_complete"})

    try:
        with keep_awake():
            progress({"task": "preflight", "phase": "checking", "event": "checking"})
            if not torch.cuda.is_available():
                raise RuntimeError("Full laptop experiment requires CUDA; no automatic CPU or architecture fallback")
            if {s: manifest["splits"][s]["valid_count"] for s in EXPECTED_COUNTS} != EXPECTED_COUNTS:
                raise ValueError("Full experiment requires complete official prepared splits")
            if manifest["assets"]["dataset_revision"] != DATASET_REVISION or manifest["assets"]["model_revision"] != MODEL_REVISION:
                raise ValueError("Prepared data uses different dataset/model revisions")
            for split, checksum in manifest["prepared_file_sha256"].items():
                if sha256(Path(config.data_path) / f"{split}.jsonl") != checksum:
                    raise ValueError(f"Prepared split changed: {split}")
            if ensure_assets(Path(args.segmenter_path).resolve().parent, offline=True) != manifest["assets"]:
                raise ValueError("Asset manifest changed")
            expected_model = next(a["sha256"] for a in manifest["assets"]["files"] if a["path"] == "phobert/pytorch_model.bin")
            if sha256(Path(config.model_path) / "pytorch_model.bin") != expected_model:
                raise ValueError("PhoBERT model weights changed")
            train, dev = load_split(config.data_path, "train"), load_split(config.data_path, "dev")
            if len(train) != 5062 or len(dev) != 723:
                raise ValueError("Full experiment cannot use a training subset")
            status["evidence_supervised_train_count"] = sum(has_evidence_supervision(row) for row in train)
            if not (output / "resource_check.json").exists():
                write_json(output / "resource_check.json", profile_long_claims(config, train, progress))
            trials = []
            for rate in LEARNING_RATES:
                cfg = TrainConfig(**{**config.to_dict(), "learning_rate": rate})
                directory = output / "tuning" / f"lr_{rate}"
                result = training_job(cfg, directory, ("M1",), task(f"lr_{rate}", cfg), train)
                trials.append({"learning_rate": rate, "validation_macro_f1": result["M1"]["metrics"]["classification"]["macro_f1"], "directory": str(directory)})
                write_json(output / "learning_rate_trials.json", trials)
                finish_task()
            chosen_lr = select_trial(trials, "learning_rate")
            parent_directory = output / "tuning" / f"lr_{chosen_lr}"
            lambda_trials = []
            for weight in LAMBDAS:
                cfg = TrainConfig(**{**config.to_dict(), "learning_rate": chosen_lr, "evidence_lambda": weight})
                directory = output / "tuning" / f"lambda_{weight}"
                result = lambda_job(cfg, directory, parent_directory, task(f"lambda_{weight}", cfg), train, dev)
                lambda_trials.append({"evidence_lambda": weight, "validation_macro_f1": result["M2"]["metrics"]["classification"]["macro_f1"], "directory": str(directory)})
                write_json(output / "lambda_trials.json", lambda_trials)
                finish_task()
            chosen_lambda = select_trial(lambda_trials, "evidence_lambda")
            selected = {"seed": 42, "split": "dev", "learning_rate": chosen_lr, "evidence_lambda": chosen_lambda,
                        "learning_rate_trials": trials, "lambda_trials": lambda_trials}
            write_json(output / "selected_hyperparameters.json", selected)
            final_config = TrainConfig(**{**config.to_dict(), "learning_rate": chosen_lr, "evidence_lambda": chosen_lambda})
            m0_directory = output / "seeds" / "seed42-m0"
            training_job(final_config, m0_directory, ("M0",), task("seed42_m0", final_config), train)
            finish_task()
            collect_seed42(output / "seeds" / "seed42", {"M0": m0_directory, "M1": parent_directory,
                           "M2": output / "tuning" / f"lambda_{chosen_lambda}"}, final_config)
            for seed in (13, 2026):
                cfg = TrainConfig(**{**final_config.to_dict(), "seed": seed})
                training_job(cfg, output / "seeds" / f"seed{seed}", MODELS, task(f"seed{seed}", cfg), train)
                finish_task()
            # Test records are loaded only after hyperparameters and all models
            # have been finalized. No test-driven retries or checkpoint selection.
            test = load_split(config.data_path, "test")
            progress({"task": "final_evaluation", "phase": "checking", "event": "checking", "completed_test_models": 0, "total_test_models": 9})
            for seed in SEEDS:
                directory = output / "seeds" / f"seed{seed}"
                verify_initialization(verify_completed_training(directory))
                for mode in MODELS:
                    name = mode.lower()
                    checkpoint_path = directory / f"{name}_best.pt"
                    marker = directory / f"{name}_test_complete.json"
                    checksum = sha256(checkpoint_path)
                    if marker.exists():
                        record = read_json(marker)
                        if record["checkpoint_sha256"] != checksum or any(sha256(directory / n) != h for n, h in record["artifact_sha256"].items()):
                            raise ValueError("Completed test evaluation changed")
                    else:
                        progress({"seed": seed, "model": mode, "phase": f"{name}_test", "event": "loading", "completed_claims": 0, "total_claims": len(test)})
                        system = system_from_checkpoint(checkpoint_path, "cuda")
                        torch.cuda.reset_peak_memory_stats()
                        metrics = evaluate(system, test, directory / f"{name}_test", progress=lambda e: progress({**e, "phase": f"{name}_test"}))
                        reference = next(p for p in read_jsonl(directory / f"{name}_dev_predictions.jsonl") if p["id"] == dev[0]["id"])
                        prepared = system.prediction(dev[0])
                        raw = predict_record(system, dev[0]["claim"], dev[0]["context"], args.segmenter_path)
                        reload_difference = max(abs(prepared["probabilities"][key] - reference["probabilities"][key]) for key in prepared["probabilities"])
                        raw_difference = max(abs(raw["probabilities"][key] - prepared["probabilities"][key]) for key in raw["probabilities"])
                        if max(reload_difference, raw_difference) > 1e-6:
                            raise AssertionError("Checkpoint reload/raw-text prediction changed probabilities")
                        write_json(directory / f"{name}_example_prediction.json", raw)
                        files = [f"{name}_test_metrics.json", f"{name}_test_predictions.jsonl", f"{name}_example_prediction.json"]
                        record = {"checkpoint_sha256": checksum, "test_count": metrics["classification"]["count"],
                                  "reload_max_probability_difference": reload_difference, "raw_input_max_probability_difference": raw_difference,
                                  "artifact_sha256": {n: sha256(directory / n) for n in files}}
                        write_json(marker, record)
                        del system
                        release_gpu()
                    if record["test_count"] != EXPECTED_COUNTS["test"]:
                        raise ValueError("Test evaluation is incomplete")
                    status["completed_test_models"] += 1
                    progress({"event": "test_model_complete"})
            progress({"task": "report", "phase": "report", "event": "analysis"})
            write_full_report(output)
            completed = {"status": "complete", "started_utc": status["started_utc"], "completed_utc": now(), "seeds": list(SEEDS),
                         "models": list(MODELS), "selected_hyperparameters": selected, "report": str(output / "REPORT.md"),
                         "report_checksums": {n: sha256(output / n) for n in ("REPORT.md", "results.csv", "comparison.json", "paired_bootstrap.json")}}
            write_json(output / "completion.json", completed)
            status.update({"status": "complete", "phase": "complete", "event": "complete", "updated_utc": now()})
            write_json(output / "status.json", status)
            write_json(mirror / "status.json", status)
            if mirror != output:
                for name in ("REPORT.md", "results.csv", "comparison.json", "paired_bootstrap.json", "completion.json", "selected_hyperparameters.json"):
                    shutil.copy2(output / name, mirror / name)
            return completed
    except BaseException as error:
        failure = {"status": "stopped" if isinstance(error, InterruptedError) else "failed", "exception": type(error).__name__,
                   "message": str(error), "updated_utc": now(), "architecture_changed": False}
        if isinstance(error, torch.cuda.OutOfMemoryError):
            failure["gpu_memory"] = {"allocated_mib": torch.cuda.memory_allocated()/1024**2, "reserved_mib": torch.cuda.memory_reserved()/1024**2}
        status.update(failure)
        write_json(output / "status.json", status)
        write_json(mirror / "status.json", status)
        write_json(output / "failure.json", failure)
        raise
