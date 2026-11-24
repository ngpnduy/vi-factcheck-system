"""Resumable Baseline/M2 benchmark with five verifier epochs per system."""
from dataclasses import replace
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import torch

from .config import DATASET_REVISION, EXPECTED_COUNTS, TrainConfig
from .data import load_split
from .full import (LAMBDAS, LEARNING_RATES, SEEDS, checkpoint_cleanup, exclusive_run,
                   now, seal_training, training_job, verify_completed_training)
from .io import read_json, read_jsonl, sha256, write_json
from .model import VerificationSystem
from .pilot import profile_long_claims, release_gpu, require_space
from .training import (evaluate, fingerprint, phase, seed_everything, system_from_checkpoint,
                       training_cost, training_rows)

BACKBONES = {
    "mbert": ("google-bert/bert-base-multilingual-cased", "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"),
    "xlmr": ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"),
}
ASSET_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
FIXED_LR, FIXED_LAMBDA = 1e-5, 0.1


def prepare_backbone(backbone, cache):
    from huggingface_hub import hf_hub_download

    model_id, revision = BACKBONES[backbone]
    target = Path(cache).resolve() / "backbones" / backbone / revision
    with exclusive_run(target):
        if (target / "assets.json").is_file():
            return backbone_assets(backbone, cache)[1]
        files = {}
        for name in ASSET_FILES:
            # If this preparation was interrupted, the Hub client reuses its download metadata.
            path = Path(hf_hub_download(model_id, name, revision=revision, local_dir=target))
            files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        manifest = {"backbone": backbone, "model_id": model_id, "revision": revision, "files": files}
        write_json(target / "assets.json", manifest)
    return manifest


def backbone_assets(backbone, cache):
    model_id, revision = BACKBONES[backbone]
    directory = Path(cache).resolve() / "backbones" / backbone / revision
    manifest_path = directory / "assets.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run 'vifactcheck benchmark prepare --backbone {backbone}' first")
    manifest = read_json(manifest_path)
    if ((manifest.get("backbone"), manifest.get("model_id"), manifest.get("revision"))
            != (backbone, model_id, revision) or set(manifest.get("files", {})) != set(ASSET_FILES)):
        raise ValueError("Backbone asset manifest does not match the pinned revision")
    for name, entry in manifest["files"].items():
        if sha256(directory / name) != entry["sha256"]:
            raise ValueError(f"Backbone asset checksum mismatch: {directory / name}")
    return directory, manifest


def baseline_job(config, directory, train, dev, progress):
    if (directory / "training_complete.json").is_file():
        result = verify_completed_training(directory)
        checkpoint_cleanup(directory, ["baseline_last.pt"])
        return result
    seed_everything(config.seed)
    system = VerificationSystem(config, "BASELINE")
    initial = fingerprint(system.verifier)
    checkpoint = phase(system, train, dev, directory, "baseline", "baseline",
                       config.warmup_epochs + config.continuation_epochs, progress=progress)
    trained = fingerprint(system.verifier)
    if initial == trained:
        raise AssertionError("Baseline verifier did not update")
    checks = {"verifier_initial_sha256": initial, "selected_verifier_sha256": trained}
    metrics = evaluate(system, dev, directory / "baseline_dev")
    del system
    release_gpu()
    result = {"BASELINE": {
        "checkpoint": str(checkpoint), "metrics": metrics, "checks": checks,
        "selection": read_json(directory / "baseline_selection.json"),
        "training_cost": training_cost(config, directory, "BASELINE"),
    }}
    write_json(directory / "baseline_checks.json", checks)
    write_json(directory / "experiment.json", {"config": config.to_dict(), "models": ["BASELINE"]})
    write_json(directory / "summary.json", result)
    seal_training(directory, result, ("baseline_history.json",))
    checkpoint_cleanup(directory, ["baseline_last.pt"])
    return result


def run_benchmark(args):
    output = Path(args.output).resolve()
    with exclusive_run(output):
        try:
            return run_locked(args, output)
        except Exception as error:
            failure = {"status": "failed", "time_utc": now(), "exception": type(error).__name__,
                       "message": str(error)}
            if isinstance(error, torch.cuda.OutOfMemoryError):
                failure["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
                failure["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
            write_json(output / "failure.json", failure)
            if not (output / "completion.json").is_file():
                previous = read_json(output / "status.json") if (output / "status.json").is_file() else {}
                write_json(output / "status.json", {**previous, **failure})
            raise


def run_locked(args, output):
    smoke = args.benchmark_action == "smoke"
    fixed = smoke or args.fixed_hyperparameters
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA in this job")
    if not smoke and args.device != "cuda":
        raise ValueError("Full backbone benchmark requires a CUDA job")
    model_path, assets = backbone_assets(args.backbone, args.cache)
    config = TrainConfig(
        backbone=args.backbone, model_path=str(model_path), data_path=str(Path(args.data).resolve()),
        device=args.device, smoke=smoke, reranker_architecture="listwise",
        learning_rate=FIXED_LR, evidence_lambda=FIXED_LAMBDA,
        reranker_epochs=1 if smoke else 3, warmup_epochs=1 if smoke else 3,
        continuation_epochs=1 if smoke else 2,
    )
    data = read_json(Path(config.data_path) / "manifest.json")
    if (data["assets"]["dataset_revision"] != DATASET_REVISION or
            {split: data["splits"][split]["valid_count"] for split in EXPECTED_COUNTS} != EXPECTED_COUNTS):
        raise ValueError("Run requires the pinned official prepared dataset and split counts")
    source = Path(__file__).resolve().parent
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True)
    seeds = [42] if smoke else list(SEEDS)
    plan = {
        "kind": "smoke" if smoke else "benchmark", "config": config.to_dict(), "seeds": seeds,
        "hyperparameter_policy": "fixed-before-training" if fixed else "seed42-dev-search",
        "search": None if fixed else {"learning_rates": list(LEARNING_RATES), "lambdas": list(LAMBDAS)},
        "checkpoint_selection": "highest dev Macro-F1 over all verifier epochs, earliest on ties",
        "ranking_selection": "highest dev MRR, earliest on ties",
        "source_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": {p.name: sha256(p) for p in sorted(source.glob("*.py"))},
        "assets": assets, "data_manifest": data,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cuda_build": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if args.device == "cuda" else None,
                        "packages": dict(sorted({d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                                 if d.metadata["Name"]}.items()))},
    }
    plan_path = output / "plan.json"
    if plan_path.is_file():
        if read_json(plan_path) != plan:
            raise ValueError("Source, assets, environment, or benchmark options changed; use a new output directory")
    else:
        if (output / "jobs").exists():
            raise ValueError("Output has jobs without a benchmark plan")
        minimum_gib = 35 if args.backbone == "mbert" else 60
        if not smoke and shutil.disk_usage(output).free < minimum_gib * 1024**3:
            raise RuntimeError(f"Need at least {minimum_gib} GiB free at the benchmark output")
        write_json(plan_path, plan)
    from .benchmark_report import export_locked, write_report

    if (output / "completion.json").is_file():
        return export_locked(output)
    state = {"status": "running", "backbone": args.backbone, "kind": plan["kind"],
             "job_id": os.environ.get("SLURM_JOB_ID")}
    timings_path = output / "job_timings.json"
    timings = read_json(timings_path) if timings_path.is_file() else {}

    def progress(event):
        if event.get("event") == "checkpoint":
            require_space(output, int(event["checkpoint_model_bytes"] * 1.15))
        write_json(output / "status.json", {**state, **event, "updated_utc": now(),
                                           "completed_training_jobs": len(timings)})

    def job(relative, cfg, mode, parent=None):
        directory = output / "jobs" / relative
        train, dev = training_rows(cfg)
        callback = lambda event: progress({**event, "job": relative, "seed": cfg.seed, "system": mode})
        callback({"event": "job_start"})
        started = time.monotonic()
        if mode == "BASELINE":
            result = baseline_job(cfg, directory, train, dev, callback)
        else:
            result = training_job(cfg, directory, (mode,), callback, train, initialization_from=parent)
        if relative not in timings:
            timings[relative] = {"observed_seconds_this_attempt": time.monotonic() - started,
                                 **result[mode]["training_cost"]}
            write_json(timings_path, timings)
        callback({"event": "job_complete"})
        return directory, result[mode]

    if smoke and config.device == "cuda" and not (output / "resource_probe.json").is_file():
        write_json(output / "resource_probe.json",
                   profile_long_claims(config, load_split(config.data_path, "train"), progress))

    trial_rows = {"baseline_lr": [], "m1_lr": [], "m2_lambda": []}
    if fixed:
        selected = [(seed, job(f"seed-{seed}/baseline", replace(config, seed=seed), "BASELINE"),
                     job(f"seed-{seed}/m2", replace(config, seed=seed), "M2")) for seed in seeds]
        choices = {"baseline_lr": FIXED_LR, "m2_lr": FIXED_LR, "evidence_lambda": FIXED_LAMBDA}
    else:
        def trial(key, parameter, value, cfg, mode, parent=None):
            path, result = job(f"tuning/{key}-{value:g}", cfg, mode, parent)
            item = {"job": str(path.relative_to(output)), parameter: value,
                    "dev_macro_f1": result["metrics"]["classification"]["macro_f1"]}
            trial_rows[key].append(item)
            write_json(output / "tuning_trials.json", trial_rows)
            return path, result

        baseline_trials = [trial("baseline_lr", "lr", lr, replace(config, learning_rate=lr), "BASELINE")
                           for lr in LEARNING_RATES]
        m1_trials = [trial("m1_lr", "lr", lr, replace(config, learning_rate=lr), "M1")
                     for lr in LEARNING_RATES]
        baseline_index = best_trial(trial_rows["baseline_lr"])
        m1_index = best_trial(trial_rows["m1_lr"])
        m2_lr = LEARNING_RATES[m1_index]
        m2_config = replace(config, learning_rate=m2_lr)
        m2_trials = [trial("m2_lambda", "lambda", weight, replace(m2_config, evidence_lambda=weight),
                           "M2", m1_trials[m1_index][0]) for weight in LAMBDAS]
        m2_index = best_trial(trial_rows["m2_lambda"])
        baseline_lr, weight = LEARNING_RATES[baseline_index], LAMBDAS[m2_index]
        choices = {"baseline_lr": baseline_lr, "m2_lr": m2_lr, "evidence_lambda": weight}
        selected = [(42, baseline_trials[baseline_index], m2_trials[m2_index])]
        for seed in SEEDS[1:]:
            selected.append((seed,
                             job(f"seed-{seed}/baseline", replace(config, seed=seed, learning_rate=baseline_lr), "BASELINE"),
                             job(f"seed-{seed}/m2", replace(m2_config, seed=seed, evidence_lambda=weight), "M2")))

    write_json(output / "selected_hyperparameters.json", {"policy": plan["hyperparameter_policy"], **choices})
    checkpoints = {}
    for seed, baseline, m2 in selected:
        for mode, (directory, result) in (("BASELINE", baseline), ("M2", m2)):
            checkpoint = Path(result["checkpoint"])
            checksum = sha256(checkpoint)
            if checksum != result["selection"]["checkpoint_sha256"]:
                raise ValueError(f"Selected checkpoint changed: {checkpoint}")
            checkpoints[f"{seed}/{mode}"] = {
                "job": str(directory.relative_to(output)), "checkpoint": str(checkpoint.relative_to(output)),
                "sha256": checksum, "bytes": checkpoint.stat().st_size, "selection": result["selection"],
                "training_cost": result["training_cost"],
            }
    write_json(output / "selected_checkpoints.json", checkpoints)
    (output / "checkpoints.sha256").write_text(
        "".join(f"{item['sha256']}  {item['checkpoint']}\n" for item in checkpoints.values()),
        encoding="utf-8")

    rows = training_rows(config)[1] if smoke else load_split(config.data_path, "test")
    if not smoke and len(rows) != EXPECTED_COUNTS["test"]:
        raise ValueError("Test must include all official claim IDs")
    split = "dev" if smoke else "test"
    for key, item in checkpoints.items():
        seed, mode = key.split("/")
        prefix = output / "evaluation" / f"seed-{seed}" / f"{mode.lower()}_{split}"
        marker = prefix.with_name(prefix.name + "_complete.json")
        parts = [prefix.with_name(prefix.name + suffix)
                 for suffix in ("_metrics.json", "_predictions.jsonl")]
        if marker.is_file():
            record = read_json(marker)
            if record != {"checkpoint_sha256": item["sha256"],
                          "files": {p.name: sha256(p) for p in parts}}:
                raise ValueError(f"Completed evaluation artifact changed: {prefix}")
            continue
        progress({"event": "final_evaluation", "seed": int(seed), "system": mode, "split": split})
        system = system_from_checkpoint(output / item["checkpoint"], config.device)
        if smoke:
            previous = read_jsonl(output / item["job"] / f"{mode.lower()}_dev_predictions.jsonl")[0]
            row = next(row for row in rows if row["id"] == previous["id"])
            prediction = system.prediction(row)
            if any(abs(prob - previous["probabilities"][label]) > 1e-6
                   for label, prob in prediction["probabilities"].items()):
                raise AssertionError("Reload changed development probabilities")
            from .cli import predict_record
            raw = predict_record(system, row["claim"], row["context"], Path(args.cache) / "vncorenlp")
            if (raw["ranking"] != prediction["ranking"] or
                    any(abs(prob - raw["probabilities"][label]) > 1e-6
                        for label, prob in prediction["probabilities"].items())):
                raise AssertionError("Raw-input prediction differs from prepared input")
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        evaluate(system, rows, prefix,
                 progress=lambda event: progress({**event, "seed": int(seed), "system": mode, "split": split}))
        write_json(marker, {"checkpoint_sha256": item["sha256"], "files": {p.name: sha256(p) for p in parts}})
        del system
        release_gpu()
    write_report(output, plan, rows, checkpoints, split)
    write_json(output / "completion.json", {"status": "passed" if smoke else "complete",
                                          "completed_utc": now(), "split": split,
                                          "evaluation_count": len(checkpoints),
                                          "report_sha256": sha256(output / "comparison.json")})
    progress({"status": "passed" if smoke else "complete", "event": "complete"})
    return export_locked(output)


def best_trial(rows):
    return max(range(len(rows)), key=lambda i: rows[i]["dev_macro_f1"])
