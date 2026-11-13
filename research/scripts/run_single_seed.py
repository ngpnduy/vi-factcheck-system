"""Run all five slide configurations on full data with one fixed training seed.

Rerun the same command to resume completed epochs and sealed child jobs.
Hyperparameters are fixed before training; no search or test-based selection.
"""
import argparse
import csv
import io
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from vifactcheck.cli import predict_record
from vifactcheck.config import ALL_MODELS, DATASET_REVISION, EXPECTED_COUNTS, MODEL_REVISION, TrainConfig, comparison_pairs
from vifactcheck.data import has_evidence_supervision, load_split
from vifactcheck.full import (collect_seed42, exclusive_run, now, training_job,
                             verify_completed_training, verify_initialization)
from vifactcheck.io import read_json, read_jsonl, sha256, write_json
from vifactcheck.pilot import keep_awake, profile_long_claims, release_gpu, require_space
from vifactcheck.reporting import paired_article_bootstrap
from vifactcheck.training import evaluate, system_from_checkpoint


def write_report(output, config, models_directory):
    summary = verify_completed_training(models_directory)
    rows, predictions, metrics = [], {}, {}
    for mode in ALL_MODELS:
        name = mode.lower()
        metric = read_json(models_directory / f"{name}_test_metrics.json")
        prediction = read_jsonl(models_directory / f"{name}_test_predictions.jsonl")
        if len(prediction) != EXPECTED_COUNTS["test"] or metric["classification"]["count"] != EXPECTED_COUNTS["test"]:
            raise ValueError("Report requires complete official test predictions")
        predictions[mode], metrics[mode] = prediction, metric
        rows.append({"model": mode, "seed": config.seed,
                     "dev_macro_f1": summary[mode]["metrics"]["classification"]["macro_f1"],
                     "test_macro_f1": metric["classification"]["macro_f1"],
                     "accuracy": metric["classification"]["accuracy"],
                     **metric["retrieval"], **metric["cost"]})
    paired = {f"{a}-{b}": paired_article_bootstrap(predictions[a], predictions[b])
              for a, b in comparison_pairs(ALL_MODELS)}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (output / "results.csv").write_text(buffer.getvalue(), encoding="utf-8-sig")
    report = {"study": "full-data fixed-hyperparameter single-seed study", "config": config.to_dict(),
              "models": rows, "test_metrics": metrics, "paired_article_bootstrap": paired,
              "training_history_files": [str(p) for p in sorted((output / "training").glob("*/*_history.json"))],
              "limitation": "No new hyperparameter search or between-training-seed variance estimate. Bootstrap intervals are conditional on the fitted models."}
    write_json(output / "comparison.json", report)
    write_json(output / "paired_bootstrap.json", paired)
    lines = ["# Single-seed slide experiment", "",
             f"Seed {config.seed}; architecture {config.reranker_architecture}; all 5,062 train / 723 dev / 1,447 test claims.", "",
             f"Fixed learning rate {config.learning_rate}, evidence lambda {config.evidence_lambda}. Values transferred from the earlier pairwise study, without a new search for this architecture. Checkpoints selected on validation only.", "",
             "Reranker: 3 epochs. Shared weighted warmup: 3 epochs. M1/M2/A2: 2 continuation epochs each. M0/A1: 5 verifier epochs each. Separate PhoBERT parameters, 256/96 token limits, 20/5 sentence limits, accumulation 8, AMP, activation checkpointing and Adafactor.", "",
             "| Model | Dev Macro-F1 | Test Macro-F1 | Accuracy |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['dev_macro_f1']:.4f} | {row['test_macro_f1']:.4f} | {row['accuracy']:.4f} |")
    lines += ["", "Per-label scores, confusion matrices, retrieval denominators, alignment coverage, truncation and inference/memory costs are in comparison.json and models/*_test_metrics.json. Epoch histories record training runtime and peak GPU memory; shared phases ran once.", "",
              "| Contrast | Macro-F1 difference | Conditional 95% interval |", "|---|---:|---|"]
    for contrast, record in paired.items():
        value = record["metrics"]["macro_f1"]
        lines.append(f"| {contrast} | {value['difference']:+.4f} | [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}] |")
    lines += ["", report["limitation"], "", "This is a separate study from the original three-seed pairwise results. Official splits share source articles. The candidate-attention layer is inspired by the slides; it is not a complete reproduction of M-ReRank/RAV.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run(args):
    output, mirror = Path(args.output).resolve(), Path(args.status_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(seed=args.seed, model_path=str(Path(args.model_path).resolve()),
                         data_path=str(Path(args.data).resolve()), reranker_architecture=args.reranker_architecture)
    manifest = read_json(Path(config.data_path) / "manifest.json")
    sources = list((Path(__file__).resolve().parents[1] / "vifactcheck").glob("*.py")) + [Path(__file__).resolve()]
    plan = {"format_version": 1, "study": "full-data fixed-hyperparameter single-seed study",
            "models": list(ALL_MODELS), "config": config.to_dict(), "data_manifest_sha256": sha256(Path(config.data_path) / "manifest.json"),
            "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p) for p in sources},
            "segmenter_path": str(Path(args.segmenter_path).resolve()), "torch_num_threads": 2,
            "hyperparameters": "2e-5 and 0.5 transferred from prior pairwise validation tuning; no new search",
            "resume": "completed child jobs are hash-verified; incomplete epochs restart from last complete epoch"}
    plan_path = output / "single_seed_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Source, data or settings changed; use a new experiment directory")
    write_json(plan_path, plan)
    write_json(mirror / plan_path.name, plan)
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(Path(__file__).resolve().parents[1])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if sha256(target) != sha256(source):
            raise ValueError("Source snapshot changed")
    write_json(mirror / "LOCATION.json", {"checkpoint_and_result_directory": str(output), "status_directory": str(mirror)})
    status = read_json(output / "status.json") if (output / "status.json").exists() else {}
    status.update(status="running", pid=os.getpid(), started_utc=status.get("started_utc", now()),
                  output=str(output), seed=config.seed, models=list(ALL_MODELS), architecture=config.reranker_architecture,
                  train_count=5062, validation_count=723, test_count=1447, completed_test_models=0, total_test_models=5)

    def progress(event):
        if (mirror / "STOP_REQUESTED").exists() or (output / "STOP_REQUESTED").exists():
            raise InterruptedError("Stop requested; incomplete epoch will restart on resume")
        free = require_space(output, int(event.get("checkpoint_model_bytes", 0) * 1.15))
        status.update(event)
        status.update(updated_utc=now(), free_disk_gib=free / 1024**3)
        write_json(output / "status.json", status)
        if mirror != output:
            write_json(mirror / "status.json", status)

    try:
        with keep_awake():
            progress({"task": "preflight", "phase": "checking", "event": "checking"})
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required; no automatic architecture or CPU fallback")
            if {s: manifest["splits"][s]["valid_count"] for s in EXPECTED_COUNTS} != EXPECTED_COUNTS:
                raise ValueError("Single-seed study requires complete official splits")
            if manifest["assets"]["dataset_revision"] != DATASET_REVISION or manifest["assets"]["model_revision"] != MODEL_REVISION:
                raise ValueError("Unexpected dataset/model revision")
            for split, checksum in manifest["prepared_file_sha256"].items():
                if sha256(Path(config.data_path) / f"{split}.jsonl") != checksum:
                    raise ValueError(f"Prepared split changed: {split}")
            model_checksum = next(a["sha256"] for a in manifest["assets"]["files"] if a["path"] == "phobert/pytorch_model.bin")
            if sha256(Path(config.model_path) / "pytorch_model.bin") != model_checksum:
                raise ValueError("PhoBERT model weights changed")
            train, dev = load_split(config.data_path, "train"), load_split(config.data_path, "dev")
            if len(train) != 5062 or len(dev) != 723:
                raise ValueError("Prepared split count mismatch")
            status["evidence_supervised_train_count"] = sum(has_evidence_supervision(r) for r in train)
            resource = output / "resource_check.json"
            if not resource.exists():
                write_json(resource, profile_long_claims(config, train, progress))
            # Seal and clean each child before the next to bound disk usage.
            parent = output / "training" / "weighted-parent"
            ablations = output / "training" / "joint-ablations"
            baseline = output / "training" / "baseline"
            for task, directory, modes, initialization in (
                    ("weighted-parent", parent, ("M1",), None),
                    ("joint-ablations", ablations, ("A1", "M2", "A2"), parent),
                    ("baseline", baseline, ("M0",), None)):
                for key in ("epoch", "total_epochs", "completed_claims", "total_claims", "epoch_training_seconds"):
                    status.pop(key, None)
                progress({"task": task, "phase": "starting", "event": "checking"})
                training_job(config, directory, modes, progress, train, initialization_from=initialization)
            model_sources = {"M0": baseline, "M1": parent, "A1": ablations, "M2": ablations, "A2": ablations}
            models = output / "models"
            summary = collect_seed42(models, model_sources, config)
            verify_initialization(summary)
            # Read test records only once all five training branches are final.
            test = load_split(config.data_path, "test")
            for mode in ALL_MODELS:
                name = mode.lower()
                checkpoint = models / f"{name}_best.pt"
                marker = models / f"{name}_test_complete.json"
                checksum = sha256(checkpoint)
                progress({"task": "final-evaluation", "model": mode, "phase": f"{name}_test", "event": "loading", "completed_claims": 0, "total_claims": len(test)})
                if marker.exists():
                    record = read_json(marker)
                    if record["checkpoint_sha256"] != checksum or any(sha256(models / n) != h for n, h in record["artifact_sha256"].items()):
                        raise ValueError("Completed test artifacts changed")
                else:
                    system = system_from_checkpoint(checkpoint, "cuda")
                    torch.cuda.reset_peak_memory_stats()
                    metrics = evaluate(system, test, models / f"{name}_test", progress=progress)
                    reference = read_jsonl(models / f"{name}_dev_predictions.jsonl")[0]
                    row = next(r for r in dev if r["id"] == reference["id"])
                    prepared = system.prediction(row)
                    raw = predict_record(system, row["claim"], row["context"], args.segmenter_path)
                    difference = max(abs(prepared["probabilities"][k] - reference["probabilities"][k]) for k in prepared["probabilities"])
                    raw_difference = max(abs(prepared["probabilities"][k] - raw["probabilities"][k]) for k in prepared["probabilities"])
                    if max(difference, raw_difference) > 1e-6:
                        raise AssertionError("Reload/raw-input prediction differs")
                    write_json(models / f"{name}_example_prediction.json", raw)
                    files = [f"{name}_test_metrics.json", f"{name}_test_predictions.jsonl", f"{name}_example_prediction.json"]
                    record = {"checkpoint_sha256": checksum, "test_count": metrics["classification"]["count"],
                              "reload_max_probability_difference": difference, "raw_input_max_probability_difference": raw_difference,
                              "artifact_sha256": {n: sha256(models / n) for n in files}}
                    write_json(marker, record)
                    del system
                    release_gpu()
                if record["test_count"] != EXPECTED_COUNTS["test"]:
                    raise ValueError("Test count mismatch")
                status["completed_test_models"] += 1
                progress({"event": "test_model_complete"})
            progress({"task": "report", "phase": "report", "event": "analysis"})
            write_report(output, config, models)
            complete = {"status": "complete", "seed": config.seed, "models": list(ALL_MODELS),
                        "started_utc": status["started_utc"], "completed_utc": now(),
                        "report_checksums": {n: sha256(output / n) for n in ("REPORT.md", "results.csv", "comparison.json", "paired_bootstrap.json")}}
            write_json(output / "completion.json", complete)
            if mirror != output:
                for name in ("REPORT.md", "results.csv", "comparison.json", "paired_bootstrap.json", "completion.json", "resource_check.json"):
                    shutil.copy2(output / name, mirror / name)
            progress({"status": "complete", "phase": "complete", "event": "complete"})
            return complete
    except BaseException as error:
        failure = {"status": "stopped" if isinstance(error, InterruptedError) else "failed", "exception": type(error).__name__, "message": str(error), "updated_utc": now()}
        if isinstance(error, torch.cuda.OutOfMemoryError):
            failure["gpu_memory"] = {"allocated_mib": torch.cuda.memory_allocated() / 1024**2, "reserved_mib": torch.cuda.memory_reserved() / 1024**2}
        status.update(failure)
        write_json(output / "failure.json", failure)
        write_json(output / "status.json", status)
        write_json(mirror / "status.json", status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--segmenter-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reranker-architecture", choices=("pairwise", "listwise"), default="listwise")
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    with exclusive_run(arguments.output):
        run(arguments)
