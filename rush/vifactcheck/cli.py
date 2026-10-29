import argparse
import json
import os
from pathlib import Path
import sys
import time

from .config import ALL_MODELS, STANDARD_MODELS, TrainConfig, comparison_pairs, reranker_options
from .io import read_json, write_json


def add_training_arguments(parser):
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--model-path", default=".cache/phobert")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--evidence-lambda", type=float, default=0.5)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--train-per-label", type=int, help="Stratified training subset; no evidence-alignment filtering")
    add_reranker_arguments(parser)


def add_reranker_arguments(parser):
    parser.add_argument("--reranker-architecture", choices=("pairwise", "listwise"), default="pairwise")
    parser.add_argument("--listwise-layers", type=int, default=1)
    parser.add_argument("--listwise-heads", type=int, default=8)
    parser.add_argument("--listwise-feedforward", type=int, default=1536)
    parser.add_argument("--listwise-dropout", type=float, default=0.1)


def configuration(args, smoke=False):
    return TrainConfig(seed=args.seed, learning_rate=args.lr, evidence_lambda=args.evidence_lambda, accumulation_steps=args.accumulation, device=args.device, model_path=args.model_path, data_path=args.data, reranker_epochs=1 if smoke else args.reranker_epochs, warmup_epochs=1 if smoke else args.warmup_epochs, continuation_epochs=1 if smoke else args.continuation_epochs, smoke=smoke, train_per_label=args.train_per_label, **reranker_options(args))


def predict_record(system, claim, context, segmenter_path=".cache/vncorenlp"):
    from .data import bm25_candidates
    from .text import VnCoreSegmenter, normalize, recover_sentences
    if not isinstance(claim, str) or not normalize(claim) or not isinstance(context, str) or not normalize(context):
        raise ValueError("claim and context must be nonempty strings")
    segmented = VnCoreSegmenter(segmenter_path).segment_many([claim, context])
    sentences = recover_sentences(context, segmented[1])
    claim_segmented = " ".join(segmented[0])
    candidates, scores = bm25_candidates(claim_segmented, sentences, system.config.candidate_count)
    row = {"id": "user-input", "claim": claim, "claim_segmented": claim_segmented, "context": context, "sentences": sentences, "candidate_ids": candidates, "bm25_scores": scores, "label": 2, "alignment": {"status": "missing", "gold_sentence_ids": []}}
    return system.prediction(row)


def smoke_run(config, output, models=STANDARD_MODELS):
    from .data import load_split
    from .training import run_experiment, system_from_checkpoint
    import torch
    started = time.perf_counter()
    results = run_experiment(config, output, models)
    # Check loaded checkpoints using actual full models, and raw-input prediction
    # using a real validation article; no gold enters this prediction path.
    rows = {r["id"]: r for r in load_split(config.data_path, "dev")}
    reload_checks = {}
    for mode, result in results.items():
        reference = read_json(Path(output) / f"{mode.lower()}_dev_metrics.json")
        from .io import read_jsonl
        predictions = read_jsonl(Path(output) / f"{mode.lower()}_dev_predictions.jsonl")
        row = rows[predictions[0]["id"]]
        system = system_from_checkpoint(result["checkpoint"], config.device)
        loaded = system.prediction(row)
        difference = max(abs(loaded["probabilities"][label] - predictions[0]["probabilities"][label]) for label in loaded["probabilities"])
        if difference > 1e-6:
            raise AssertionError("Reloaded checkpoint changed prediction")
        predicted = predict_record(system, row["claim"], row["context"])
        raw_difference = max(abs(loaded["probabilities"][label] - predicted["probabilities"][label]) for label in loaded["probabilities"])
        if raw_difference > 1e-6:
            raise AssertionError("Raw-input prediction differs from prepared article")
        write_json(Path(output) / f"{mode.lower()}_example_prediction.json", predicted)
        reload_checks[mode] = {"reload_max_probability_difference": difference, "raw_input_max_probability_difference": raw_difference, "validation_macro_f1": reference["classification"]["macro_f1"]}
        del system
        import gc
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()
    report = {"status": "passed", "purpose": "pipeline validation only; not full-data research results", "training_count": 24, "validation_count": 12, "seed": config.seed, "total_seconds": time.perf_counter() - started, "models": results, "reload_and_prediction_checks": reload_checks}
    write_json(Path(output) / "smoke_report.json", report)
    return report


def tune(args):
    from .training import run_experiment, training_rows
    from .full import lambda_job
    output = Path(args.output)
    base = configuration(args)
    base.seed = 42
    trials = []
    for lr in (1e-5, 2e-5):
        cfg = TrainConfig(**{**base.to_dict(), "learning_rate": lr})
        result = run_experiment(cfg, output / f"lr_{lr}", ("M1",))
        trials.append({"learning_rate": lr, "validation_macro_f1": result["M1"]["metrics"]["classification"]["macro_f1"]})
    chosen_lr = max(trials, key=lambda trial: trial["validation_macro_f1"])["learning_rate"]
    parent_directory = output / f"lr_{chosen_lr}"
    train, dev = training_rows(base)
    lambda_trials = []
    for weight in (0.1, 0.5, 1.0):
        cfg = TrainConfig(**{**base.to_dict(), "learning_rate": chosen_lr, "evidence_lambda": weight})
        result = lambda_job(cfg, output / f"lambda_{weight}", parent_directory, lambda event: None, train, dev)
        lambda_trials.append({"evidence_lambda": weight, "validation_macro_f1": result["M2"]["metrics"]["classification"]["macro_f1"]})
    choice = {"seed": 42, "split": "dev", "learning_rate": chosen_lr, "evidence_lambda": max(lambda_trials, key=lambda trial: trial["validation_macro_f1"])["evidence_lambda"], "learning_rate_trials": trials, "lambda_trials": lambda_trials}
    write_json(output / "selected_hyperparameters.json", choice)
    return choice


def compare(args):
    import numpy as np
    records = {}
    architecture = None
    for directory in args.runs:
        summary = read_json(Path(directory) / "summary.json")
        experiment = read_json(Path(directory) / "experiment.json")
        current_architecture = TrainConfig(**experiment["config"]).to_dict()
        current_architecture = {key: current_architecture[key] for key in reranker_options(argparse.Namespace())}
        if architecture is not None and current_architecture != architecture:
            raise ValueError("Do not combine pairwise/list-aware architectures or attention settings in one comparison")
        architecture = current_architecture
        if experiment["config"]["smoke"]:
            raise ValueError("Smoke runs cannot be aggregated as research experiments")
        seed = experiment["config"]["seed"]
        for mode, result in summary.items():
            if seed in records.setdefault(mode, {}):
                raise ValueError(f"Duplicate {mode} seed {seed}")
            records[mode][seed] = result["metrics"]["classification"]["macro_f1"]
    report = {"models": {mode: {"per_seed_macro_f1": values, "mean": float(np.mean(list(values.values()))), "sample_std": float(np.std(list(values.values()), ddof=1)) if len(values) > 1 else None} for mode, values in records.items()}, "paired_seed_differences": {}}
    for left, right in comparison_pairs(records):
        if left in records and right in records:
            seeds = sorted(set(records[left]) & set(records[right]))
            report["paired_seed_differences"][f"{left}-{right}"] = {str(seed): records[left][seed] - records[right][seed] for seed in seeds}
    write_json(args.output, report)
    return report


def main(argv=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Article-conditioned ViFactCheck M0/M1/A1/M2/A2")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Download pinned assets and preprocess all official splits")
    prep.add_argument("--output", default="data/prepared")
    prep.add_argument("--cache", default=".cache")
    prep.add_argument("--offline", action="store_true")
    pilot = sub.add_parser("pilot", help="Run the fixed 900-claim seed-42 pilot and generate its test report")
    pilot.add_argument("--output", default="runs/pilot-900-seed42")
    pilot.add_argument("--data", default="data/prepared")
    pilot.add_argument("--model-path", default=".cache/phobert")
    pilot.add_argument("--hours", type=float, default=6)
    pilot.add_argument("--deadline-utc", help="Absolute ISO UTC deadline; preserved on resume")
    status_parser = sub.add_parser("status", help="Read pilot progress without loading models")
    status_parser.add_argument("--run", default="runs/pilot-900-seed42")
    full = sub.add_parser("full", help="Resume full-data tuning, three-seed training and nine final evaluations")
    full.add_argument("--output", default="runs/full-m0-m1-m2")
    full.add_argument("--data", default="data/prepared")
    full.add_argument("--model-path", default=".cache/phobert")
    full.add_argument("--segmenter-path", default=".cache/vncorenlp")
    full.add_argument("--status-dir", help="Optional project-local directory for mirrored progress and final reports")
    full.add_argument("--include-ablations", action="store_true", help="Include A1/A2 and evaluate all 15 configuration-seed combinations")
    add_reranker_arguments(full)
    for name in ("train", "tune", "smoke"):
        child = sub.add_parser(name)
        add_training_arguments(child)
        child.add_argument("--output", default=f"runs/{name}-42")
        if name != "smoke":
            child.add_argument("--reranker-epochs", type=int, default=3)
            child.add_argument("--warmup-epochs", type=int, default=3)
            child.add_argument("--continuation-epochs", type=int, default=2)
        if name in ("train", "smoke"):
            child.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(STANDARD_MODELS))
        if name == "train":
            child.add_argument("--resume", help="Resume a locally generated *_last.pt phase checkpoint")
            child.add_argument("--initialization-from", help="Reuse reranker_best.pt and warmup_best.pt from a matching earlier experiment")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data", default="data/prepared")
    ev.add_argument("--split", choices=("dev", "test"), default="dev")
    ev.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ev.add_argument("--output", default="runs/evaluation")
    pred = sub.add_parser("predict")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--input", required=True, help="UTF-8 JSON containing claim and context")
    pred.add_argument("--segmenter-path", default=".cache/vncorenlp")
    pred.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    pred.add_argument("--output", default="runs/prediction.json")
    comp = sub.add_parser("compare")
    comp.add_argument("--runs", nargs="+", required=True)
    comp.add_argument("--output", default="runs/comparison.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .data import prepare
            result = prepare(args.output, args.cache, args.offline)
        elif args.command == "pilot":
            from .pilot import run_pilot
            if args.hours <= 0:
                raise ValueError("Pilot hours must be positive")
            result = run_pilot(args)
        elif args.command == "status":
            result = read_json(Path(args.run) / "status.json")
        elif args.command == "full":
            from .full import run_full
            result = run_full(args)
        elif args.command == "train":
            from .training import resume_checkpoint, run_experiment
            if args.resume and args.initialization_from:
                raise ValueError("Use --resume alone; its checkpoint records the initialization source")
            result = {"checkpoint": str(resume_checkpoint(args.resume))} if args.resume else run_experiment(configuration(args), args.output, args.models, initialization_from=args.initialization_from)
        elif args.command == "smoke":
            result = smoke_run(configuration(args, smoke=True), args.output, args.models)
        elif args.command == "tune":
            result = tune(args)
        elif args.command == "evaluate":
            from .data import load_split
            from .training import evaluate, system_from_checkpoint
            system = system_from_checkpoint(args.checkpoint, args.device)
            result = evaluate(system, load_split(args.data, args.split), args.output)
        elif args.command == "predict":
            from .training import system_from_checkpoint
            record = read_json(args.input)
            system = system_from_checkpoint(args.checkpoint, args.device)
            result = predict_record(system, record["claim"], record["context"], args.segmenter_path)
            write_json(args.output, result)
        else:
            result = compare(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as error:
        import torch
        failure = {"status": "failed", "exception": type(error).__name__, "message": str(error)}
        if isinstance(error, torch.cuda.OutOfMemoryError):
            failure["memory"] = {"allocated_mib": torch.cuda.memory_allocated() / 1024**2, "reserved_mib": torch.cuda.memory_reserved() / 1024**2}
            failure["message"] += " No model layers, token budgets, or evidence counts were changed. Close competing GPU applications or explicitly use CPU."
        if args.command in ("train", "tune", "smoke"):
            write_json(Path(args.output) / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
