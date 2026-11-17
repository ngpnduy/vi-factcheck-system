"""Isolated, dev-only extended-epoch diagnostic for the release/v2 PhoBERT system."""
import argparse
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import torch

from .config import DATASET_ID, DATASET_REVISION, EXPECTED_COUNTS, MODEL_ID, MODEL_REVISION, TrainConfig
from .data import has_evidence_supervision, stratified_subset
from .io import read_json, sha256, write_json, write_jsonl
from .model import VerificationSystem, evidence_loss
from .training import (copy_checkpoint, phase, restore_rng, rng_state,
                       seed_everything, training_rows)

DIAGNOSTIC_ID = "phobert-convergence-loss-v1"


def loss_summary(system, rows, stage):
    """Stable eval-mode NLL/BCE; objective averages BCE over all claims, too."""
    if not rows:
        raise ValueError("Loss evaluation needs nonempty rows")
    system.eval()
    verification_sum, evidence_sum, eligible = 0.0, 0.0, 0
    considered = 0
    with torch.no_grad():
        for row in rows:
            if stage == "reranker":
                if not has_evidence_supervision(row):
                    continue
                ids, scores = system.rank_scores(row, gradients=False)
                rank_loss = evidence_loss(scores, row, ids)
            else:
                result = system(row, training_stage=stage)
                nll = -result["log_probabilities"][row["label"]]
                if not torch.isfinite(nll):
                    raise FloatingPointError("Nonfinite diagnostic NLL")
                verification_sum += float(nll)
                rank_loss = result["evidence_loss"]
            considered += 1
            if rank_loss is not None:
                if not torch.isfinite(rank_loss):
                    raise FloatingPointError("Nonfinite diagnostic BCE")
                evidence_sum += float(rank_loss)
                eligible += 1
    contribution = system.config.evidence_lambda * evidence_sum / considered if considered and stage != "reranker" else None
    return {"verification_nll": verification_sum / considered if considered and stage != "reranker" else None,
            "eligible_evidence_bce": evidence_sum / eligible if eligible else None,
            "evidence_weighted_contribution": contribution,
            "objective_mean": ((verification_sum + system.config.evidence_lambda * evidence_sum) / considered
                               if stage != "reranker" and considered else evidence_sum / eligible if eligible else None),
            "claim_count": considered, "eligible_evidence_count": eligible,
            "mode": "eval_dropout_disabled", "stage": stage}


def make_observer(probe):
    def observe(system, train, dev, stage, record):
        # Monitoring must not change dropout RNG or the subsequent optimizer path.
        state = rng_state()
        modes = [(module, module.training) for module in system.modules()]
        started = time.perf_counter()
        try:
            result = {"dev": loss_summary(system, dev, stage),
                      "train_probe": loss_summary(system, probe, stage),
                      "train_probe_ids": [row["id"] for row in probe]}
            result["seconds"] = time.perf_counter() - started
            return result
        finally:
            restore_rng(state)
            for module, was_training in modes:
                module.training = was_training
    return observe


def monitor_logs(output):
    records = {}
    for name in ("reranker", "warmup", "m2"):
        file = output / f"{name}_history.json"
        if file.exists():
            for row in read_json(file):
                records[(row["stage"], row["global_epoch"])] = row
    rows = sorted(records.values(), key=lambda row: (row["stage"] != "reranker", row["global_epoch"]))
    write_jsonl(output / "diagnostics.jsonl", rows)
    fields = ("stage", "global_epoch", "train_objective", "train_verification_nll", "train_eligible_bce",
              "train_evidence_count", "train_probe_nll", "dev_nll", "dev_eligible_bce", "dev_evidence_count", "dev_macro_f1", "dev_mrr")
    temporary = output / "diagnostics.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            loss, diagnostics = row.get("loss_components", {}), row.get("diagnostics", {})
            dev = diagnostics.get("dev", {})
            writer.writerow({"stage": row["stage"], "global_epoch": row["global_epoch"],
                             "train_objective": row["average_loss"], "train_verification_nll": loss.get("verification_nll"),
                             "train_eligible_bce": loss.get("eligible_evidence_bce"), "train_evidence_count": loss.get("eligible_evidence_count"),
                             "train_probe_nll": diagnostics.get("train_probe", {}).get("verification_nll"),
                             "dev_nll": dev.get("verification_nll"), "dev_eligible_bce": dev.get("eligible_evidence_bce"),
                             "dev_evidence_count": dev.get("eligible_evidence_count"),
                             "dev_macro_f1": row["validation"].get("classification", {}).get("macro_f1"),
                             "dev_mrr": row["validation"]["retrieval"].get("mrr")})
    temporary.replace(output / "diagnostics.csv")
    return rows


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", default="data/prepared")
    value.add_argument("--model-path", default=".cache/phobert")
    value.add_argument("--output", required=True)
    value.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    value.add_argument("--total-epochs", type=int, default=15, help="Verifier epochs, excludes reranker pretraining")
    value.add_argument("--warmup-epochs", type=int, default=3)
    value.add_argument("--reranker-epochs", type=int, default=3)
    value.add_argument("--lr-horizon-epochs", type=int, default=5, help="Keep the original five-epoch LR ramp when observing longer training")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--lr", type=float, default=1e-5)
    value.add_argument("--evidence-lambda", type=float, default=0.1)
    value.add_argument("--check-only", action="store_true")
    value.add_argument("--smoke", action="store_true", help="24 train/12 dev; diagnostic pipeline check only")
    return value


def preflight(args):
    if args.total_epochs <= args.warmup_epochs or args.lr_horizon_epochs < 1:
        raise ValueError("Total epochs must exceed warmup; LR horizon must be positive")
    config = TrainConfig(seed=args.seed, learning_rate=args.lr, evidence_lambda=args.evidence_lambda,
                         warmup_epochs=args.warmup_epochs, continuation_epochs=args.total_epochs - args.warmup_epochs,
                         reranker_epochs=args.reranker_epochs, model_path=str(Path(args.model_path).resolve()),
                         data_path=str(Path(args.data).resolve()), device=args.device, smoke=args.smoke,
                         reranker_architecture="listwise")
    output = Path(args.output).resolve()
    # Only current prepared train/dev are loaded. The test split is never read.
    train, dev = training_rows(config)
    data_manifest = read_json(Path(config.data_path) / "manifest.json")
    if (data_manifest.get("assets", {}).get("dataset_id") != DATASET_ID or
            data_manifest.get("assets", {}).get("dataset_revision") != DATASET_REVISION):
        raise ValueError("Expected the pinned official dataset revision")
    expected = {"train": 24, "dev": 12} if config.smoke else EXPECTED_COUNTS
    if len(train) != expected["train"] or len(dev) != expected["dev"]:
        raise ValueError("Unexpected train/dev claim counts for this diagnostic")
    for rows in (train, dev):
        if len({r["id"] for r in rows}) != len(rows) or {r["label"] for r in rows} != {0, 1, 2}:
            raise ValueError("Each split requires unique IDs and all three labels")
    if not any(has_evidence_supervision(row) for row in train):
        raise ValueError("Training needs eligible positive/negative evidence supervision")
    model_files = {}
    for name in ("config.json", "vocab.txt", "bpe.codes", "pytorch_model.bin"):
        file = Path(config.model_path) / name
        if not file.is_file():
            raise FileNotFoundError(f"Prepare the pinned PhoBERT assets first: {file}")
        model_files[name] = sha256(file)
    model_config = read_json(Path(config.model_path) / "config.json")
    if model_config.get("model_type") != "roberta" or model_config.get("vocab_size") != 64001:
        raise ValueError("This convergence pilot supports the pinned PhoBERT-base assets only")
    assets_file = Path(config.model_path).parent / "assets.json"
    assets = read_json(assets_file)
    expected_files = {entry["path"]: entry["sha256"] for entry in assets["files"]}
    if assets.get("model_id") != MODEL_ID or assets.get("model_revision") != MODEL_REVISION:
        raise ValueError("Expected the recorded PhoBERT-base revision")
    if any(model_files[name] != expected_files.get("phobert/" + name) for name in model_files):
        raise ValueError("PhoBERT file hash differs from its asset manifest")
    hardware = {"device": config.device, "gpu": None}
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check the P100-compatible PyTorch build")
        hardware.update(gpu=torch.cuda.get_device_name(0), memory_mib=torch.cuda.get_device_properties(0).total_memory / 1024**2,
                        capability=list(torch.cuda.get_device_capability(0)), architectures=torch.cuda.get_arch_list())
        if hardware["capability"] == [6, 0] and "sm_60" not in hardware["architectures"]:
            raise RuntimeError("P100 needs a PyTorch build containing sm_60")
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < 35 * 1024**3:
        raise RuntimeError(f"Need at least 35 GiB free before this diagnostic; available {free / 1024**3:.2f} GiB")
    counts = Counter(row["label"] for row in train)
    per_label = min(32, min(counts.values()))
    probe = stratified_subset(train, per_label, config.seed)
    root = Path(__file__).resolve().parents[2]
    sources = list((root / "src/vifactcheck").glob("*.py")) + [root / "scripts/run_convergence.py"]
    source_hashes = {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in sources}
    identity = {"diagnostic_id": DIAGNOSTIC_ID, "purpose": "exploratory dev-only convergence; not the published best-of-five benchmark",
                "config": config.to_dict(), "output": str(output), "lr_horizon_epochs": args.lr_horizon_epochs,
                "train_ids": [row["id"] for row in train], "dev_ids": [row["id"] for row in dev],
                "train_probe_ids": [row["id"] for row in probe], "data_manifest_sha256": sha256(Path(config.data_path) / "manifest.json"),
                "data_files_sha256": {name: sha256(Path(config.data_path) / f"{name}.jsonl") for name in ("train", "dev")},
                "model_files_sha256": model_files, "source_sha256": source_hashes,
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "rank-bm25")}}
    if (output / "convergence_plan.json").exists():
        if read_json(output / "convergence_plan.json") != identity:
            raise ValueError("Output belongs to different data/source/settings/environment; use a new directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Refusing a nonempty output without this diagnostic's plan")
    return config, train, dev, probe, identity, hardware


def run(args):
    config, train, dev, probe, identity, hardware = preflight(args)
    if args.check_only:
        return {"status": "ready", "writes": False, "train_count": len(train), "dev_count": len(dev),
                "total_verifier_epochs": args.total_epochs, "lr_horizon_epochs": args.lr_horizon_epochs, "hardware": hardware}
    from .pilot import profile_long_claims, require_space
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "convergence_plan.json", identity)
    args._output_accepted = True
    write_json(output / "hardware.json", hardware)
    root = Path(__file__).resolve().parents[2]
    for name in identity["source_sha256"]:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    def progress(event):
        require_space(output, int(event.get("checkpoint_model_bytes", 0) * 1.15))
        write_json(output / "status.json", {"diagnostic_id": DIAGNOSTIC_ID, "status": "running", **event})
        if event.get("event") == "epoch_complete":
            monitor_logs(output)
        if (output / "STOP").exists():
            raise InterruptedError("STOP requested; restart the identical command to replay an incomplete epoch")
    if config.device == "cuda" and not (output / "resource_check.json").exists():
        write_json(output / "resource_check.json", profile_long_claims(config, train, progress))
    seed_everything(config.seed)
    observer = make_observer(probe)
    options = dict(progress=progress, epoch_observer=observer, diagnostic_id=DIAGNOSTIC_ID,
                   lr_horizon_epochs=args.lr_horizon_epochs)
    # M2 and M1 construct the verifier first; use the original M1 warmup semantics.
    system = VerificationSystem(config, "M1")
    ranker = phase(system, train, dev, output, "reranker", "reranker", config.reranker_epochs, **options)
    warmup = phase(system, train, dev, output, "warmup", "warmup", config.warmup_epochs, source=ranker, **options)
    warmup_end = output / "warmup_end.pt"
    if warmup_end.exists():
        if sha256(warmup_end) != sha256(output / "warmup_last.pt"):
            raise ValueError("Final warmup source changed")
    else:
        copy_checkpoint(output / "warmup_last.pt", warmup_end)
    del system
    from .pilot import release_gpu
    release_gpu()
    seed_everything(config.seed)
    system = VerificationSystem(config, "M2")
    best = phase(system, train, dev, output, "m2", "continuation", config.continuation_epochs,
                 source=warmup_end, selection_source=warmup, **options)
    monitor_logs(output)
    result = {"status": "complete", "diagnostic_id": DIAGNOSTIC_ID, "purpose": identity["purpose"],
              "best_checkpoint": str(best), "best_sha256": sha256(best), "last_checkpoint": str(output / "m2_last.pt"),
              "total_verifier_epochs": args.total_epochs, "test_evaluated": False,
              "interpretation": "Best dev checkpoint within this horizon; not proof of convergence or a global optimum"}
    write_json(output / "completion.json", result)
    write_json(output / "status.json", result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        # Preflight failures must not create an output directory.
        output = Path(args.output)
        if getattr(args, "_output_accepted", False):
            failure = {"status": "stopped" if isinstance(error, InterruptedError) else "failed",
                       "error": type(error).__name__, "message": str(error)}
            if isinstance(error, torch.cuda.OutOfMemoryError):
                failure["memory"] = {"allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                                     "reserved_mib": torch.cuda.memory_reserved() / 1024**2}
            write_json(output / "failure.json", failure)
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
