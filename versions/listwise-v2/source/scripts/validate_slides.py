"""Real-PhoBERT control checks; one train claim per case, not benchmark training."""
import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vifactcheck.config import TrainConfig
from vifactcheck.data import has_evidence_supervision, load_split
from vifactcheck.io import read_jsonl, sha256, write_json
from vifactcheck.model import PairTokenizer, VerificationSystem
from vifactcheck.pilot import backbone_probes, release_gpu
from vifactcheck.training import (fingerprint, isolated_joint_gradient,
                                  isolated_stop_gradient, make_optimizer,
                                  seed_everything, system_from_checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/prepared")
    parser.add_argument("--model-path", default=".cache/phobert")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--legacy-checkpoint", type=Path,
                        help="Optional trusted original pairwise M1 checkpoint with adjacent dev predictions")
    parser.add_argument("--output", type=Path, default=Path("runs/slides-validation"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic architecture or device fallback")
    report = {"purpose": "pipeline/control validation only; not trained-ablation benchmark scores",
              "started_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    args.output.mkdir(parents=True, exist_ok=True)
    train, dev = load_split(args.data, "train"), load_split(args.data, "dev")
    tokenizer = PairTokenizer(str(Path(args.model_path).resolve()))
    longest = sorted((r for r in train if has_evidence_supervision(r)),
                     key=lambda r: sum(len(r["sentences"][i]["segmented"]) for i in r["candidate_ids"]),
                     reverse=True)[:8]
    if not longest:
        raise ValueError("Real validation requires eligible evidence supervision")
    row = max(longest, key=lambda r: sum(tokenizer.encode(r["claim_segmented"],
              r["sentences"][i]["segmented"])[1]["pair_tokens"] for i in r["candidate_ids"]))
    if args.legacy_checkpoint:
        print("Checking compatibility with the original pairwise M1 checkpoint", flush=True)
        reference = read_jsonl(args.legacy_checkpoint.with_name("m1_dev_predictions.jsonl"))[0]
        sample = next(r for r in dev if r["id"] == reference["id"])
        system = system_from_checkpoint(args.legacy_checkpoint, args.device)
        predicted = system.prediction(sample)
        difference = max(abs(predicted["probabilities"][label] - reference["probabilities"][label])
                         for label in predicted["probabilities"])
        if difference > 1e-6 or predicted["ranking"] != reference["ranking"]:
            raise AssertionError("Original checkpoint prediction changed")
        report["legacy_reload"] = {"checkpoint": str(args.legacy_checkpoint.resolve()),
                                    "checkpoint_sha256": sha256(args.legacy_checkpoint),
                                    "max_probability_difference": difference, "same_ranking": True}
        del system
        release_gpu()
    cases = [(architecture, mode) for architecture in ("pairwise", "listwise") for mode in ("A1", "A2")]
    cases.append(("listwise", "M2"))
    try:
        for architecture, mode in cases:
            print(f"Validating {architecture} {mode}", flush=True)
            started = time.perf_counter()
            seed_everything(42)
            config = TrainConfig(device=args.device, model_path=str(Path(args.model_path).resolve()),
                                 data_path=str(Path(args.data).resolve()), reranker_architecture=architecture)
            system = VerificationSystem(config, mode, tokenizer)
            system.eval()
            comparison = None
            if mode == "A1":
                system.mode = "M1"
                weighted = system.prediction(row)
                system.mode = "A1"
                uniform = system.prediction(row)
                if weighted["ranking"] != uniform["ranking"]:
                    raise AssertionError("A1 changed M1 sentence selection")
                n = len(uniform["selected_sentences"])
                if any(abs(sentence["weight"] - 1 / n) > 1e-6 for sentence in uniform["selected_sentences"]):
                    raise AssertionError("A1 weights are not uniform")
                comparison = {"same_m1_selection": True, "uniform_weights": True}
            stage = "a1" if mode == "A1" else "continuation"
            checks = isolated_stop_gradient(system, row) if mode == "A2" else (
                isolated_joint_gradient(system, row) if mode == "M2" else {})
            system.configure_training(stage)
            before = backbone_probes(system)
            before_context = fingerprint(system.reranker.contextualizer) if architecture == "listwise" else None
            frozen_hash = fingerprint(system.reranker) if mode == "A1" else None
            optimizer = make_optimizer(system, config)
            # A diagnostic optimizer update, not the eight-claim training budget.
            # Lower initial loss scaling avoids spending trial steps calibrating AMP.
            scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda", init_scale=128)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            output = system(row, training_stage=stage)
            loss = -output["log_probabilities"][row["label"]]
            if output["evidence_loss"] is not None:
                loss = loss + config.evidence_lambda * output["evidence_loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not all(torch.isfinite(p.grad).all().item() for p in system.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite diagnostic gradients")
            scaler.step(optimizer)
            scaler.update()
            system.zero_grad(set_to_none=True)
            del output, loss, optimizer, scaler
            after = backbone_probes(system)
            changed = {key: before[key] != after[key] for key in before}
            for key, value in changed.items():
                expected = mode != "A1" or key.startswith("verifier.")
                if value != expected:
                    raise AssertionError(f"Unexpected encoder update: {mode}: {key}")
            if mode == "A1" and fingerprint(system.reranker) != frozen_hash:
                raise AssertionError("A1 changed its frozen reranker")
            context_changed = None
            if architecture == "listwise":
                context_changed = before_context != fingerprint(system.reranker.contextualizer)
                if context_changed != (mode != "A1"):
                    raise AssertionError("Unexpected list-attention update")
            inference_row = {key: value for key, value in dev[0].items() if key not in ("label", "alignment")}
            prediction = system.prediction(inference_row)
            if abs(sum(prediction["probabilities"].values()) - 1) > 1e-5:
                raise AssertionError("Invalid prediction probabilities")
            case = {"architecture": architecture, "mode": mode, "config": config.to_dict(), "checks": checks,
                    "selection_control": comparison, "encoder_probe_changes": changed,
                    "list_attention_changed": context_changed, "training_claim_id": row["id"],
                    "candidate_count": len(row["candidate_ids"]), "selected_count": config.evidence_count,
                    "optimizer_updates": 1, "seconds": time.perf_counter() - started,
                    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
                    "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if args.device == "cuda" else None,
                    "prediction_without_gold": prediction}
            report["cases"].append(case)
            write_json(args.output / "validation.json", report)
            print(f"Passed {architecture} {mode}; peak allocated {case['peak_cuda_allocated_mib']} MiB", flush=True)
            del system
            gc.collect()
            release_gpu()
        report.update(status="passed", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "validation.json", report)
        print("All real-model control checks passed", flush=True)
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__, message=str(error), architecture_changed=False)
        if isinstance(error, torch.cuda.OutOfMemoryError):
            report["gpu_memory"] = {"allocated_mib": torch.cuda.memory_allocated() / 1024**2,
                                    "reserved_mib": torch.cuda.memory_reserved() / 1024**2}
        write_json(args.output / "validation.json", report)
        raise


if __name__ == "__main__":
    main()
