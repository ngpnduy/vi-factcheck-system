"""Train M0 with one NLL per claim and select checkpoints only on validation."""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from factcheck.data.vifactcheck import _sha256
from factcheck.data.word_segmentation import VnCoreNLPSegmenter
from factcheck.evaluation.evaluate_m0 import (
    retrieval_metrics,
    summarize_seed_metrics,
    verdict_metrics,
)
from factcheck.m0_config import M0Config
from factcheck.training.m0_dataset import M0Collator, M0Dataset
from factcheck.training.m0_runtime import (
    choose_device,
    load_components,
    model_batch,
    predict_batches,
    read_rows,
    runtime_metadata,
    save_predictions,
    seed_everything,
    synchronize,
    verify_prepared,
    write_json,
)


def train(config: M0Config, seed: int, *, smoke: bool = False) -> dict:
    """Run a fresh experiment. Existing output directories are never overwritten."""
    seed_everything(seed, config.cpu_threads)
    device = choose_device(config.device)
    data_dir = Path(config.data_dir)
    manifest = verify_prepared(data_dir, ("train", "validation"))
    run_dir = Path(config.output_dir) / (f"smoke_seed_{seed}" if smoke else f"seed_{seed}")
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}. Choose another --output-dir")
    train_inputs = read_rows(data_dir / "train.inputs.jsonl")
    train_targets = read_rows(data_dir / "train.targets.jsonl")
    val_inputs = read_rows(data_dir / "validation.inputs.jsonl")
    val_targets = read_rows(data_dir / "validation.targets.jsonl")
    if smoke:
        # A seeded subset is a technical check, never a validation research score.
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(len(train_inputs), generator=generator)[:16].tolist()
        train_inputs = [train_inputs[i] for i in selected]
        train_ids = {row["id"] for row in train_inputs}
        train_targets = [row for row in train_targets if row["id"] in train_ids]
        val_inputs = val_inputs[:12]
        val_ids = {row["id"] for row in val_inputs}
        val_targets = [row for row in val_targets if row["id"] in val_ids]
    train_data = M0Dataset(train_inputs, train_targets, top_k=config.top_k)
    val_data = M0Dataset(val_inputs, top_k=config.top_k)
    total_epochs = 1 if smoke else config.warmup_epochs + config.continuation_epochs
    batch_size = 1 if smoke else config.batch_size
    accumulation = 1 if smoke else config.gradient_accumulation_steps
    model, tokenizer = load_components(config)
    model.to(device)
    # Do not create a misleading run directory until assets/model loaded successfully.
    run_dir.mkdir(parents=True)
    record = {"config": config.to_dict(), "seed": seed, "smoke": smoke,
              "device": str(device), "status": "running", "runtime": runtime_metadata(),
              "prepared_manifest": manifest, "train_claims": len(train_data),
              "validation_claims": len(val_data), "epochs": total_epochs,
              "batch_size": batch_size, "gradient_accumulation_steps": accumulation,
              "scheduler": "linear warmup/decay over all epochs; no phase reset",
              "checkpoint_selection": "validation macro_f1, strict improvement; earlier tie",
              "source_sha256": {str(path.relative_to(Path('src'))): _sha256(path)
                                 for path in sorted(Path('src/factcheck').rglob('*'))
                                 if path.suffix in {'.py', '.java'}}}
    write_json(run_dir / "run.json", record)
    try:
        with VnCoreNLPSegmenter(Path(config.word_segmenter_dir)) as segment:
            record["word_segmenter"] = segment.manifest
            write_json(run_dir / "run.json", record)
            collator = M0Collator(tokenizer, segment, max_length=config.max_length,
                                  claim_max_tokens=config.claim_max_tokens)
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True, collate_fn=collator,
                generator=torch.Generator().manual_seed(seed), num_workers=0,
            )
            val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collator,
                                    num_workers=0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                         weight_decay=config.weight_decay)
            steps = math.ceil(len(train_loader) / accumulation) * total_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(steps * config.warmup_ratio),
                num_training_steps=steps,
            )
            history, best_f1, best_epoch = [], -1.0, None
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for epoch in range(1, total_epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum, claim_count, group_claims = 0.0, 0, 0
                train_truncation = {"claims": 0, "evidence": 0, "pairs": 0}
                epoch_started = time.perf_counter()
                for step, batch in enumerate(train_loader, start=1):
                    result = model(**model_batch(batch, device, labels=True))
                    count = batch["num_claims"]
                    loss = result["loss"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    # Sum per-claim gradients, then divide by real claims in this group.
                    # This also handles final partial batches/accumulation windows correctly.
                    (loss * count).backward()
                    loss_sum += float(loss.detach()) * count
                    claim_count += count
                    group_claims += count
                    for key in train_truncation:
                        train_truncation[key] += batch["truncation"][key]
                    if step % accumulation == 0 or step == len(train_loader):
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(group_claims)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        group_claims = 0
                    if step == 1 or step % 50 == 0 or step == len(train_loader):
                        print(f"seed={seed} epoch={epoch}/{total_epochs} "
                              f"batch={step}/{len(train_loader)} loss={loss_sum/claim_count:.5f}",
                              flush=True)
                synchronize(device)
                train_seconds = time.perf_counter() - epoch_started
                predictions, timing = predict_batches(
                    model, val_loader, device, seed=seed, model_id=f"m0/seed_{seed}/epoch_{epoch}",
                )
                metrics = verdict_metrics(val_targets, predictions)
                entry = {"epoch": epoch, "phase": "initial" if epoch <= config.warmup_epochs
                         else "continuation", "train_loss": loss_sum / claim_count,
                         "train_seconds": train_seconds, "train_truncation": train_truncation,
                         "validation": metrics, "validation_cost": timing}
                history.append(entry)
                print(f"validation macro_f1={metrics['macro_f1']:.6f}", flush=True)
                if metrics["macro_f1"] > best_f1:
                    best_f1, best_epoch = metrics["macro_f1"], epoch
                    model.save_pretrained(run_dir / "best", safe_serialization=True)
                    tokenizer.save_pretrained(run_dir / "best")
                    save_predictions(run_dir / "validation.predictions.jsonl", predictions)
                    write_json(run_dir / "validation.metrics.json", {
                        "verdict": metrics, "retrieval": retrieval_metrics(val_inputs, val_targets),
                        "cost": timing, "seed": seed, "smoke": smoke,
                    })
                write_json(run_dir / "history.json", history)
            record.update(status="complete", best_epoch=best_epoch, best_macro_f1=best_f1,
                          wall_seconds=time.perf_counter() - started,
                          peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()
                          if device.type == "cuda" else None)
            write_json(run_dir / "run.json", record)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(run_dir / "run.json", record)
        raise
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/m0.yaml"))
    parser.add_argument("--seed", type=int, help="One seed; otherwise run all configured seeds")
    parser.add_argument("--smoke", action="store_true", help="16 train / 12 validation, 1 epoch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = M0Config.load(args.config)
    overrides = {key: str(getattr(args, key)) for key in ("device", "data_dir", "output_dir")
                 if getattr(args, key) is not None}
    config = replace(config, **overrides)
    seeds = [args.seed] if args.seed is not None else ([42] if args.smoke else config.seeds)
    reports = []
    for seed in seeds:
        record = train(config, seed, smoke=args.smoke)
        reports.append({"seed": seed, "macro_f1": record["best_macro_f1"]})
    if len(reports) > 1:
        # Keep the aggregate explicit; individual prediction files remain authoritative.
        metrics = []
        import json
        for seed in seeds:
            path = Path(config.output_dir) / f"seed_{seed}" / "validation.metrics.json"
            metrics.append(json.loads(path.read_text())["verdict"])
        write_json(Path(config.output_dir) / "validation.summary.json",
                   summarize_seed_metrics(metrics))


if __name__ == "__main__":
    main()
