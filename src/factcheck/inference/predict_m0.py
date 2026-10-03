"""Predict M0 verdicts from gold-free inputs and a saved best checkpoint."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from torch.utils.data import DataLoader

from factcheck.data.vifactcheck import _sha256
from factcheck.data.word_segmentation import VnCoreNLPSegmenter
from factcheck.m0_config import M0Config
from factcheck.training.m0_dataset import M0Collator, M0Dataset
from factcheck.training.m0_runtime import (
    choose_device,
    load_components,
    predict_batches,
    save_predictions,
    seed_everything,
    write_json,
)


def predict(checkpoint: Path, inputs: Path, output: Path, *, device: str = "auto",
            batch_size: int | None = None, word_segmenter_dir: Path | None = None) -> dict:
    if output.exists() or output.with_suffix(".metadata.json").exists():
        raise FileExistsError(f"Prediction output already exists: {output}")
    run = json.loads((checkpoint.parent / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "complete":
        raise ValueError("Training run is not complete")
    config = M0Config(**run["config"])
    if word_segmenter_dir is not None:
        config = replace(config, word_segmenter_dir=str(word_segmenter_dir))
    seed_everything(run["seed"], config.cpu_threads)
    target_device = choose_device(device)
    size = config.batch_size if batch_size is None else batch_size
    if size < 1:
        raise ValueError("batch_size must be positive")
    # Neither this function nor this Dataset opens a target/gold file.
    dataset = M0Dataset(inputs, top_k=config.top_k)
    model, tokenizer = load_components(config, checkpoint)
    model.to(target_device)
    with VnCoreNLPSegmenter(Path(config.word_segmenter_dir)) as segment:
        if segment.manifest != run["word_segmenter"]:
            raise ValueError("Word-segmenter resources differ from the training run")
        collator = M0Collator(tokenizer, segment, max_length=config.max_length,
                              claim_max_tokens=config.claim_max_tokens)
        loader = DataLoader(dataset, batch_size=size, collate_fn=collator, num_workers=0)
        predictions, cost = predict_batches(
            model, loader, target_device, seed=run["seed"], model_id=str(checkpoint),
        )
    save_predictions(output, predictions)
    report = {"checkpoint": str(checkpoint), "seed": run["seed"], "smoke": run["smoke"],
              "protocol_status": config.protocol_status, "device": str(target_device),
              "input_sha256": _sha256(inputs), "prediction_sha256": _sha256(output),
              "cost": cost,
              "checkpoint_sha256": {path.name: _sha256(path)
                                    for path in sorted(checkpoint.iterdir()) if path.is_file()}}
    write_json(output.with_suffix(".metadata.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--word-segmenter-dir", type=Path)
    print(json.dumps(predict(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
