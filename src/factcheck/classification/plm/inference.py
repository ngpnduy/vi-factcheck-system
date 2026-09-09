"""PLM inference: generate predictions on the test set.

Loads the best checkpoint saved by train.py and runs inference on the test split.
Writes a predictions JSONL file consumable by evaluator.py.

Usage::

    uv run python -m factcheck.classification.plm.inference \\
        --model-name vinai/phobert-base \\
        --checkpoint outputs/phobert_base_gold/best_model.pt \\
        --input-mode gold_evidence \\
        --output outputs/phobert_base_gold/test_predictions.jsonl

Then evaluate::

    uv run python -m factcheck.evaluation.evaluator \\
        --predictions outputs/phobert_base_gold/test_predictions.jsonl \\
        --gold data/processed/vifactcheck/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from factcheck.classification.plm.dataset import FactCheckDataset, IDX2LABEL
from factcheck.classification.plm.model import PLMClassifier

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LEN = 256
DATA_DIR = Path("data/processed/vifactcheck")


@torch.no_grad()
def run_inference(
    model: PLMClassifier,
    loader: DataLoader,
    ids: list[str],
    device: torch.device,
) -> list[dict]:
    """Run model inference and return list of {id, prediction} dicts."""
    model.eval()
    results = []
    idx = 0
    for batch in tqdm(loader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)       # [B, 3]
        predicted = logits.argmax(dim=1).cpu().tolist()
        for pred in predicted:
            results.append({"id": ids[idx], "prediction": IDX2LABEL[pred]})
            idx += 1
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PLM inference on the test split")
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best_model.pt saved by train.py")
    parser.add_argument("--input-mode", default="gold_evidence",
                        choices=["gold_evidence", "full_context"])
    parser.add_argument("--split", default="test",
                        choices=["test", "validation"],
                        help="Which split to run inference on")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for predictions JSONL")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Model     : {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")

    # Map split name to filename (validation.jsonl not dev.jsonl)
    split_file = "validation.jsonl" if args.split == "validation" else "test.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = FactCheckDataset(
        args.data_dir / split_file,
        tokenizer,
        args.max_length,
        args.input_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    # Load checkpoint
    model = PLMClassifier.from_pretrained(args.model_name).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = run_inference(model, loader, dataset.ids, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{len(results)} predictions written to: {args.output}")


if __name__ == "__main__":
    main()
