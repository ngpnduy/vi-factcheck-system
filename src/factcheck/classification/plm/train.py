"""PLM training script for Vietnamese fact-checking.

Hyperparameters match the original ViFactCheck paper (plm_training.py):
    lr          = 5e-6   (0.5e-5)
    epochs      = 10
    batch_size  = 16
    optimizer   = AdamW
    loss        = CrossEntropyLoss
    dropout     = 0.3  (in model.py)
    max_length  = 256

Usage::

    uv run python -m factcheck.classification.plm.train \\
        --model-name vinai/phobert-base \\
        --input-mode gold_evidence \\
        --output-dir outputs/phobert_base_gold

    # Full context experiment:
    uv run python -m factcheck.classification.plm.train \\
        --model-name vinai/phobert-large \\
        --input-mode full_context \\
        --output-dir outputs/phobert_large_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from factcheck.classification.plm.dataset import IDX2LABEL, FactCheckDataset
from factcheck.classification.plm.model import PLMClassifier
from factcheck.evaluation.metrics import compute_metrics

# --- Default hyperparameters from paper ---
DEFAULT_LR = 5e-6        # 0.5e-5
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LEN = 256
DATA_DIR = Path("data/processed/vifactcheck")


def train_one_epoch(
    model: PLMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)  # [B, 3]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_split(
    model: PLMClassifier,
    loader: DataLoader,
    dataset: FactCheckDataset,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds: list[str] = []
    all_true: list[str] = []
    for batch in tqdm(loader, desc="  eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        logits = model(input_ids, attention_mask)
        predicted = logits.argmax(dim=1).cpu()
        all_preds.extend([IDX2LABEL[i.item()] for i in predicted])
        all_true.extend([IDX2LABEL[i.item()] for i in labels])
    return compute_metrics(all_true, all_preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PLM classifier for ViFactCheck")
    parser.add_argument("--model-name", default="vinai/phobert-base",
                        help="HuggingFace model name (default: vinai/phobert-base)")
    parser.add_argument("--input-mode", default="gold_evidence",
                        choices=["gold_evidence", "full_context"],
                        help="Input mode: gold_evidence or full_context")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plm"),
                        help="Directory to save best model checkpoint and logs")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model : {args.model_name}")
    print(f"Mode  : {args.input_mode}")

    # --- Data ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = FactCheckDataset(
        args.data_dir / "train.jsonl", tokenizer, args.max_length, args.input_mode,
    )
    dev_ds = FactCheckDataset(
        args.data_dir / "validation.jsonl", tokenizer, args.max_length, args.input_mode,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)

    # --- Model ---
    model = PLMClassifier.from_pretrained(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training loop ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev_metrics = evaluate_split(model, dev_loader, dev_ds, device)

        print(f"  train_loss={train_loss:.4f}  dev_macro_f1={dev_metrics['macro_f1']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, **dev_metrics})

        # Save best checkpoint by dev Macro-F1
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
            print(f"  ✓ New best model saved (dev F1={best_f1:.4f})")

    # --- Save training history ---
    (args.output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nTraining complete. Best dev Macro-F1: {best_f1:.4f}")
    print(f"Checkpoint saved to: {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
