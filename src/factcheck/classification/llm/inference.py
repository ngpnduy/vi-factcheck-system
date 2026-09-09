"""LLM inference for ViFactCheck.

Loads the LoRA adapter saved by train_unsloth.py and runs inference on the test set.
Outputs a predictions.jsonl file consumable by evaluator.py.

⚠️ The original ViFactCheck repo has NO inference code for LLM.
This script is written based on:
    1. The Alpaca prompt format used in training (llm_training.py / llm_multi-evidence.py)
    2. The output format: model was trained to output "0", "1", or "2"
    3. We parse the first digit (0/1/2) from the model's generated response.

Usage::

    uv run python -m factcheck.classification.llm.inference \\
        --base-model unsloth/gemma-7b-bnb-4bit \\
        --adapter-dir outputs/gemma7b_gold/lora_adapter \\
        --input-mode gold_evidence \\
        --output outputs/gemma7b_gold/test_predictions.jsonl

Then evaluate::

    bash scripts/eval_plm.sh \\
        --predictions outputs/gemma7b_gold/test_predictions.jsonl \\
        --gold data/processed/vifactcheck/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from factcheck.classification.llm.prompt import build_inference_prompt
from factcheck.data.schemas import DatasetRecord, VerdictLabel

# Label mapping: "0"→SUPPORTED, "1"→REFUTED, "2"→NOT_ENOUGH_INFO
# (matches training output from llm_multi-evidence.py: table["output"] = table["labels"].astype(str))
INT_TO_LABEL = {
    "0": VerdictLabel.SUPPORTED,
    "1": VerdictLabel.REFUTED,
    "2": VerdictLabel.NOT_ENOUGH_INFO,
}

DATA_DIR = Path("data/processed/vifactcheck")


def _load_records(path: Path) -> list[DatasetRecord]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(DatasetRecord.model_validate_json(line))
    return records


def _parse_label(generated_text: str) -> str:
    """Extract label from LLM output.

    The model was trained to output only "0", "1", or "2".
    We find the first occurrence of 0, 1, or 2 in the response.
    Fallback to NOT_ENOUGH_INFO (2) if no valid label found.
    """
    match = re.search(r"[012]", generated_text)
    if match:
        return INT_TO_LABEL.get(match.group(), VerdictLabel.NOT_ENOUGH_INFO)
    return VerdictLabel.NOT_ENOUGH_INFO  # safe fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM inference for ViFactCheck")
    parser.add_argument("--base-model", default="unsloth/gemma-7b-bnb-4bit",
                        help="Base model name (same as used during training)")
    parser.add_argument("--adapter-dir", type=Path, required=True,
                        help="Path to LoRA adapter dir saved by train_unsloth.py")
    parser.add_argument("--input-mode", default="gold_evidence",
                        choices=["gold_evidence", "full_context", "retrieved_evidence"])
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for predictions JSONL")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=16,
                        help="Max tokens to generate (label is 1 char, 16 is generous)")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError:
        raise ImportError("unsloth is required. Install with: uv add unsloth")

    print(f"Base model  : {args.base_model}")
    print(f"Adapter     : {args.adapter_dir}")
    print(f"Input mode  : {args.input_mode}")

    # --- Load base model + LoRA adapter ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_dir),  # Load directly from saved adapter dir
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        token=args.hf_token,
    )
    FastLanguageModel.for_inference(model)  # Enable optimized inference

    # --- Load data ---
    split_file = "validation.jsonl" if args.split == "validation" else "test.jsonl"
    records = _load_records(args.data_dir / split_file)
    print(f"Loaded {len(records)} records from {split_file}")

    # --- Run inference ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for record in tqdm(records, desc="LLM Inference"):
        prompt = build_inference_prompt(record, args.input_mode)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        label = _parse_label(generated)
        results.append({"id": record.id, "prediction": label, "raw_output": generated})

    # --- Write predictions ---
    with args.output.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{len(results)} predictions written to: {args.output}")


if __name__ == "__main__":
    main()
