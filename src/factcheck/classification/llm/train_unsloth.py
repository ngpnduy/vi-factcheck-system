"""LLM fine-tuning for ViFactCheck using unsloth + SFTTrainer.

Ported from the original ViFactCheck repo:
    - llm_training.py       → --input-mode gold_evidence
    - llm_full_context.py   → --input-mode full_context
    - llm_multi-evidence.py → --input-mode retrieved_evidence

Key details from original code:
    - Model: unsloth/gemma-7b-bnb-4bit (default), unsloth/mistral-7b-bnb-4bit
    - LoRA: r=16, lora_alpha=16, dropout=0, target_modules=[q/k/v/o/gate/up/down_proj]
    - Prompt: Alpaca template (Instruction / Input / Response)
    - Dataset field: "instructions", "input", "output"
    - Output label: str(label_int)  e.g. "0", "1", "2"
    - Training: epochs=5, lr=2e-4, batch=4, grad_accum=4, adamw_8bit, seed=3407
    - Checkpoint: saved via SFTTrainer output_dir (NOT in original repo — we add this)

Usage::

    # Gold Evidence (from llm_training.py)
    uv run python -m factcheck.classification.llm.train_unsloth \\
        --model-name unsloth/gemma-7b-bnb-4bit \\
        --input-mode gold_evidence \\
        --hf-token YOUR_TOKEN \\
        --output-dir outputs/gemma7b_gold

    # Full Context (from llm_full_context.py)
    uv run python -m factcheck.classification.llm.train_unsloth \\
        --model-name unsloth/gemma-7b-bnb-4bit \\
        --input-mode full_context \\
        --output-dir outputs/gemma7b_full

    # Retrieved Evidence (from llm_multi-evidence.py)
    uv run python -m factcheck.classification.llm.train_unsloth \\
        --model-name unsloth/mistral-7b-bnb-4bit \\
        --input-mode retrieved_evidence \\
        --output-dir outputs/mistral7b_retrieved
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset

from factcheck.classification.llm.prompt import (
    INSTRUCTION_TEXT,
    build_input_text,
    formatting_prompts_func,
)
from factcheck.data.schemas import DatasetRecord, VerdictLabel

# --- Hyperparameters from original repo ---
MAX_SEQ_LENGTH = 2048
DTYPE = None          # auto-detect
LOAD_IN_4BIT = True

# LoRA config from original (same across all 3 scripts)
LORA_CONFIG = dict(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# TrainingArguments from llm_training.py (gold/full) and llm_multi-evidence.py (retrieved)
TRAIN_ARGS_GOLD_FULL = dict(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # effective batch = 16
    warmup_steps=5,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
)

TRAIN_ARGS_RETRIEVED = dict(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,    # effective batch = 64 (from llm_multi-evidence.py)
    warmup_steps=5,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
)

LABEL_MAP = {
    VerdictLabel.SUPPORTED: "0",
    VerdictLabel.REFUTED: "1",
    VerdictLabel.NOT_ENOUGH_INFO: "2",
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


def _records_to_hf_dataset(
    records: list[DatasetRecord],
    input_mode: str,
) -> Dataset:
    """Convert DatasetRecords to HuggingFace Dataset with Alpaca format.

    Output columns: instructions, input, output
    (matches formatting_prompts_func expected fields)
    """
    rows = []
    for r in records:
        input_text = build_input_text(r, input_mode)
        rows.append({
            "instructions": INSTRUCTION_TEXT,
            "input": input_text,
            "output": LABEL_MAP[r.label],
        })
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LLM for ViFactCheck (unsloth)")
    parser.add_argument("--model-name", default="unsloth/gemma-7b-bnb-4bit",
                        help="Unsloth 4-bit model name on HuggingFace")
    parser.add_argument("--input-mode", default="gold_evidence",
                        choices=["gold_evidence", "full_context", "retrieved_evidence"],
                        help="Input mode: gold_evidence, full_context, or retrieved_evidence")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm"),
                        help="Directory to save LoRA adapter checkpoint")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (needed for gated models)")
    args = parser.parse_args()

    # Lazy import so the module is importable without unsloth installed
    try:
        # Unsloth must patch transformers before importing the trainer.
        # isort: off
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer
        from transformers import TrainingArguments
        # isort: on
    except ImportError:
        raise ImportError(
            "unsloth and trl are required. Install with:\n"
            "  uv add unsloth trl"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model     : {args.model_name}")
    print(f"Input mode: {args.input_mode}")
    print(f"Output dir: {args.output_dir}")

    # --- Load model & tokenizer ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=args.hf_token,
    )

    # --- Configure LoRA (exact from original) ---
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    # --- Prepare data ---
    train_records = _load_records(args.data_dir / "train.jsonl")
    train_dataset = _records_to_hf_dataset(train_records, args.input_mode)

    # Apply Alpaca prompt formatting (matches formatting_prompts_func from original)
    train_dataset = train_dataset.map(
        lambda ex: formatting_prompts_func(ex, tokenizer),
        batched=True,
    )

    # --- TrainingArguments ---
    train_arg_dict = (
        TRAIN_ARGS_RETRIEVED if args.input_mode == "retrieved_evidence"
        else TRAIN_ARGS_GOLD_FULL
    )
    training_args = TrainingArguments(
        **train_arg_dict,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        output_dir=str(args.output_dir),
    )

    # --- SFTTrainer (exact from original) ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer_stats = trainer.train()
    print("\nTraining complete!")
    print(f"  Total steps : {trainer_stats.global_step}")
    print(f"  Train loss  : {trainer_stats.training_loss:.4f}")

    # --- Save LoRA adapter ---
    # SFTTrainer already saves to output_dir at the end.
    # We also explicitly save for clarity.
    model.save_pretrained(str(args.output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "lora_adapter"))
    print(f"  LoRA adapter saved to: {args.output_dir / 'lora_adapter'}")


if __name__ == "__main__":
    main()
