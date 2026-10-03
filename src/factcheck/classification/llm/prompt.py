"""Alpaca prompt template and instruction text for LLM fact-checking.

All strings are copied EXACTLY from the original ViFactCheck repo:
    - ALPACA_PROMPT : from llm_training.py / llm_full_context.py (Cell 51)
    - INSTRUCTION_TEXT : from llm_multi-evidence.py (transform_data function)
    - build_input_text : from llm_multi-evidence.py (transform_data function, input column)

The original instruction text from llm_multi-evidence.py line 82:
    "You will be presented with a long context, followed by a statement.
     Your task is to fact-check the statement based on the provided context.
     You must categorize the statement into one of three categories:
     Supported(0): ..., Refuted(1): ..., Not Enough Information(2): ...
     Only output the code of the label, do not generate irrelevant content."
"""

from __future__ import annotations

from factcheck.data.schemas import DatasetRecord

# Alpaca prompt template — copied exactly from llm_training.py
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
)

# Instruction text — copied exactly from llm_multi-evidence.py (line 82)
INSTRUCTION_TEXT = (
    "You will be presented with a long context, followed by a statement. "
    "Your task is to fact-check the statement based on the provided context. "
    "You must categorize the statement into one of three categories: "
    "Supported(0): Choose this if the statement is true and fully supported by the context. "
    "Refuted(1): Choose this if the statement is false and contradicted by the context. "
    "Not Enough Information(2): Choose this if the statement contains content that is not "
    "covered by the context, making it impossible to determine its accuracy. "
    "Only output the code of the label, do not generate irrelevant content."
)


def build_input_text(record: DatasetRecord, input_mode: str) -> str:
    """Build the ###Input field for the Alpaca prompt.

    Modes:
        gold_evidence      : ###Context: <gold evidence> ###Sentence: <claim>
        full_context       : ###Context: <full article context> ###Sentence: <claim>
        retrieved_evidence : ###Context: <top-5 retrieved sentences> ###Sentence: <claim>

    The format "###Context: ... ###Sentence: ..." is taken from
    llm_multi-evidence.py line 80:
        f"###Context: {'. '.join(row['evidence_top5'])} ###Sentence: {row['Statement']}"
    """
    if input_mode == "gold_evidence":
        context_text = ". ".join(e.text for e in record.evidence) if record.evidence else ""
    elif input_mode == "full_context":
        context_text = record.context or ""
    elif input_mode == "retrieved_evidence":
        # Retrieved evidence is stored in record.evidence after retrieval step
        context_text = ". ".join(e.text for e in record.evidence) if record.evidence else ""
    else:
        raise ValueError(f"Unknown input_mode: {input_mode!r}")

    return f"###Context: {context_text} ###Sentence: {record.claim}"


def formatting_prompts_func(examples: dict, tokenizer) -> dict:
    """Format examples into Alpaca prompt strings.

    Ported exactly from llm_training.py (formatting_prompts_func):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN

    Args:
        examples: dict with keys "instructions", "input", "output" (batched)
        tokenizer: the loaded tokenizer (needed for EOS token)
    """
    instructions = examples["instructions"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, inp, output in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(instruction, inp, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}


def build_inference_prompt(record: DatasetRecord, input_mode: str) -> str:
    """Build the inference-time Alpaca prompt (Response left empty for generation).

    At inference time, we fill Instruction and Input but leave Response blank
    so the model generates the label.
    """
    input_text = build_input_text(record, input_mode)
    return ALPACA_PROMPT.format(INSTRUCTION_TEXT, input_text, "")
