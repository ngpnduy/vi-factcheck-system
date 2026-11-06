# Implementation of the 0410 progress slides

**Protocol v2 supersedes the warmup-best initialization and legacy checkpoint reuse commands below.** New runs branch from the final warmup epoch, select best across all verifier epochs, and reject v1 initialization. Use the [current guide](../LISTWISE_V2.md). The slide implementation notes below describe the original v1 protocol.

Run commands from `vi-factcheck-system/rush/`. This updates the standalone Rush
research implementation. The main `src/factcheck/` package has a separate protocol.

Source: the supplied twelve-page presentation, especially pages 7, 8, 10, and 11.
Original PDF SHA-256:
`53753c668d9bf69728b9d0c14e4fb71073650b5e6acc821af8a65c11bf0e8d87`.

## Configuration contract

| Mode | Selection | Pooling | Reranker continuation updates |
|---|---|---|---|
| M0 | BM25 top five | Uniform probability mean | None |
| M1 | BM25 top twenty, cross-encoder top five | Softmax-weighted probability mean | Frozen |
| A1 | Exactly M1's frozen selector | Uniform probability mean | Frozen |
| M2 | Same architecture as M1 | Softmax-weighted probability mean | Verification loss + balanced evidence BCE |
| A2 | Same architecture as M1 | Detached softmax verification weights | Balanced evidence BCE only |

A1 trains its own verifier for the same five-epoch budget as M0, from identical
initial verifier parameters. Its reranker is the same pretrained selector as M1,
so selected sentence IDs match. It is a trained pooling ablation.

M1, M2, and A2 share the best three-epoch weighted warmup checkpoint, including
optimizer and random-number state, then continue for two epochs. A2 detaches only
the selected scores used by verification; evidence BCE remains connected to all
eligible candidates. NEI gets verification loss only and cannot update A2's
reranker. REFUTED gold remains positive evidence supervision.

The four slide contrasts are A1-M0, M1-A1, M2-M1, and M2-A2. Five configurations
and three seeds produce fifteen final test evaluations. Validation-only tuning
uses seed 42, M1 for learning rate, and M2 for lambda. A2 inherits M2's lambda;
all lambda trials share the same selected-LR warmup checkpoint.

## Optional list-aware attention

`--reranker-architecture pairwise` retains the original architecture by default.
`listwise` contextualizes claim-sentence vectors together before scoring them.
PhoBERT still processes one pair at a time. Attention sees one article's actual
candidate count, at most twenty, without dummy sentences or cross-claim mixing.

The slides do not specify layer settings. Recorded implementation defaults are:
one pre-normalized Transformer encoder layer, PhoBERT hidden size 768, eight
heads, feedforward size 1536, GELU, and dropout 0.1. There are no candidate
positional embeddings, so candidate reordering permutes scores equivalently.
Original sentence IDs resolve ranking ties. The layer is part of the reranker:
frozen in M1/A1 and trainable in M2/A2, with its weights saved in checkpoints.

Override defaults with `--listwise-layers`, `--listwise-heads`,
`--listwise-feedforward`, and `--listwise-dropout`. Architecture/settings must
match when resuming or reusing initialization. A pairwise checkpoint cannot
initialize the added list-aware parameters.

This is inspired by the slides, not a reproduction of complete published
M-ReRank or RAV systems. BM25 stays fixed; there is no dense retriever for KL
distillation. The verifier still pools per-sentence probabilities. Multi-hop or
REFUTED gains require new experiments. References:
[M-ReRank](https://aclanthology.org/2024.findings-emnlp.428/),
[RAV](https://aclanthology.org/2024.findings-acl.551/),
[PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).

## Setup and validation

Use the [setup guide](../README.md). Preparation, dataset labels, conservative
alignment, token limits 256/96, and evidence limits 20/5 remain the same.

```powershell
# Balanced 24-train / 12-dev pipeline validation, all five modes:
.\.venv\Scripts\python.exe -m vifactcheck smoke `
  --models M0 M1 A1 M2 A2 --output runs\smoke-five-pairwise

# Optional candidate interaction:
.\.venv\Scripts\python.exe -m vifactcheck smoke `
  --models M0 M1 A1 M2 A2 --reranker-architecture listwise `
  --output runs\smoke-five-listwise

# Smaller real-model gradient/memory checks; no full benchmark training:
.\.venv\Scripts\python.exe scripts\validate_slides.py `
  --data data\prepared --model-path .cache\phobert --output runs\slides-validation
```

The diagnostic makes one optimizer update on one training claim per case. It
keeps the full token/evidence limits. Its measurements do not establish memory
use for every training claim. [Validation evidence](../SLIDES_VALIDATION.md)
records checks performed for this implementation, not new benchmark scores.

## Reuse original pairwise initialization for A1/A2

`--initialization-from` accepts a matching directory containing
`reranker_best.pt`, `warmup_best.pt`, `experiment.json`, and `summary.json`.
Data IDs, checkpoint phases, configuration, and hashes are checked. Parent files
are read without alteration; external paths/hashes are saved for resume.

For the completed laptop run, resolve its physical location using the original
`LOCATION.json`. Seed 42's parent is the selected LR trial under `tuning/`.
Seed 13/2026 parents are `seeds/seed13` and `seeds/seed2026`. The collected
seed-42 directory does not contain the two initialization checkpoints.

```powershell
.\.venv\Scripts\python.exe -u -m vifactcheck train --models A1 A2 `
  --seed 42 --lr 0.00002 --evidence-lambda 0.5 `
  --data <prepared-data-directory> --model-path <phobert-directory> `
  --initialization-from <selected-LR-trial-directory> `
  --output runs\pairwise-ablations-seed42

.\.venv\Scripts\python.exe -m vifactcheck train `
  --resume runs\pairwise-ablations-seed42\a2_last.pt
```

Repeat with each seed's own parent. Use the original validation-selected LR and
lambda. Finalize all ablation training before test evaluation. Original scores
remain measurements of the pairwise three-mode study, not list-aware results.

## A new complete five-configuration study

For a separate full-data, **single-seed** study without repeating the search:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_single_seed.py `
  --seed 42 --reranker-architecture listwise `
  --data <prepared-data-directory> --model-path <phobert-directory> `
  --segmenter-path <vncorenlp-directory> `
  --output <checkpoint-directory> --status-dir runs\single-seed42-listwise
```

This trains M0/M1/A1/M2/A2 on all official training claims, uses all validation
claims for checkpoint selection, then evaluates all five on the complete test
split and verifies raw-input prediction. Learning rate 2e-5 and lambda 0.5 are
fixed from the earlier pairwise study; they are not newly tuned for list-aware
attention. The full 3/3/2 and five-epoch budgets remain. Eight long training
claims first check memory with both M2 encoders trainable. No model changes or
automatic CPU fallback occur on memory failure.

Rerun the identical command to resume. An incomplete epoch restarts from its
last completed epoch. Source/settings/data changes are rejected; completed
training and test artifacts are verified by hash. Completed child jobs retain
their best/parent checkpoints and release superseded last states to bound disk
usage. Status, final CSV/JSON/Markdown reports and the output location are
mirrored into the status directory. Create `STOP_REQUESTED` there to stop at
the next progress boundary; remove it before explicitly resuming.

The report gives four paired article-bootstrap contrasts and labels the study
as one seed; it does not report variation across training seeds. It is separate
from both the earlier results and the three-seed protocol below.

```powershell
.\.venv\Scripts\python.exe -u -m vifactcheck full --include-ablations `
  --output runs\full-five-pairwise --status-dir runs\full-five-pairwise-status

powershell -ExecutionPolicy Bypass -File scripts\start_full.ps1 `
  -IncludeAblations -RerankerArchitecture listwise `
  -OutputDirectory "$env:LOCALAPPDATA\ViFactCheck\full-five-listwise" `
  -StatusDirectory runs\full-five-listwise-status
```

The full runner shares initialization and reuses seed-42 tuning winners. It
writes fifteen prediction sets, per-label scores, confusion matrices, retrieval
denominators, memory/runtime costs, three-seed means/sample SD, and the four
paired comparisons with article-cluster bootstrap intervals. Test never selects
hyperparameters/checkpoints.

`evaluate` and `predict` accept A1/A2 checkpoints through their saved mode and
architecture. `compare --runs ...` summarizes validation Macro-F1, includes the
new contrasts, and rejects mixed architectures; it does not report test scores.

Use a new output directory for changed source/settings. Completed experiment
snapshots, checkpoints, and reports remain in their original locations. This
code update does not automatically launch a new full experiment.
