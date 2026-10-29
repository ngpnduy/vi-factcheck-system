# Rush-version: M0/M1/A1/M2/A2 with BM25 and PhoBERT

This standalone subproject implements M0, M1, A1, M2, and A2 from the preserved [Proposal_v2.docx](docs/Proposal_v2.docx) and the 0410 progress slides. A claim is checked against its supplied news article. Retrieval searches sentences inside that article; the system predicts `SUPPORTED`, `REFUTED`, or `NOT_ENOUGH_INFO`. See the [slide implementation guide](docs/SLIDES_0410_IMPLEMENTATION.md) for the ablations, optional list-aware attention, and checkpoint reuse.

Run all commands below from the repository's **`rush/` directory**. This subproject has its own `pyproject.toml` and `.venv`; the repository's root `uv.lock` continues to describe the separate `src/factcheck/` workflows. `PROVENANCE.json` records the original pilot source snapshot and exact training IDs. The full pipeline runner was added afterwards; it saves its own frozen source snapshot and hashes.

See [branch notes](docs/BRANCH_NOTES.md) before comparing these results with the existing M0. The Rush implementation uses PhoBERT-base, VnCoreNLP sentence splitting, BM25Okapi, Adafactor, effective batch eight and a linear classifier head. The existing M0 has different defaults. No datasets, caches or trained checkpoints are committed.

The project implements the proposal's experimental configurations. It does not claim to reproduce the original RAV or M-ReRank algorithms or published ViFactCheck scores. A1 and A2 are implemented; D0 and human contradiction annotation remain outside this stage. The deadline pilot includes paired article-cluster bootstrap analysis conditional on its single training seed.

| Model | Selected evidence | Aggregation | Continuation training |
|---|---|---|---|
| M0 | BM25 top 5 | Uniform mean of sentence probabilities | Verifier only |
| M1 | BM25 top 20, PhoBERT reranker top 5 | Softmax-weighted mean | Verifier only; reranker frozen |
| A1 | Identical frozen selector to M1 | Uniform mean | Verifier only; same total budget as M0 |
| M2 | Same architecture as M1 | Softmax-weighted mean | Both encoders; verification and evidence losses |
| A2 | Same architecture as M1 | Softmax-weighted mean, verification weights detached | Verifier from verification loss; reranker from evidence loss |

Pairwise reranking is the default. `--reranker-architecture listwise` adds attention
between candidates, with settings saved in checkpoints. `train` and `smoke`
accept all five `--models`; `full --include-ablations` runs fifteen final models.
[Validation evidence](SLIDES_VALIDATION.md) records pipeline checks, not new
ablation benchmark scores.

## Completed pilot results

The 900-claim, seed-42 pilot completed on 4 October 2026. Test Macro-F1 was 51.01%
for M0, 52.48% for M1 and 54.84% for M2 on all 1,447 test claims. Read the
[finished report](results/pilot-900-seed42/REPORT.md) or
[download the results ZIP](results/pilot-900-seed42/README.md), which includes
per-claim predictions and the matching source snapshot. These are preliminary
single-seed results. The original three-configuration full-data experiment finished
on 20 October 2026; mean test Macro-F1 was 59.73% / 68.42% / 69.70% for M0 / M1 /
M2 across seeds 13, 42, and 2026. Those are pairwise measurements without A1/A2.
The new ablations and list-aware study have not been run as full experiments.

## Windows setup

Run commands from the project folder. Python 3.10+ and Java 8+ are required. This laptop already has Python 3.12, Java 8, and CUDA-enabled PyTorch. Its project environment reuses installed packages; it does not modify those packages.

```powershell
# Fresh, independent environment on another machine:
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# Alternatively, reuse an existing CUDA PyTorch installation:
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -ReuseInstalledPackages

# Use the environment directly; activation is optional:
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m vifactcheck prepare
.\.venv\Scripts\python.exe -m vifactcheck smoke --output runs\smoke-42
```

`requirements-tested.txt` records the exact package versions used for the completed validation. `VALIDATION.md` contains its measured results and limitations.

Preparation downloads pinned public dataset/model files and VnCoreNLP into `.cache`, writes processed data under `data/prepared`, and validates official split counts (5062/723/1447). Asset checksums and revisions are recorded. After downloads, `prepare --offline` needs no network. VnCoreNLP is run in a separate Java process and is released before training. A temporary runtime path handles its JAR's issue with spaces in Windows paths.

Smoke validation uses 24 training claims and 12 dev claims, balanced across all labels. SUPPORTED/REFUTED smoke examples must have valid evidence supervision. It uses real PhoBERT models, one reranker epoch, one verifier initialization epoch, and one continuation epoch. M0 receives two verifier epochs. Test is never used by smoke validation or tuning. Smoke scores only confirm that the pipeline works.

## Full experiments

The automated full-data, three-seed runner is available as `python -m vifactcheck full`.
See [the full pipeline guide](FULL_PIPELINE.md) for its Windows background launcher,
validation-only tuning, shared lambda initialization, resume behavior and final reports.
It keeps the full 5,062-claim training split and has no six-hour deadline.

### Deadline pilot

The report-deadline experiment keeps the full M0/M1/M2 architectures and epoch budgets but uses 900 stratified training claims (300 per label), seed 42 only, fixed LR 2e-5 and lambda 0.5, and no tuning. Selection does not filter on evidence alignment. All 723 dev and 1,447 test claims are evaluated. Results are a reduced-data single-seed pilot, not full proposal reproduction.

```powershell
.\.venv\Scripts\python.exe -u -m vifactcheck pilot --output runs\pilot-900-seed42
.\.venv\Scripts\python.exe -m vifactcheck status --run runs\pilot-900-seed42
```

The pilot profiles eight long claims before training, temporarily prevents system sleep, records progress in `status.json`, selects checkpoints on dev, evaluates test, validates raw-text prediction, and writes `REPORT.md`, `results.csv`, `comparison.json`, and paired article-cluster bootstrap intervals. Intervals are conditional on the one training seed; no between-seed standard deviation is reported. Its default deadline is six hours and is preserved on resume; `--deadline-utc` can set an absolute UTC cutoff for the first launch. OOM/space/time failures retain complete epoch checkpoints. Repeating the same command resumes the workflow. After successful completion only the three final best model checkpoints are retained; completed-parent hashes and histories remain. Existing smoke files are untouched.

The commands below launch the separate full-data, three-seed experiment.

The following commands are ready to run; full experiments are not automatically launched by setup or smoke validation.

```powershell
# Select common LR with M1; select lambda with M2, using dev and seed 42:
.\.venv\Scripts\python.exe -m vifactcheck tune --output runs\tuning

# Substitute values from runs/tuning/selected_hyperparameters.json:
.\.venv\Scripts\python.exe -m vifactcheck train --models M0 M1 M2 --seed 42 --lr 0.00002 --evidence-lambda 0.5 --output runs\seed42
.\.venv\Scripts\python.exe -m vifactcheck train --models M0 M1 M2 --seed 13 --lr 0.00002 --evidence-lambda 0.5 --output runs\seed13
.\.venv\Scripts\python.exe -m vifactcheck train --models M0 M1 M2 --seed 2026 --lr 0.00002 --evidence-lambda 0.5 --output runs\seed2026

# Run test evaluation only after the experimental settings are finalized:
.\.venv\Scripts\python.exe -m vifactcheck evaluate --checkpoint runs\seed42\m2_best.pt --split test --output runs\seed42\m2_test

# Summarize dev scores across seeds (test reports remain separate):
.\.venv\Scripts\python.exe -m vifactcheck compare --runs runs\seed13 runs\seed42 runs\seed2026

# Resume the last completed epoch of an interrupted phase:
.\.venv\Scripts\python.exe -m vifactcheck train --resume runs\seed42\m2_last.pt
```

Repeating `train` with the same output and configuration resumes all phases from their recorded checkpoints. Explicit `--resume` completes the named phase; repeat the original `train` command to continue later phases. Checkpoints save model/optimizer/scaler/random states and subset identifiers. Incomplete epochs restart from the previous complete epoch. Changes to configuration, data, or the parent checkpoint require a new output directory. Locally generated `.pt` checkpoints use Python serialization: load only trusted project checkpoints.

The reranker trains for up to three epochs; its best checkpoint is selected by dev MRR. The weighted verifier trains for three initialization epochs. M1 and M2 continue for two epochs from the same best warmup checkpoint, including its verifier optimizer state. M0 trains for five epochs from the same initial verifier weights. Checkpoint selection uses dev Macro-F1 and retains the earlier epoch on a tie. Hyperparameter trials always use seed 42 and dev only; the LR candidates are 1e-5/2e-5 and lambda candidates are 0.1/0.5/1.0.

## Predict a new claim

Save a UTF-8 JSON file with two required strings:

```json
{
  "claim": "Cầu đã mở cửa.",
  "context": "Cầu chưa mở cửa. Công trình vẫn đang được kiểm tra."
}
```

```powershell
.\.venv\Scripts\python.exe -m vifactcheck predict --checkpoint runs\smoke-42\m2_best.pt --input example.json --output runs\prediction.json
```

The included `examples/claim.json` contains an actual public dev claim and its article and can be used immediately in place of `example.json`.

The result contains the predicted label, all three probabilities, ranked sentence IDs, original selected sentences and offsets, evidence weights, and truncation metadata. No gold evidence or label is supplied to this prediction path.

## Experimental safeguards and laptop defaults

Both components use `vinai/phobert-base` with separate parameters. Every pair has at most 256 tokens including special tokens, and at most 96 claim tokens. Articles with fewer than 20 or 5 sentences use their actual sizes. Ties resolve by original sentence index. There are no synthetic evidence sentences and gold is never added to BM25 candidates.

Verification loss is negative log probability of the claim label after aggregation, computed with log-sum-exp for numerical stability. Evidence loss is balanced BCE with half its weight on positive candidates and half on negative candidates. REFUTED gold is positive retrieval supervision. Evidence loss requires valid alignment and at least one positive and one negative candidate; NEI and unavailable gold receive verification loss only. The gold matcher uses exact NFC/whitespace-normalized snippets, including multiple newline-delimited snippets. Edited, reordered, or ambiguous snippets are listed for manual review rather than matched approximately. The original text is retained. Segmentation failures are reported separately.

M2's verification gradient passes through the selected softmax weights. Discrete top-k indices have no gradient. Evidence loss supervises all eligible BM25 candidates. M1 and A1 run a frozen reranker in evaluation mode. A2 blocks the verification gradient at the selected scores and retains the evidence-loss gradient.

The laptop profile uses one claim at a time, one sentence pair per encoder call, eight-claim accumulation, FP16 autocast with FP32 parameters/aggregation, activation checkpointing, and Adafactor. All encoder parameters remain trainable in M2 and A2. Adafactor uses explicit LR, no relative scaling, no first-moment buffer, internal update clipping, no extra gradient clipping, and 10% warmup to a constant LR. The proposal does not prescribe an optimizer, effective batch size, warmup schedule, or classifier head; these are recorded implementation defaults. The heads apply dropout and a linear layer to the first-token representation.

An out-of-memory error records diagnostics and stops. The code does not automatically freeze layers, shorten inputs, switch to adapters, or reduce evidence counts. CPU execution is an explicit `--device cpu` option. Full training duration must be measured; small validation does not establish its cost or accuracy.

## Outputs

- `data/prepared/manifest.json`: counts, alignment coverage by label, asset revisions/checksums, cross-split overlap, and processed-file checksums.
- `*_alignment_review.jsonl` and `*_rejected.jsonl`: evidence and preprocessing cases requiring review.
- `runs/<run>/experiment.json`: settings, hardware, package versions, revisions, and exact sample IDs.
- `*_last.pt`, `*_best.pt`, and `*_history.json`: resumable phases and validation history. Best/last files share storage using hard links where supported.
- `*_dev_metrics.json` and `*_dev_predictions.jsonl`: classification and evidence metrics, denominators, error-stage counts, truncation counts, cost, and per-claim predictions.
- `m1_checks.json`, `a1_checks.json`, `m2_checks.json`, `a2_checks.json`: frozen-selector hashes, common initialization, and verification/evidence gradient controls.
- `smoke_report.json` and `*_example_prediction.json`: real-model validation and checkpoint/prediction consistency checks.

Evidence metrics exclude NEI and unaligned gold. MRR uses the first gold sentence in the candidate ranking and is zero when none is retrieved. Complete-gold coverage requires all annotated gold sentences. The dataset stores snippets rather than alternative evidence sets, so no alternative-set oracle is assumed. Error stages describe missing gold in BM25, missing gold after selection, and a wrong verdict despite complete selected gold; they are diagnostic categories, not causal proof.

## Sources

- Research specification: [the preserved Proposal_v2.docx](docs/Proposal_v2.docx).
- [Official ViFactCheck dataset](https://huggingface.co/datasets/tranthaihoa/vifactcheck) and [research repository](https://github.com/TTHHA/ViFactCheck).
- [PhoBERT model and segmentation guidance](https://github.com/VinAIResearch/PhoBERT).
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP).
- [Adafactor documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules).
