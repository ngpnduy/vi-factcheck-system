# ViFactCheck research: M0, M1, M2, A1 and A2

The completed **Listwise v2 three-seed study** uses all 5,062 train / 723 dev /
1,447 test claims. Validation-only tuning selected LR **1e-5** and lambda **0.1**.
M2 test Macro-F1 is **0.7066 +/- 0.0057** (mean and sample SD across seeds).

- [Completed study, reports and full ZIP download](results/listwise-best5-v2-three-seeds-20261112/README.md)
- [Authoritative Listwise v2 protocol and resume guide](LISTWISE_V2.md)
- [Repository migration and validation](MIGRATION.md)
- [Historical protocols and validation snapshots](docs/history/README.md)
- [Original proposal](docs/Proposal_v2.docx)

This is a separate implementation from the repository's `src/factcheck/` workflows.
It uses PhoBERT-base, VnCoreNLP, BM25Okapi and Adafactor. The original workflow uses
its own backbone/preprocessing and a separate dependency lock. Their scores and
data/checkpoint formats are not interchangeable. `rush/results/` keeps the
previously published pilot and single-seed studies at their original URLs.

## Windows setup

Run all commands below **from `research/`**. The validated laptop uses Python
3.12, Java 8 and an RTX 3050 4-GiB GPU. Java must be available on PATH. The package
name, console command and CLI remain `vifactcheck`.

```powershell
# Fresh research-local environment with the exact recorded package versions:
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# Alternative: reuse matching installed packages without changing global packages:
# powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -ReuseInstalledPackages

.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m vifactcheck --help
.\.venv\Scripts\python.exe -m vifactcheck prepare
```

Setup refuses to replace an existing environment. Reuse mode checks the exact pins
in `requirements-tested.txt`; mismatches stop setup. It does not install globally
or promise Python execution from the old `rush/` directory. Import verification
must resolve to `research/vifactcheck/`. The original root `uv.lock` is separate
and must not be used to install the research environment.

On a drive with insufficient space, setup accepts `-EnvironmentDirectory C:\path\to\a-new-environment`:
it creates a new environment there and a `research/.venv` junction. This laptop
uses that arrangement because the CUDA Torch package alone exceeds E's free space.
Its validation uses isolated packages; the earlier shared-package attempt hit a
native crash during optional AWS imports. No old environment was moved or changed.

Preparation records pinned dataset/model revisions and checksums and validates
5062/723/1447 split counts. Data goes in `data/prepared/`, downloads in `.cache/`.
`prepare --offline` reuses verified downloads. VnCoreNLP needs a temporary path
without spaces; the full launcher sets one outside the source tree.

## Smoke, tuning and full training

```powershell
# Pipeline check only: 24 balanced train / 12 dev, all five real PhoBERT models.
.\.venv\Scripts\python.exe -m vifactcheck smoke `
  --models M0 M1 A1 M2 A2 --reranker-architecture listwise `
  --output runs\smoke-listwise-v2

# Storage preflight only; does not launch a worker.
powershell -ExecutionPolicy Bypass -File scripts\start_listwise_v2.ps1 -CheckOnly

# Full validation-only tuning, all five models x three seeds, then all 15 tests.
# Repeating the identical command resumes verified completed phases.
powershell -ExecutionPolicy Bypass -File scripts\start_listwise_v2.ps1
```

The launcher accepts explicit data/model/segmenter/output/status directories and
can discover this laptop's assets in the enclosing workspace. It preserves the
physical run location recorded in `LOCATION.json`. A new full run requires 35 GiB
free; every checkpoint write keeps a reserve. Setup, tests and prediction do not
start training. The completed run remains in its original location.

The `full` workflow performs the LR search `{1e-5, 2e-5}` with M1 and lambda search
`{0.1, 0.5, 1.0}` with M2 on seed 42/dev, then locks settings for seeds 13/2026.
M1/M2/A2 branch from **final warmup epoch three**, including optimizer/scaler/RNG,
and select the best dev Macro-F1 across all five verifier epochs. Ties favor the
earlier epoch. See the protocol guide for standalone train/tune/evaluate/resume.

M0 uses BM25/uniform pooling; M1 frozen ranking/weighted pooling; A1 frozen
ranking/uniform pooling; M2 joint ranking/weighted pooling; A2 detaches verification
feedback to ranking while retaining evidence gradients. NEI has no evidence loss.
Both M2 encoders stay trainable. Limits remain 256/96 tokens and 20/5 sentences,
one claim per batch, accumulation eight, AMP and activation checkpointing.

## Prediction and evaluation

```powershell
.\.venv\Scripts\python.exe -m vifactcheck predict `
  --checkpoint <local-run>\seeds\seed42\m2_best.pt `
  --input examples\claim.json --segmenter-path <vncorenlp-directory> `
  --output runs\prediction.json

.\.venv\Scripts\python.exe -m vifactcheck evaluate `
  --checkpoint <local-checkpoint> --data <prepared-directory> `
  --split test --output runs\evaluation
```

Prediction requires UTF-8 JSON with `claim` and `context`, without gold labels or
evidence. It returns a label, three probabilities, selected original sentences
and offsets, evidence weights and truncation metadata. Empty input is rejected.
Only load trusted locally generated checkpoints. Archives contain predictions
and checkpoint hashes, **not weights**; use local checkpoints for new predictions.

## Export and verify the completed study

```powershell
.\.venv\Scripts\python.exe scripts\export_results.py export `
  --run <completed-run-directory> `
  --output results\listwise-best5-v2-three-seeds-20261112 `
  --archive exports\ViFactCheck-listwise-v2-three-seeds-results.zip

.\.venv\Scripts\python.exe scripts\export_results.py verify `
  --output results\listwise-best5-v2-three-seeds-20261112 `
  --archive exports\ViFactCheck-listwise-v2-three-seeds-results.zip
```

The standard-library exporter verifies source/report/artifact hashes, all paired
15 x 1,447 test predictions, recomputed classification and retrieval metrics,
three-seed sample SD and the 12 article-bootstrap records. Original artifacts are
copied without editing their bytes. The compact Git publication excludes raw
prediction files; the GitHub Release ZIP includes them and the executed source.
ZIP and payload hashes are separate. Export refuses to overwrite a different run.

The run's executed source commit is `baa0bdc`, before the directory rename. Its
15 Python modules remain byte-identical in this checkout. Publication metadata
separately records the original source, path mapping and study tag.
