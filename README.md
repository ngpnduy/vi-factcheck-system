# Vietnamese Fact-Checking System

Research code for evidence-based Vietnamese fact-checking.

The repository includes the ViFactCheck paper reproduction (PLM/LLM classifiers
and BM25/SBERT/hybrid retrieval), the M0 within-context baseline, and a separate
corpus BM25 experiment. See [the code organization guide](docs/code_organization.md)
for their entry points, data formats, and dependency requirements.

## Rush-version: standalone M0/M1/M2 pilot

The `rush/` subproject contains a separate, tested M0/M1/M2 implementation and
the 900-claim, seed-42 deadline pilot. See [its setup and commands](rush/README.md)
and [protocol differences](rush/docs/BRANCH_NOTES.md). It uses its own environment
and data/artifact directories; its scores belong to its recorded protocol.

The existing `src/factcheck/` workflows continue to use the setup below.

## Quick start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `bash scripts/setup.sh`.
3. Activate the virtual environment:
   - On macOS/Linux: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`
4. Run tests to verify the setup: `pytest`

*(Note: When you are done working, you can exit the virtual environment by running `deactivate`)*

## ViFactCheck data pipeline

Download, normalize, validate, and publish the official ViFactCheck splits:

```bash
bash scripts/download_vifactcheck.sh
bash scripts/preprocess_vifactcheck.sh
pytest
```

Raw source files are written to `data/raw/vifactcheck/`; normalized intermediate
files to `data/interim/vifactcheck/`; and the validated public artifacts to
`data/processed/vifactcheck/`. The dataset files are intentionally ignored by Git.

The private internal test set is frozen only after adjudication;
the procedure is in `docs/internal_testset_guideline.md`.

The environment is locked in `uv.lock`; do not edit that file manually.

## M0: within-context BM25 + PhoBERT

See [the M0 guide](docs/m0.md) for the protocol, configuration, artifacts, and
current experiment status. M0 retrieves sentences only from each claim's own
context and averages the verifier's per-sentence probabilities.

```bash
bash scripts/setup_m0.sh             # Download pinned public models; requires Java 11+
bash scripts/run_m0.sh prepare       # Prepare train and validation
bash scripts/run_m0.sh train --smoke # Small technical check, not a research score
bash scripts/run_m0.sh train --seed 42
```

Training uses the existing baseline dependency group. Full runs select checkpoints
on validation only. The current learning rate is provisional pending M1 selection;
official three-seed test results have not yet been produced.

## ViFactCheck paper reproduction

After the common setup and data preparation above:

```bash
bash scripts/train_plm.sh --model-name vinai/phobert-base --input-mode gold_evidence
```

Generate predictions with the PLM inference module, then evaluate them with
`scripts/eval_plm.sh`; see the [complete workflow](docs/code_organization.md). Paper sentence-retrieval
evaluation uses NLTK; download its tokenizer resources explicitly before running:

```bash
.venv/bin/python -m nltk.downloader punkt_tab
PYTHONPATH=src .venv/bin/python -m factcheck.retrieval.eval_bm25 \
  --gold data/processed/vifactcheck/validation.jsonl
```

LLM training/inference additionally require a compatible Unsloth/TRL GPU
environment; those optional dependencies are not included in the shared lockfile.
