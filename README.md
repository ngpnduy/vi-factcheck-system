# Vietnamese Fact-Checking System

The **Listwise v2 five-model, three-seed study is complete**. See the
[report and full results download](research/results/listwise-best5-v2-three-seeds-20261112/README.md).
M2 test Macro-F1 is **0.7066 +/- 0.0057**, the mean and sample SD across three seeds.

| Implementation | Entry point | Environment / protocol |
|---|---|---|
| Original paper reproduction and M0 | `src/factcheck/` and root `scripts/` | Root `uv.lock`; original data/model workflows |
| M0/M1/M2/A1/A2 research study | [research/](research/README.md) | Own pinned environment; PhoBERT-base, VnCoreNLP, Adafactor, Listwise v2 |

These implementations have separate preprocessing, dependencies and protocols.
Do not install the research requirements into the root environment or interchange
scores/checkpoints. See [code organization](docs/code_organization.md).

The standalone project moved from `rush/` to `research/`. The Python package and
CLI stay `vifactcheck`; [migration instructions](research/MIGRATION.md) describe the
forwarding PowerShell launchers and new environment location.

Historical [pilot](rush/results/pilot-900-seed42/README.md) and
[single-seed study](rush/results/full-five-listwise-seed42-20261030/README.md)
remain at their original URLs. The application version is unchanged; study
publications use separate experiment tags.

The setup below applies to the **original `src/factcheck/` workflows**.

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
official three-seed results for this original `src/factcheck/` M0 workflow have not been produced. The completed study linked above belongs to the separate research implementation.

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
