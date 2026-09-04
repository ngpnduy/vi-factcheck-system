# Vietnamese Fact-Checking System

Research code for evidence-based Vietnamese fact-checking.

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

To process an official ViNumFCR release placed in `data/raw/vinumfcr/`, run
`bash scripts/preprocess_vinumfcr.sh`. See `docs/vinumfcr_import.md` before
importing it. The private internal test set is frozen only after adjudication;
the procedure is in `docs/internal_testset_guideline.md`.

The environment is locked in `uv.lock`; do not edit that file manually.
