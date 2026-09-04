# ViNumFCR import

ViNumFCR is normalized into the shared contract in
[`data_contract.md`](data_contract.md). Its paper describes paragraph-claim-evidence
examples, three verdict labels, and official `train`, `dev`, and `test` splits.
The paper does not define a stable public dataset URL or the exact column names
used by a release, so the project deliberately does **not** guess a download endpoint.

## Before importing

1. Obtain the official data under its research license and record its URL, release
   identifier, and SHA-256 checksums in `data/raw/vinumfcr/source_metadata.json`.
2. Store each original split without edits as
   `data/raw/vinumfcr/train.jsonl`, `dev.jsonl`, and `test.jsonl`.
3. Inspect one row. The aliases accepted by the importer are documented in
   [`configs/data/vinumfcr.yaml`](../configs/data/vinumfcr.yaml). If the official
   fields differ, add an explicit alias and a fixture before processing all data.

## Processing

```bash
bash scripts/preprocess_vinumfcr.sh
```

The script preserves numerical strings verbatim and writes detected numeric
expressions to `metadata.numeric_expressions`. It maps `dev` to the canonical
`validation` split, validates the shared contract, and writes checksums to
`data/processed/vinumfcr/manifest.json`.
