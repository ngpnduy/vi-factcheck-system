# M0 M1 M2 implementation validation

This file records the original 24-train/12-dev smoke validation. The larger 900-claim pilot was launched afterwards and was still running when this branch's source snapshot was imported. Its test results are not included in this validation record. The Rush checkout's additional pilot/report tests are separate from the original 24-test smoke milestone.

## Rush branch verification

The imported checkout passed 28 tests with one skipped on 4 October 2026, using its own editable installation and environment. The skipped integration test requires downloaded VnCoreNLP assets, which are absent from this checkout. The original workspace previously passed all 29 tests with those assets available. Pilot and training command help also loaded successfully. All 13 imported Python source files match the hashes recorded in `PROVENANCE.json`; the source used by the active experiment was preserved.

The original smoke validation record follows.

Validation completed on 4 October 2026 on the local RTX 3050 Laptop GPU with 4 GiB VRAM. The original proposal was preserved.

## Outcome

All three actual PhoBERT configurations completed training, evaluation, checkpoint reload, and raw-text prediction. The final test suite passed 24 tests. Full experiments and test-set evaluation were not launched.

The GPU validation used seed 42, 24 train examples (8 per label), and 12 dev examples (4 per label). SUPPORTED/REFUTED examples were sampled from claims with eligible exact evidence alignment. One reranker epoch, one common verifier initialization epoch, and one continuation epoch were used; M0 received two verifier epochs. Token budgets stayed at 256/96 and evidence budgets at 20/5.

These scores are pipeline checks on a small, selected subset. They do not establish model superiority or reproduce published scores.

| Model | Dev Macro-F1 | Accuracy | Peak allocated GPU MiB | Peak reserved GPU MiB |
|---|---:|---:|---:|---:|
| M0 | 0.4148 | 0.4167 | 1807.3 | 1938.0 |
| M1 | 0.4312 | 0.4167 | 2321.9 | 2488.0 |
| M2 | 0.4815 | 0.5000 | 2907.5 | 3102.0 |

Total smoke workflow time: 266.4 seconds (4.44 minutes), excluding preparation and the test suite. GPU figures are PyTorch allocator measurements; driver/display allocations are separate.

## Behavioral checks

- M0, M1, and M2 used the same initial verifier parameter hash.
- M1 and M2 used exactly the same common warmup checkpoint and restored its verifier optimizer state.
- M1 reranker parameters remained unchanged.
- M2 kept all parameters of both encoders trainable and updated its reranker.
- M2 verification loss alone produced reranker-head gradient norm 0.23015809; evidence loss was excluded from that check.
- Actual first/last attention query matrices in both M2 backbones changed relative to the warmup checkpoint; see `runs/smoke-42/m2_backbone_update_checks.json`.
- Checkpoint reload and independently segmented raw-text prediction gave zero probability difference for every model on the checked dev example.
- An interrupted two-epoch unit training run resumed to the same final weights as uninterrupted training.
- Tests covered numeric/date/abbreviation boundaries, Unicode normalization, exact and ambiguous evidence matches, label gating, 256/96 truncation, weighted aggregation, top-k gradients, hand-calculated retrieval metrics, frozen M1 parameters, M2 encoder gradients, shared warmup optimizer restoration, and asset checksum repair.

## Data preparation

All official examples were retained for classification. Evidence matching was intentionally exact after Unicode/whitespace normalization. Edited or ambiguous evidence strings require review and do not provide automatic reranker labels.

| Split | Source examples | Retained | Valid evidence alignment | Unmatched | Ambiguous |
|---|---:|---:|---:|---:|---:|
| train | 5062 | 5062 | 2952 | 2064 | 46 |
| dev | 723 | 723 | 425 | 293 | 5 |
| test | 1447 | 1447 | 857 | 579 | 11 |

Retrieval metrics exclude NEI and unaligned examples. The smoke retrieval denominator was 8 SUPPORTED/REFUTED dev claims. Full-data alignment counts by label and review cases are saved under `data/prepared`.

Official splits share source articles: 495 unique URLs across train/dev, 753 across train/test, and 360 across dev/test. There is one exact normalized claim shared by train/dev and one by train/test. The prescribed official splits were retained; the overlap audit is saved in the data manifest.

## Reproduction records

- Dataset revision: `2df623632299a3459ad95feeff9ac67d7205a42a`.
- PhoBERT revision: `01daacda68afe13d83023d16ec647239e344a1e6`.
- VnCoreNLP revision: `62bbc58fe5d113c898eae112656be97dcf50b3a0`.
- Original DOCX SHA-256: `7be6c0d11b452f0fa2eb4054cbef1deab791d70902d2f376d5a91dfad94ba3e0`.
- Exact package versions: `requirements-tested.txt`.
- Settings, hardware, revisions, and exact subset IDs: `runs/smoke-42/experiment.json`.
- Complete measurements: `runs/smoke-42/smoke_report.json`.
- Setup and full-experiment commands: `README.md`.

PyTorch emitted a warning that its memory-efficient attention backward kernel may be nondeterministic despite seeded execution. The completed smoke run and reload checks passed; bitwise identity of independently repeated GPU training is not claimed.

Future full experiments still require validation-only hyperparameter selection, all three seeds, and finalized test evaluation. Their accuracy and full-run resource requirements have not been measured.
