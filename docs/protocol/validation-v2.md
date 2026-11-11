# Protocol v2 validation - 6 November 2026

This is the validation record for source commit `baa0bdc`, before the release
source was moved from `rush/vifactcheck/` to `src/vifactcheck/`. The storage and
training status below describe the check on 6 November, not a live run status.

All 85 unit/integration cases passed with zero failures, errors or skips. Two benign SWIG deprecation warnings were emitted.

Coverage includes every possible best epoch (1-5) for M1/M2/A2, earlier-epoch ties, branching from final epoch three, optimizer/RNG resume, recovery between last-save and best-link, legacy inference-only compatibility, the 35-GiB start gate, per-write reserve checks, and terminal-state gradient checks when warmup wins.

A complete stubbed worker exercised fresh LR/lambda selection, thirteen sealed jobs, seed-42 winner reuse, all three seeds/five configurations, fifteen paired test records and completed resume without repeated work. This is orchestration validation, not real full-data training.

## Real PhoBERT smoke

Seed 42, 24 balanced training claims and 12 validation claims; all five configurations completed training, dev evaluation, reload and raw-text prediction in 9.54 minutes on the RTX 3050 Laptop GPU. Full token limits 256/96 and evidence limits 20/5 were retained.

Smoke uses one reranker epoch, one warmup epoch and one continuation epoch; M0/A1 receive two verifier epochs. Selection therefore covers two smoke verifier epochs. The five-epoch selection cases are covered separately by regression tests. Smoke scores are pipeline checks.

| Model | Peak allocated MiB | Peak reserved MiB | Selected smoke epoch | Reload/raw difference |
|---|---:|---:|---:|---:|
| M0 | 1807.29 | 1938 | 1 | 0 / 0 |
| M1 | 2407.07 | 2576 | 2 | 0 / 0 |
| M2 | 2948.85 | 3136 | 2 | 0 / 0 |
| A1 | 2407.07 | 2576 | 2 | 0 / 0 |
| A2 | 2953.46 | 3138 | 2 | 0 / 0 |

M2 verification-only reranker-head gradient norm: 0.318885. A2 verification-only norm: 0; evidence-only norm: 2.884920. Both encoders' first/last query matrices changed in M2/A2. The complete M1/A1 selector stayed frozen and identical.

These are smoke-claim memory peaks, not a full-dataset worst-case guarantee. The full worker profiles eight largest training workloads after its storage gate. CPU resume fixtures match exactly; seeded GPU training is not claimed to be bitwise reproducible because PyTorch warned about nondeterministic memory-efficient attention backward.

## Checkpoint compatibility and storage

The published v1 listwise M1 PhoBERT checkpoint reloaded with inferred protocol 1, identical sentence ranking and zero maximum probability difference from its saved dev prediction. Its stored checksum matched and it was not modified.

After the GPU smoke, a universal pre-write capacity guard was added. Final validation checked low-space refusal without touching the existing file, then wrote/read each of the five real smoke payloads on C and verified complete model/optimizer/scaler/RNG/metadata equality. Only the temporary probe files were removed. Numerical training/selection code was unchanged; both source-hash sets are recorded below.

At the recorded check, full tuning/three-seed training had **not started**: storage readiness was `needs_space`, with 6.08 GiB free versus the required 35 GiB. Historical checkpoints and published result hashes were preserved.

[Machine-readable evidence and source hashes](https://github.com/ngpnduy/vi-factcheck-system/blob/baa0bdc80379d26e661bd1dcf5e37b3b48ba14a8/rush/docs/validation/listwise-v2-20261106.json) remain at the archived commit. Local raw artifacts were in the original experiment workspace `runs/listwise-v2-validation-20261106/`. See [launch/resume instructions](v2.md).
