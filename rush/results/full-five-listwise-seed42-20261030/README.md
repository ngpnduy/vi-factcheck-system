# Completed five-model list-aware experiment

Completed on 30 October 2026 at 13:06 Vietnam time. All five configurations trained on all 5,062 official training claims, selected checkpoints using all 723 validation claims, and evaluated all 1,447 test claims. Training seed: 42.

| Model | Test Macro-F1 | Test accuracy |
|---|---:|---:|
| M0 | 57.71% | 59.09% |
| M1 | 67.68% | 68.00% |
| A1 | 58.62% | 59.50% |
| M2 | 70.46% | 70.49% |
| A2 | 66.42% | 66.76% |

## Read or download

- [Original experiment report](REPORT.md)
- [Score table](results.csv)
- [Download the complete results ZIP](ViFactCheck-five-models-listwise-seed42-results.zip?raw=true)
- [ZIP SHA-256](ViFactCheck-five-models-listwise-seed42-results.zip.sha256)
- [Full metrics and comparisons](comparison.json)
- [Training runtime and memory](runtime_memory.json)
- [Export verification and file hashes](EXPORT_MANIFEST.json)

The ZIP contains all dev/test per-claim predictions, selected sentences and weights, examples, per-label scores, confusion matrices, evidence metrics with denominators, alignment/truncation statistics, training histories, behavioral checks, exact claim IDs, configurations, dataset/model revisions, and the matching frozen source. Original result files retain their recorded bytes. Checkpoint weights, downloaded datasets/assets, environments and application logs are excluded.

## Protocol and interpretation

Reranked configurations use list-aware attention over at most 20 BM25 candidates, selecting up to five sentences. M0 uses BM25 directly. PhoBERT pair/claim limits remain 256/96 tokens. Adafactor, eight-claim accumulation, mixed precision and activation checkpointing were used. Reranker pretraining: three epochs; shared verifier warmup: three; M1/M2/A2 continuation: two each; M0/A1 verifier training: five each.

Learning rate 2e-5 and evidence lambda 0.5 were transferred from prior pairwise validation tuning. No new search was performed for list-aware attention; test scores never selected checkpoints or settings.

| Contrast | Test Macro-F1 difference | Conditional 95% interval |
|---|---:|---:|
| A1-M0 | +0.92 points | [-0.98, +2.88] |
| M1-A1 | +9.06 points | [+6.35, +11.66] |
| M2-M1 | +2.78 points | [+0.72, +4.88] |
| M2-A2 | +4.05 points | [+1.99, +6.18] |

Intervals use 2,000 paired article-cluster bootstrap replicates and are conditional on these fitted models. This one-seed study does not measure between-training-seed variation. Official splits share source articles. This implementation is inspired by the slides rather than a complete reproduction of M-ReRank/RAV.

The laptop required a forced restart during the first M1 continuation epoch. Verified reranker/warmup checkpoints were reused, the unfinished epoch restarted, and all five configurations subsequently completed. The checkpoint audit is included in the ZIP; the cause of the freeze was not established.

## Reproduction and provenance

Training source commit: [`c07b0b1`](https://github.com/ngpnduy/vi-factcheck-system/commit/c07b0b1c7fd6d209609f1d423f57aa099c31d438). The ZIP's source_snapshot matches every hash in single_seed_plan.json. See the [setup guide](https://github.com/ngpnduy/vi-factcheck-system/blob/c07b0b1c7fd6d209609f1d423f57aa099c31d438/rush/README.md) and [single-seed commands](https://github.com/ngpnduy/vi-factcheck-system/blob/c07b0b1c7fd6d209609f1d423f57aa099c31d438/rush/docs/SLIDES_0410_IMPLEMENTATION.md).

Experiment metadata retains historical absolute laptop paths. To rerun elsewhere, install dependencies and use your own paths for data, model, segmenter, and output. Trained checkpoint weights are stored locally and are not included, so inference with these fitted models requires obtaining those weights separately.
