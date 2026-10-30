# Single-seed slide experiment

Seed 42; architecture listwise; all 5,062 train / 723 dev / 1,447 test claims.

Fixed learning rate 2e-05, evidence lambda 0.5. Values transferred from the earlier pairwise study, without a new search for this architecture. Checkpoints selected on validation only.

Reranker: 3 epochs. Shared weighted warmup: 3 epochs. M1/M2/A2: 2 continuation epochs each. M0/A1: 5 verifier epochs each. Separate PhoBERT parameters, 256/96 token limits, 20/5 sentence limits, accumulation 8, AMP, activation checkpointing and Adafactor.

| Model | Dev Macro-F1 | Test Macro-F1 | Accuracy |
|---|---:|---:|---:|
| M0 | 0.5886 | 0.5771 | 0.5909 |
| M1 | 0.6900 | 0.6768 | 0.6800 |
| A1 | 0.6015 | 0.5862 | 0.5950 |
| M2 | 0.7180 | 0.7046 | 0.7049 |
| A2 | 0.6928 | 0.6642 | 0.6676 |

Per-label scores, confusion matrices, retrieval denominators, alignment coverage, truncation and inference/memory costs are in comparison.json and models/*_test_metrics.json. Epoch histories record training runtime and peak GPU memory; shared phases ran once.

| Contrast | Macro-F1 difference | Conditional 95% interval |
|---|---:|---|
| A1-M0 | +0.0092 | [-0.0098, +0.0288] |
| M1-A1 | +0.0906 | [+0.0635, +0.1166] |
| M2-M1 | +0.0278 | [+0.0072, +0.0488] |
| M2-A2 | +0.0405 | [+0.0199, +0.0618] |

No new hyperparameter search or between-training-seed variance estimate. Bootstrap intervals are conditional on the fitted models.

This is a separate study from the original three-seed pairwise results. Official splits share source articles. The candidate-attention layer is inspired by the slides; it is not a complete reproduction of M-ReRank/RAV.
