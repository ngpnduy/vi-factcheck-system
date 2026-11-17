# Saved-history convergence audit

Input: `E:\Bean\hcmut\documents(real)\programming inergration project\vi-factcheck-system\reports\experiments\listwise-best5-v2-three-seeds-20261112`

This is a read-only analysis of recorded train/dev histories. It does not train, load weights, or read test scores.

| Model | Seed | Epochs | Best dev epoch | Best dev Macro-F1 | Best at final epoch | Final-stage signal |
|---|---:|---:|---:|---:|---|---|
| M2 | 13 | 5 | 4 | 0.709033 | False | training_loss_decreases_but_dev_declines |
| M2 | 42 | 5 | 5 | 0.706201 | True | dev_improving_at_stage_boundary |
| M2 | 2026 | 5 | 4 | 0.687712 | False | training_loss_decreases_but_dev_declines |

## Interpretation

- Five verifier epochs and two joint-continuation observations do not establish convergence or a global optimum.
- Warmup and continuation can use different objectives and trainable parameters; do not interpret their losses as one stationary optimization curve.
- Lower training loss alone does not establish better verdict prediction; use held-out dev Macro-F1 and, when recorded, dev verification NLL.
- A dev decline with lower training loss is an overfitting signal, not proof of overfitting from one transition.
- No test scores are read or used for selection. A longer run is a new diagnostic protocol and must not overwrite the published five-epoch results.
- Existing histories may not contain dev NLL or separate verification/evidence losses; missing values remain blank rather than being reconstructed from Macro-F1.
- New `loss_components` fields describe online training with dropout enabled. `diagnostics.train_probe` and `diagnostics.dev` describe fixed-checkpoint evaluation with dropout disabled; compare their NLL on clearly reported sample counts.
- An observed plateau describes only this finite window, learning rate, objective and data. It is not certification of the best possible model.

## Within-stage loss changes

| Model | Seed | Stage | Epoch range | Observations | First objective | Last objective | Latest dev F1 change |
|---|---:|---|---|---:|---:|---:|---:|
| M2 | 13 | warmup | 1-3 | 3 | 1.057997 | 0.633950 | 0.056924 |
| M2 | 13 | continuation | 4-5 | 2 | 0.447933 | 0.281225 | -0.001006 |
| M2 | 42 | warmup | 1-3 | 3 | 1.058126 | 0.664071 | 0.039803 |
| M2 | 42 | continuation | 4-5 | 2 | 0.492155 | 0.334965 | 0.027551 |
| M2 | 2026 | warmup | 1-3 | 3 | 1.059311 | 0.671655 | 0.028526 |
| M2 | 2026 | continuation | 4-5 | 2 | 0.507966 | 0.341371 | -0.006629 |

## Provenance

- `seeds/seed13-branches/m2_history.json`: SHA-256 `63035cab4dbc1ca65c01abd5f846e53d5eccfd5dab6310c308c55743b0ea3b81`; identical copies: none.
- `tuning/lambda_0.1/m2_history.json`: SHA-256 `198c1d75eb02fb8dd4ad02620e56a0049d047513a176cd467ba42da167244ce0`; identical copies: none.
- `seeds/seed2026-branches/m2_history.json`: SHA-256 `d2e3914678c249640c5b9479a6b40ca48848c58fa8d0a7022bda6fa964eeb502`; identical copies: none.

Nonwinning tuning histories were excluded using the study's recorded dev-selected hyperparameters. Use `--all-trials` to inspect them separately.
