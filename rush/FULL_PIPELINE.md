# Full-data M0/M1/M2 pipeline

The full runner uses all 5,062 train, 723 dev and 1,447 test claims. It preserves the
256/96 token and 20/5 evidence limits, separate PhoBERT-base parameters, Adafactor,
eight-claim accumulation, mixed precision and activation checkpointing. M2 trains
both encoders. A1, A2, D0 and manual annotation remain outside this run.

## Execution order

1. Verify pinned data/assets and profile eight long full-training claims using M2.
2. Tune LR 1e-5/2e-5 with M1 and seed 42 on dev only. Each trial has three reranker,
   three initialization and two continuation epochs.
3. Tune M2 lambda 0.1/0.5/1.0 with seed 42 and the selected LR. All lambda trials
   branch from the same selected-LR warmup checkpoint and run two continuation
   epochs. Initialization is independent of lambda because warmup uses M1.
4. Reuse the selected seed-42 M1/M2 tuning checkpoints. Train M0 for five epochs
   with the selected LR and the same initial verifier weights. Lambda affects M2
   evidence loss only; the LR-trial M1 checkpoint stores its unused default lambda.
5. Train M0/M1/M2 on all training data with seeds 13 and 2026. M1 and M2 share the
   best three-epoch initialization checkpoint within each seed.
6. After all settings and training are finalized, evaluate all nine checkpoints on
   test and verify checkpoint reload/raw-text prediction consistency.
7. Write per-seed scores, means and sample SD, per-label results, confusion matrices,
   evidence metrics and denominators, costs and per-seed article bootstrap intervals.

Tuning ties retain search order; checkpoint ties retain the earlier epoch. Test
scores never select settings. The pilot already evaluated this public test split;
the full run is not a new blind test set. Official source-article overlap remains
an evaluation limitation.

## Start and resume on Windows

Prepare the environment/assets using the README first. The background launcher
uses a local C-drive output directory by default; it does not change power settings.
It temporarily prevents automatic system sleep during the process. Shutdown,
restart and manually sleeping the laptop interrupt the run.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_full.ps1 `
  -OutputDirectory "$env:LOCALAPPDATA\ViFactCheck\full-m0-m1-m2-20261013" `
  -StatusDirectory "runs\full-m0-m1-m2-20261013"

.\.venv\Scripts\python.exe -m vifactcheck status --run runs\full-m0-m1-m2-20261013
```

Repeat the same launcher after an interruption to resume. Completed jobs are
verified and skipped. Incomplete epochs restart from their last completed epoch.
Completed jobs discard superseded last checkpoints only after a completion ledger
is written; best checkpoints and required parent best checkpoints are retained.
An OS lock prevents simultaneous writers. Changed source/data/settings require a
new output directory. The earlier smoke and 900-claim pilot directories are untouched.

To stop gracefully, create `STOP_REQUESTED` in the status directory. The runner
stops at its next progress update and preserves complete-epoch checkpoints. Remove
that file before explicitly resuming. No automatic architecture changes or retries
follow memory, storage or numerical failures; diagnostics are saved.

The command also works in the foreground:

```powershell
.\.venv\Scripts\python.exe -u -m vifactcheck full `
  --output "$env:LOCALAPPDATA\ViFactCheck\full-m0-m1-m2-20261013" `
  --status-dir runs\full-m0-m1-m2-20261013
```

## Records

The status directory holds `LOCATION.json`, `launch.json`, progress and console
logs. Final report and comparison files are mirrored there after completion.
`LOCATION.json` records the physical output path, including any Windows packaged-app
AppData redirection. The launcher reuses that recorded location when resuming.
The output directory holds all trial/seed checkpoints, exact configurations,
source/data hashes, metrics, predictions and completion ledgers. Runtime requires
roughly 60–80 hours on the pilot laptop; this estimate depends on current throughput.
Keep the laptop plugged in. There is no six-hour deadline or training subset.
