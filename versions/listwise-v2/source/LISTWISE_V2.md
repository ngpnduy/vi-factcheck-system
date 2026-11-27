# Listwise v2: best of five epochs, fresh tuning, three seeds

This is a new training protocol. Published October results stay in their original
folders. Legacy checkpoints still support prediction/evaluation, but cannot
initialize or resume a protocol-v2 experiment.

## Training and selection

- Pretrain the reranker for three epochs; select by validation MRR.
- Train three common weighted-verifier warmup epochs. Keep `warmup_best.pt`
  (best validation Macro-F1) and `warmup_end.pt` (the final epoch-three model,
  optimizer, AMP scaler and RNG state).
- M1/M2/A2 all branch from **warmup_end.pt**, then train global epochs four and
  five. Select their prediction checkpoint over **all five verifier epochs**;
  exact ties keep the earlier global epoch. A warmup checkpoint can win for
  M2/A2 and is saved with the correct target configuration/mode.
- M0/A1 train five verifier epochs from common initial verifier weights. M1/A1
  keep the same frozen selector. M2 trains both encoders; A2 blocks verification
  feedback to ranking but retains eligible evidence-loss gradients. NEI receives
  verification loss only.
- Validate gradients and encoder changes on the terminal continuation state,
  independently of which checkpoint wins. `*_last.pt` always stores that actual
  resume state. `*_history.json` records all verifier epochs with local/global
  epoch numbers. `*_selection.json` and `summary.json` record selected global
  epoch/stage, dev score, initialization/source hashes and final checkpoint hash.

## Fresh listwise tuning and final evaluation

Use all 5,062 official train and 723 dev claims. Tune only on seed 42/dev:

1. Train M1 listwise with LR `1e-5`, then `2e-5`; compare best-of-five dev scores.
2. With the selected LR, train M2 for lambda `0.1`, `0.5`, then `1.0`, sharing the
   same final epoch-three initialization and warmup-best selection baseline.
3. Trial ties keep search order. All configurations share the selected LR;
   M2/A2 share the selected lambda. Pairwise-study hyperparameters are not reused.
4. Reuse the new M1/M2 tuning winners for seed 42 and train its M0/A1/A2. Train
   all five configurations independently on seeds 13 and 2026 using the locked
   hyperparameters. No old seed-42 result is imported into this new study.
5. Finalize all training before the **15 test evaluations**, each using all
   1,447 test claims. Test never selects settings or checkpoints.

Thirteen sealed training jobs limit disk use: two LR trials, three lambda
trials, two remaining seed-42 jobs, and three child jobs per remaining seed.
Completed children release obsolete last states while preserving their best
and required warmup best/end checkpoints. Collection uses hard links when
available. Source/data/config hashes and completion ledgers guard resume.

Reports include selected epoch/stage, per-seed and three-seed mean/sample-SD
scores, per-label metrics, confusion matrices, retrieval denominators, alignment,
truncation, runtime and peak GPU memory. Four contrasts (A1-M0, M1-A1, M2-M1,
M2-A2) receive paired article-bootstrap intervals per seed. Repeated test claims
across seeds are not independent samples.

## Windows commands

Run from `vi-factcheck-system/rush/`. The launcher discovers prepared data/model/
segmenter folders in this subproject or its enclosing laptop workspace.

```powershell
# Storage check only; does not launch a training worker.
powershell -ExecutionPolicy Bypass -File scripts\start_listwise_v2.ps1 -CheckOnly

# Start or resume after the storage requirement is met.
powershell -ExecutionPolicy Bypass -File scripts\start_listwise_v2.ps1
```

The new checkpoint folder is `full-five-listwise-best5-v2` in the persistent
ViFactCheck base on C. The launcher uses the earlier run's `LOCATION.json` when
available, otherwise `%USERPROFILE%\AppData\Local\ViFactCheck`. Progress and
reports are mirrored to the enclosing workspace's
`runs/full-five-listwise-best5-v2/`; `LOCATION.json` records the actual physical
output location, including packaged-app redirection. Directory overrides are
`-OutputDirectory`, `-StatusDirectory`, `-DataDirectory`, `-ModelDirectory` and
`-SegmenterDirectory`.

Both launcher and worker require **35 GiB free before starting**. Every checkpoint
write also requires a 2-GiB reserve plus its temporary file. `needs_space` means
the full experiment has not started. Existing experiment weights are preserved.

The equivalent direct command is:

```powershell
.\.venv\Scripts\python.exe -u -m vifactcheck full `
  --include-ablations --reranker-architecture listwise `
  --data <prepared-directory> --model-path <phobert-directory> `
  --segmenter-path <vncorenlp-directory> `
  --output <new-directory-on-C> --status-dir <workspace-status-directory>
```

Before the full run, validate all five real models:

```powershell
.\.venv\Scripts\python.exe -u -m vifactcheck smoke `
  --models M0 M1 A1 M2 A2 --reranker-architecture listwise `
  --data <prepared-directory> --model-path <phobert-directory> `
  --segmenter-path <vncorenlp-directory> --output runs\smoke-listwise-v2
```

Smoke uses 24 train / 12 dev claims, one reranker epoch, one warmup and one
continuation epoch; selection covers both verifier epochs. M0/A1 receive two
epochs. It seals models individually to reduce peak disk use and checks terminal
gradients, encoder changes, checkpoint reload and raw-text prediction. Smoke
scores are pipeline checks. The recorded outcome is in `V2_VALIDATION.md`.

Laptop settings stay at PhoBERT-base, separate encoder parameters, pair/claim
limits 256/96, BM25 top 20 / selected top five, batch one claim, accumulation
eight, Adafactor with explicit LR and relative scaling disabled, AMP and
activation checkpointing. CUDA OOM stops with diagnostics and no automatic
architecture/token changes. The full worker first profiles eight largest
training workloads with both M2 encoders trainable.

## Resume and timing

Rerun the identical launcher command. Completed jobs are hash-verified; incomplete
epochs restart from their last completed epoch. Best-of-five state, optimizer,
scaler and RNG survive resume. Checkpoint/metadata writes flush before replacement.
If power is lost between committing last and linking best, recover the winner
from last or warmup when possible; otherwise stop for an integrity audit. Damaged
progress files are backed up and reconstructed from durable experiment state.

Create `STOP_REQUESTED` in the status directory to stop at a progress boundary;
remove it before explicitly resuming. A process lock prevents duplicate writers.
Changed source, settings, protocol or data require a fresh output directory.

Budget approximately **90-110 uninterrupted compute hours**. After the first LR
trial, `runtime_estimate.json` rescales this using measured training time. Future
downtime and repeated unsaved work are additional.
