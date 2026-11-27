# Completed M0/M1/M2 pilot results

All three configurations completed training, evaluation and raw-text prediction
checks on 4 October 2026. The workflow took about 4 hours 5 minutes on an RTX 3050
Laptop GPU. This is a reduced-data, single-seed pilot: 900 training claims, seed 42,
all 723 validation claims and all 1,447 test claims. Hyperparameters were fixed
before training and were not tuned.

| Model | Test Macro-F1 | Test accuracy |
|---|---:|---:|
| M0 | 51.01% | 51.69% |
| M1 | 52.48% | 54.11% |
| M2 | 54.84% | 56.46% |

## Read or download

- [Complete report](REPORT.md): per-label scores, evidence metrics, confidence
  intervals, runtime, memory, confusion matrices, examples and limitations.
- [Results ZIP (6.5 MB)](M0-M1-M2-pilot-900-seed42-results.zip?raw=true): extract it
  and open `REPORT.html` in a browser. It includes the report, metrics, all dev/test
  predictions, experiment records and the matching source code.
- [Results CSV](results.csv): summary scores for a spreadsheet or report table.
- [Browser report](M0-M1-M2-pilot-report.html?raw=true): download and open locally.
- [ZIP SHA256](M0-M1-M2-pilot-900-seed42-results.zip.sha256): archive integrity check.

The ZIP contains the source snapshot at commit `d8416b9`, which produced these
results. Source hashes are recorded in `pilot_spec.json` and the archive manifest.
Historical source-import notes in that snapshot predate completion; the finished
pilot results supersede their earlier running status.

## Interpretation

M2 achieved the best test Macro-F1. Its main gain over M1 came from NEI detection,
while REFUTED F1 and top-five evidence coverage declined. M0 performed best on
validation. These results support further investigation of joint training;
they do not establish consistent superiority across seeds or full-data training.
Official splits share source articles, and no unseen-article evaluation was run.

## Experiment records

The JSON files in this folder preserve full metrics, settings, exact sampled IDs,
epoch histories, behavioral checks and paired article-cluster bootstrap results.
Per-claim prediction files are included in the ZIP to avoid duplicating them here.
`completion.json` records checksums for the test artifacts and prediction checks.

The three retained trained checkpoints total approximately 2.7 GB. They remain in
the original laptop run folder and are excluded from this results package.
Downloaded assets, datasets and environments are also excluded. Running these
trained models requires obtaining their checkpoints separately; see the source
[setup and command guide](../../README.md).
