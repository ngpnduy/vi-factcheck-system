# Listwise v2: five models, three seeds

Completed 12 November 2026 (Vietnam time), protocol 2. All 5,062 train / 723 dev /
1,447 test claims were used, with seeds 42, 13 and 2026. Validation-only tuning
selected LR 1e-5 and lambda 0.1. See [the unchanged report](REPORT.md) and
[result table](results.csv) for means and sample SD, per-seed scores, selected
epochs, retrieval denominators, memory, runtime and paired article bootstrap.

| Model | Test Macro-F1, mean +/- sample SD |
|---|---:|
| M0 | 0.5871 +/- 0.0158 |
| M1 | 0.6733 +/- 0.0140 |
| A1 | 0.5919 +/- 0.0038 |
| M2 | 0.7066 +/- 0.0057 |
| A2 | 0.6799 +/- 0.0047 |

## Download and verification

- [Full results ZIP](https://github.com/ngpnduy/vi-factcheck-system/releases/download/listwise-v2-three-seeds-20261112/ViFactCheck-listwise-v2-three-seeds-results.zip)
- [ZIP SHA-256](https://github.com/ngpnduy/vi-factcheck-system/releases/download/listwise-v2-three-seeds-20261112/ViFactCheck-listwise-v2-three-seeds-results.zip.sha256)
- [GitHub Release](https://github.com/ngpnduy/vi-factcheck-system/releases/tag/listwise-v2-three-seeds-20261112)

Git contains compact reports, metrics, histories and provenance. The ZIP also
contains all predictions, original experiment configurations and manifests,
tuning artifacts, and the exact executed source snapshot. Weights and datasets
are not included. References to local weight paths in original manifests are
historical records, not downloadable files. Original absolute paths and file
bytes are preserved for hash verification; supply your own paths when running.

From the repository's `research/` directory:

```powershell
.\.venv\Scripts\python.exe scripts\export_results.py verify --output results\listwise-best5-v2-three-seeds-20261112 --archive <downloaded-zip>
```

`EXPORT_MANIFEST.json` hashes the original archive payload and compact files.
The ZIP digest is external to avoid a self-referential checksum. Verification
recomputes classification/retrieval and three-seed aggregates from predictions,
checks all 15 x 1,447 claims, test hashes, selection hashes and 12 bootstrap
records. It does not load weights or start training.

## Provenance and reproduction

Executed source: `baa0bdc80379d26e661bd1dcf5e37b3b48ba14a8`. The study tag identifies the later
publication commit, not the training commit. `PROVENANCE.json` maps the former
`rush/vifactcheck/` location to `research/vifactcheck/`. The package/CLI stays
`vifactcheck`; all 15 executed Python files retain their exact original bytes.
Use `REPRODUCTION_REQUIREMENTS.txt` and the current research setup/protocol guide.
Published October studies remain separate at their original `rush/results/` URLs.
The recorded 132.52-hour wall time includes interruptions and repeated work.
