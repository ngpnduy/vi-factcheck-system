# Migration from rush/ to research/

Only tracked code, tests, configuration, examples and scripts moved. No dataset,
cache, local environment, checkpoint, parent-workspace prototype or completed run
was moved or deleted. `rush/results/` publications retain their original bytes
and URLs. `src/factcheck/`, root `uv.lock`, application version and Git history
remain separate and unchanged.

| Previous location | Current location |
|---|---|
| `rush/vifactcheck/` | `research/vifactcheck/` (same package and exact 15 source files) |
| `rush/tests/`, examples and setup | `research/` counterparts |
| `rush/LISTWISE_V2.md` | `research/LISTWISE_V2.md` |
| Older guides/validation/provenance | `research/docs/history/` |
| `rush/results/` | Unchanged historical publications |
| `rush/scripts/*.ps1` | Forwarding setup/full/Listwise-v2 launchers |

Create `research/.venv` using the setup guide. Do not move an existing environment
or run Python from `rush/`. Existing checkpoints keep their format and model/data
paths. Use explicit asset paths or the existing `LOCATION.json` when evaluating
or resuming; changed settings/source/data require a separate experiment directory.
No full training is repeated for this migration.

`setup.ps1 -EnvironmentDirectory <new-directory>` optionally stores the new
environment on another local drive and exposes it through a `research/.venv`
junction. This laptop uses a new isolated environment on C; the prior local/shared
environments remain untouched.

The completed v2 run executed commit `baa0bdc80379d26e661bd1dcf5e37b3b48ba14a8`.
Its source bytes and original artifacts remain immutable. The study Release tag
identifies the later publication commit, not a claim that training ran after the
rename. The original source snapshot and per-file hashes are in the results ZIP.

[Publication validation evidence](docs/validation/publication-20261113.json) records
the relocated test suite, checkpoint prediction checks, artifact verification,
link/ignore/launcher checks, historical hashes and original-workflow test status.
The [final archive and link checks](docs/validation/publication-export-20261113.json)
are recorded after building the publication package.
