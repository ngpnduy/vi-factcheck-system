# Release process

`VERSION` is the authoritative project version. `scripts/version.sh` updates the
matching `project.version` in `pyproject.toml` so package metadata remains in sync.

## Normal release

1. Ensure the intended code, lockfile, and dataset manifests have been reviewed.
2. Run `bash scripts/version.sh bump patch` (or `minor` / `major`).
3. Review and commit the resulting `VERSION` and `pyproject.toml` changes.
4. With a clean working tree, run `bash scripts/version.sh tag`.
5. Push the tag explicitly: `git push origin vX.Y.Z`.

The script refuses to tag a dirty worktree or duplicate tag. It never pushes to a
remote itself. Do not use `release` as a one-command publication action: it only
prepares the bump and still requires review and commit before tagging.

## Research study publications

Study tags such as `listwise-v2-three-seeds-20261112` identify a verified research
publication, not an application version release. Keep `VERSION`, the original
application metadata and `uv.lock` unchanged. Commit compact reports and provenance
under `research/results/`; upload the full prediction/source archive and external
SHA-256 file to the GitHub Release. Keep existing `rush/results/` URLs intact.
The publication records the original executed source commit separately from the
later publication tag. Run `research/scripts/export_results.py verify` before
publication and verify the downloaded Release asset afterward.

## Dataset release record

For every experiment release, record the Git tag, processed dataset manifest
checksums, internal-test release checksum (if authorized), configuration file,
and model checkpoint identifier in `outputs/metrics/` or the experiment report.
