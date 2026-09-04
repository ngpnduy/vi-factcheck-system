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

## Dataset release record

For every experiment release, record the Git tag, processed dataset manifest
checksums, internal-test release checksum (if authorized), configuration file,
and model checkpoint identifier in `outputs/metrics/` or the experiment report.
