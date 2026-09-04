# Internal evaluation data

This directory is deliberately excluded from Git except for documentation. Keep
candidate and frozen test records private until the team approves their release.

Create adjudicated canonical records in `adjudicated/test_v1.jsonl`, then freeze
them exactly once with:

```bash
PYTHONPATH=src uv run python -m factcheck.data.internal_testset freeze
```

The command creates the ignored artifacts `test_v1.jsonl` and
`test_v1.manifest.json`. Never edit either artifact after freezing; create
`test_v2` for any correction or expansion.
