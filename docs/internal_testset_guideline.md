# Internal test-set annotation guideline

## Scope

`test_v1` is a private, final-only evaluation set for robustness, domain shift,
and abstention analysis. Never use it for training, few-shot examples, prompt
editing, model selection, or threshold tuning.

## Source and record requirements

1. Select a public, reputable primary or authoritative source and save its URL,
   title, publication/access date, and a sufficient text excerpt.
2. Create a claim that is understandable without hidden context. For numerical
   claims, retain the number, unit, time period, and comparison basis.
3. Store the source text as `context`; store the minimal supporting or
   contradicting excerpt as gold `evidence`.
4. Add `metadata.provenance` with `source_url`, `source_title`, and
   `retrieved_at` (ISO 8601 date/time).

## Label definitions

- `SUPPORTED`: the provided evidence directly entails the full claim.
- `REFUTED`: the provided evidence directly contradicts at least one essential
  part of the claim.
- `NOT_ENOUGH_INFO`: the provided context cannot determine the claim. It is not
  a synonym for false; do not add fabricated evidence for this label.

Split compound claims before annotation when their clauses can receive different
labels. Reject claims whose date, unit, entity, or comparison target is ambiguous.

## Annotation and adjudication

Two members annotate each candidate independently. Disagreements are resolved by
a third reviewer or a recorded group decision. The final canonical record must
include this metadata:

```json
{
  "annotation": {
    "status": "adjudicated",
    "annotator_ids": ["member_1", "member_2"],
    "adjudicated_by": "member_3",
    "decision_note": "Brief reason for the final label"
  },
  "provenance": {
    "source_url": "https://example.org/source",
    "source_title": "Source title",
    "retrieved_at": "2026-09-04T00:00:00+07:00"
  }
}
```

## Freeze policy

Run the freeze command only after all records are adjudicated. It writes a
SHA-256 checksum and label counts to `test_v1.manifest.json`. A frozen release is
immutable: fixes, additional records, or label changes must become `test_v2`.
