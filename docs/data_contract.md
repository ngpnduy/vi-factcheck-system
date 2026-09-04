# Data contract

This contract defines the common processed-data format for every dataset used by
the project. It is the interface consumed by the classifier baseline, prompting
baseline, retrieval modules, and evaluation code. Dataset-specific source files
remain unchanged under `data/raw/`.

## Storage rules

- Encoding: UTF-8.
- Exchange format: JSON Lines (`.jsonl`), one record per line.
- Required public splits: `train`, `validation`, and `test`.
- Labels are uppercase strings: `SUPPORTED`, `REFUTED`, `NOT_ENOUGH_INFO`.
- `NOT_ENOUGH_INFO` is a dataset label. It is not the later system action
  `ABSTAIN`; an abstention decision is stored with predictions, never as a
  ground-truth label.
- Each processed dataset directory includes a `manifest.json` with source
  information, record counts, and SHA-256 checksums.

## Canonical record (schema version 1.0.0)

```json
{
  "schema_version": "1.0.0",
  "id": "vifactcheck:train:123",
  "claim": "Một phát biểu tiếng Việt cần được kiểm chứng.",
  "context": "Toàn văn ngữ cảnh nguồn, nếu dataset cung cấp.",
  "evidence": [
    {
      "evidence_id": "vifactcheck:train:123:e1",
      "text": "Đoạn bằng chứng vàng do dataset gán nhãn.",
      "source_title": "Tên báo hoặc nguồn",
      "source_url": "https://example.org/article",
      "position": null
    }
  ],
  "label": "SUPPORTED",
  "dataset": "vifactcheck",
  "split": "train",
  "metadata": {
    "topic": "Health",
    "source_split": "train",
    "annotation_id": 123,
    "source_index": 456
  }
}
```

### Field definitions

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | Yes | Version of this contract, currently `1.0.0`. |
| `id` | Yes | Stable, unique ID. Never derive it from row order alone. |
| `claim` | Yes | Claim to be verified; it must be non-empty after trimming whitespace. |
| `context` | No | Full source context when provided by the originating dataset. Use `null` when unavailable. |
| `evidence` | Yes | List of gold evidence snippets. It may be empty only when the originating dataset does not supply a snippet. |
| `evidence[].evidence_id` | Yes | Stable ID within the record. |
| `evidence[].text` | Yes | Non-empty gold evidence snippet. |
| `evidence[].source_title` | No | Publisher/title of the source. |
| `evidence[].source_url` | No | Original source URL, if licensed and supplied. |
| `evidence[].position` | No | Sentence/paragraph position when known. |
| `label` | Yes | `SUPPORTED`, `REFUTED`, or `NOT_ENOUGH_INFO`. |
| `dataset` | Yes | Lowercase identifier, for example `vifactcheck` or `vinumfcr`. |
| `split` | Yes | `train`, `validation`, or `test`. |
| `metadata` | Yes | Dataset-specific scalar metadata. Do not put secrets or model outputs here. |

`evidence` contains *gold annotated evidence*, not evidence retrieved by a model.
Retrieved candidates and predictions belong under `outputs/`, so evaluation
remains reproducible and the gold data is never overwritten.

## ViFactCheck mapping

| ViFactCheck field | Canonical field |
| --- | --- |
| `Statement` | `claim` |
| `Context` | `context` |
| `Evidence` | `evidence[0].text` |
| `Topic` | `metadata.topic` |
| `Author` | `evidence[0].source_title` |
| `Url` | `evidence[0].source_url` |
| `labels = 0` | `label = SUPPORTED` |
| `labels = 1` | `label = REFUTED` |
| `labels = 2` | `label = NOT_ENOUGH_INFO` |
| `annotation_id` | `metadata.annotation_id` (may repeat across claims) |
| `index` | Stable ID component and `metadata.source_index` |

The official source split `dev` is published as canonical `validation`. The
official `train` and `test` splits keep their names; the pipeline never
reshuffles them.

## Validation gates

Every processed release must pass these checks before it can be used by a
baseline:

1. Each JSON line conforms to the canonical schema.
2. IDs are unique across all splits.
3. Claims, labels, and non-empty evidence texts are valid.
4. The output split matches the file name.
5. The expected official source splits are present.
6. `manifest.json` records counts and SHA-256 checksums of each split.

Changes that alter this contract require a new `schema_version` and a new
processed-data release; do not silently rewrite a published split.
