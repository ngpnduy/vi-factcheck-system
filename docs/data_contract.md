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
| `dataset` | Yes | Lowercase identifier, for example `vifactcheck`. |
| `split` | Yes | `train`, `validation`, or `test`. |
| `metadata` | Yes | Dataset-specific scalar metadata. Do not put secrets or model outputs here. |

`evidence` contains *gold annotated evidence*, not evidence retrieved by a model.
Retrieved candidates and predictions belong under `outputs/`, so evaluation
remains reproducible and the gold data is never overwritten.

## Fixed-RAG retrieval corpus

The corpus builder reads canonical processed data and writes a separate,
label-free retrieval index under `data/interim/vifactcheck/rag_corpus/`:

```text
corpus_chunks.jsonl
manifest.json
```

Each corpus line contains a deterministic chunk of `context` plus source
metadata only:

```json
{
  "schema_version": "1.0.0",
  "chunk_id": "vifactcheck:document:6e1c1b54d4a5fa9a:chunk:0000",
  "document_id": "vifactcheck:document:6e1c1b54d4a5fa9a",
  "dataset": "vifactcheck",
  "source_split": "train",
  "chunk_index": 0,
  "word_start": 0,
  "word_end": 200,
  "text": "Một đoạn context để BM25 truy xuất.",
  "source_url": "https://example.org/article",
  "topic": "Health"
}
```

The corpus must never contain `label` or any gold `evidence[].text`. In the
initial closed-corpus protocol, documents from all official splits may be
searchable, but test labels and gold evidence remain unavailable to both the
retriever and verifier. The corpus manifest records source checksums and its
own SHA-256 checksum to make that setup reproducible.

ViFactCheck can contain several claims drawn from the same article. The corpus
therefore deduplicates documents by source URL (or by context bytes if no URL is
available), rather than indexing repeated copies of the same article. Its
`document_id` is a stable SHA-256-derived source identifier; it is intentionally
different from the processed record ID.

### Fixed BM25S run and evaluation

The first retriever is a deterministic BM25S baseline. It uses the same Unicode
lowercasing/tokenization procedure for claims and corpus chunks, preserves
Vietnamese diacritics, and returns a fixed top-`k` list for each claim. The
sparse index is cached under `data/interim/` and is rebuilt automatically when
the corpus checksum changes. A retrieval run writes its results under `outputs/`,
never into the corpus or processed dataset.

Gold evidence is used only by the retrieval evaluator. A retrieved chunk counts
as relevant when it belongs to the query record's source document and contains
one of that record's gold evidence strings after normalization. The evaluator
reports source-document `Recall@k`/`MRR` for every query, plus evidence-level
`Recall@k` (at least one relevant chunk appears), chunk-level recall, `MRR`, and
`Precision@k` for records whose evidence can be aligned. Unalignable evidence is
listed in a separate diagnostic JSONL instead of being silently counted as a
miss.

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
