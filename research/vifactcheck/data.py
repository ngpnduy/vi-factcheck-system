from collections import Counter
from pathlib import Path
import random
import hashlib

from rank_bm25 import BM25Okapi

from .assets import ensure_assets
from .config import EXPECTED_COUNTS
from .io import read_json, read_jsonl, sha256, write_json, write_jsonl
from .text import VnCoreSegmenter, align_evidence, normalize, recover_sentences


def bm25_candidates(claim_segmented, sentences, limit=20):
    if not sentences:
        raise ValueError("No article sentences")
    corpus = [s["segmented"].casefold().split() for s in sentences]
    scores = BM25Okapi(corpus, k1=1.5, b=0.75).get_scores(claim_segmented.casefold().split())
    ids = sorted(range(len(sentences)), key=lambda i: (-float(scores[i]), i))[:limit]
    return ids, [float(scores[i]) for i in ids]


def example_id(row, split):
    return f"{split}:{row['annotation_id']}:{row['index']}"


def prepare(output="data/prepared", cache=".cache", offline=False):
    import pyarrow.parquet as pq
    output, cache = Path(output), Path(cache)
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        if all((output / f"{split}.jsonl").exists() for split in EXPECTED_COUNTS):
            return manifest
    assets = ensure_assets(cache, offline=offline)
    segmenter = VnCoreSegmenter(cache / "vncorenlp")
    split_stats, overlap_sets = {}, {}
    for split, expected in EXPECTED_COUNTS.items():
        table = pq.read_table(cache / "dataset" / f"{split}.parquet")
        if table.num_rows != expected:
            raise ValueError(f"{split} contains {table.num_rows} rows; expected {expected} for pinned revision")
        rows = table.to_pylist()
        prepared, rejected = [], []
        for start in range(0, len(rows), 128):
            part = output / ".parts" / f"{split}_{start}.json"
            if part.exists():
                cached = read_json(part)
                prepared.extend(cached["prepared"])
                rejected.extend(cached["rejected"])
                continue
            previous_prepared, previous_rejected = len(prepared), len(rejected)
            batch = rows[start:start + 128]
            valid = [r for r in batch if isinstance(r.get("Statement"), str) and normalize(r["Statement"]) and isinstance(r.get("Context"), str) and normalize(r["Context"]) and r.get("labels") in (0, 1, 2)]
            rejected.extend({"id": example_id(r, split), "reason": "empty claim/context or invalid label"} for r in batch if r not in valid)
            texts = [text for r in valid for text in (r["Statement"], r["Context"])]
            segmented = segmenter.segment_many(texts)
            for i, row in enumerate(valid):
                claim_segmented = " ".join(segmented[2 * i])
                try:
                    sentences = recover_sentences(row["Context"], segmented[2 * i + 1])
                except ValueError as error:
                    rejected.append({"id": example_id(row, split), "reason": str(error)})
                    continue
                alignment = align_evidence(row["Context"], sentences, row.get("Evidence") or "")
                candidate_ids, scores = bm25_candidates(claim_segmented, sentences)
                prepared.append({"id": example_id(row, split), "claim": row["Statement"], "claim_segmented": claim_segmented, "context": row["Context"], "label": int(row["labels"]), "gold_evidence": row.get("Evidence") or "", "sentences": sentences, "candidate_ids": candidate_ids, "bm25_scores": scores, "alignment": alignment, "article_id": row.get("Url") or hashlib.sha256(normalize(row["Context"]).encode()).hexdigest()})
            write_json(part, {"prepared": prepared[previous_prepared:], "rejected": rejected[previous_rejected:]})
            print(f"Prepared {split}: {min(start + 128, len(rows))}/{len(rows)}", flush=True)
        write_jsonl(output / f"{split}.jsonl", prepared)
        write_jsonl(output / f"{split}_rejected.jsonl", rejected)
        write_jsonl(output / f"{split}_alignment_review.jsonl", [{"id": r["id"], "label": r["label"], "status": r["alignment"]["status"], "claim": r["claim"], "evidence": r["gold_evidence"], "context": r["context"]} for r in prepared if r["alignment"]["status"] != "valid"])
        split_stats[split] = {"source_count": len(rows), "valid_count": len(prepared), "rejected_count": len(rejected), "labels": dict(Counter(r["label"] for r in prepared)), "alignment": dict(Counter(r["alignment"]["status"] for r in prepared)), "alignment_by_label": {str(label): dict(Counter(r["alignment"]["status"] for r in prepared if r["label"] == label)) for label in (0, 1, 2)}}
        overlap_sets[split] = {"claims": {normalize(r["claim"]) for r in prepared}, "articles": {r["article_id"] for r in prepared}}
    overlaps = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlaps[f"{left}-{right}"] = {key: len(overlap_sets[left][key] & overlap_sets[right][key]) for key in ("claims", "articles")}
    manifest = {"assets": assets, "splits": split_stats, "cross_split_unique_overlap": overlaps, "gold_matching": "exact NFC and whitespace-normalized snippets; no fuzzy supervision", "bm25": {"k1": 1.5, "b": 0.75, "candidate_count": 20}, "prepared_file_sha256": {split: sha256(output / f"{split}.jsonl") for split in EXPECTED_COUNTS}}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_split(path, split):
    if split not in EXPECTED_COUNTS:
        raise ValueError("split must be train, dev, or test")
    manifest = read_json(Path(path) / "manifest.json")
    file = Path(path) / f"{split}.jsonl"
    if sha256(file) != manifest["prepared_file_sha256"][split]:
        raise ValueError(f"Prepared {split} differs from its recorded checksum")
    return read_jsonl(file)


def has_evidence_supervision(row):
    if row["label"] == 2 or row["alignment"]["status"] != "valid":
        return False
    gold = set(row["alignment"]["gold_sentence_ids"])
    positives = sum(i in gold for i in row["candidate_ids"])
    return 0 < positives < len(row["candidate_ids"])


def stratified_subset(rows, per_label, seed):
    """Sample all labels without filtering on alignment or gold evidence."""
    rng, result = random.Random(seed), []
    for label in (0, 1, 2):
        pool = [r for r in rows if r["label"] == label]
        if len(pool) < per_label:
            raise ValueError(f"Not enough label {label} examples for the requested subset")
        result.extend(rng.sample(pool, per_label))
    rng.shuffle(result)
    return result


def smoke_subset(rows, per_label, seed=42):
    rng, result = random.Random(seed), []
    for label in (0, 1, 2):
        eligible = [r for r in rows if r["label"] == label and (label == 2 or has_evidence_supervision(r))]
        if len(eligible) < per_label:
            raise ValueError(f"Not enough eligible label {label} examples for balanced smoke run")
        result.extend(rng.sample(eligible, per_label))
    rng.shuffle(result)
    return result
