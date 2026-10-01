"""M0 data boundary and retrieval tests; no model downloads are required."""

import json
from pathlib import Path

import pytest

from factcheck.data.evidence_alignment import align_evidence
from factcheck.data.prepare_m0 import prepare_dataset
from factcheck.data.sentences import split_sentences
from factcheck.data.vifactcheck import _read_jsonl, _sha256, _write_jsonl
from factcheck.retrieval.context_bm25 import ContextBM25Retriever


def test_sentences_preserve_offsets_decimal_dates_abbreviations_and_quotes():
    context = '  TS. An ở TP.HCM nhận 3.14 điểm ngày 01.10.2026. "Không đúng!"\nCâu cuối.  '
    sentences = split_sentences(context)
    assert [sentence.text for sentence in sentences] == [
        'TS. An ở TP.HCM nhận 3.14 điểm ngày 01.10.2026.', '"Không đúng!"', "Câu cuối."
    ]
    for sentence in sentences:
        assert context[sentence.start:sentence.end] == sentence.text
    assert split_sentences(context) == sentences
    assert len({sentence.context_id for sentence in sentences}) == 1
    assert [sentence.index for sentence in sentences] == [0, 1, 2]


def test_alignment_is_whitespace_only_unique_and_supports_multiple_sentences():
    context = "Hà Nội có 3.14 triệu người.\n\nKhông phải 5 triệu người."
    sentences = split_sentences(context)
    aligned = align_evidence(
        context, sentences, ["Hà  Nội có 3.14 triệu người. Không phải 5 triệu người."]
    )
    assert aligned["alignment_status"] == "aligned"
    assert aligned["gold_sentence_ids"] == [sentence.sentence_id for sentence in sentences]
    assert align_evidence(context, sentences, ["3,14 triệu"])['alignment_status'] == "unmatched"
    assert align_evidence(context, sentences, ["hà Nội"])['alignment_status'] == "unmatched"
    repeated = "Mưa lớn hôm nay. Mưa lớn hôm nay."
    result = align_evidence(repeated, split_sentences(repeated), ["Mưa lớn hôm nay."])
    assert result["alignment_status"] == "ambiguous"
    assert result["gold_sentence_ids"] == []


def test_punctuation_fragment_is_attached_without_changing_raw_text():
    context = 'Bản chất tôi là như thế! ". Gần đây có tranh cãi.'
    sentences = split_sentences(context)
    assert len(sentences) == 2
    assert sentences[0].text == 'Bản chất tôi là như thế! ".'
    assert all(context[sentence.start:sentence.end] == sentence.text for sentence in sentences)


def test_alignment_partial_is_not_eligible_for_main_retrieval_metric():
    context = "Hà Nội ở Việt Nam. Huế ở Việt Nam."
    result = align_evidence(context, split_sentences(context), ["Hà Nội", "Paris"])
    assert result["alignment_status"] == "partial"
    assert len(result["gold_sentence_ids"]) == 1


def test_context_scope_short_context_and_stable_ties():
    sentences = split_sentences("Mưa lớn hôm nay. Nắng đẹp hôm qua. Trời lạnh ngày mai.")
    retriever = ContextBM25Retriever.build(sentences)
    ranked = retriever.retrieve("từkhôngtồntại", top_k=2)
    assert [candidate["index"] for candidate in ranked] == [0, 1]
    assert [candidate["score"] for candidate in ranked] == [0.0, 0.0]
    ranked = retriever.retrieve("Trời lạnh", top_k=20)
    assert ranked[0]["index"] == 2
    assert len(ranked) == 3
    assert all(candidate["context_id"] == sentences[0].context_id for candidate in ranked)
    others = split_sentences("Một bài báo hoàn toàn khác.")
    with pytest.raises(ValueError, match="exactly one context"):
        ContextBM25Retriever.build(sentences + others)
    with pytest.raises(ValueError, match="Claim has no indexable tokens"):
        retriever.retrieve("?!")
    with pytest.raises(ValueError, match="no indexable tokens"):
        ContextBM25Retriever.build(split_sentences("!!!"))


def _row(record_id: str, split: str = "train") -> dict:
    return {
        "id": record_id, "claim": "Hà Nội ở Việt Nam.",
        "context": "Hà Nội ở Việt Nam. Paris ở Pháp.", "label": "SUPPORTED",
        "dataset": "vifactcheck", "split": split,
        "evidence": [{"evidence_id": "e1", "text": "Hà Nội ở Việt Nam."}],
    }


def _source(directory: Path, rows: dict[str, list[dict]]) -> None:
    files = {}
    for split, records in rows.items():
        path = directory / f"{split}.jsonl"
        _write_jsonl(path, records)
        files[path.name] = {"records": len(records), "sha256": _sha256(path)}
    (directory / "manifest.json").write_text(json.dumps({"files": files}))


def test_prepare_has_no_gold_influence_conserves_rows_and_audits(tmp_path):
    source = tmp_path / "source"
    train = [_row("train_1"), _row("train_2")]
    validation = [_row("val_1", "validation")]
    _source(source, {"train": train, "validation": validation})
    first = tmp_path / "first"
    result = prepare_dataset(source, first)
    original_inputs = (first / "train.inputs.jsonl").read_bytes()
    assert result["audit"]["splits"]["train"]["records"] == 2
    assert result["audit"]["context_overlap_unique_counts"] == {"train__validation": 1}
    assert result["audit"]["pilot"]["contexts"] == 1
    assert not (first / "test.inputs.jsonl").exists()
    assert len(_read_jsonl(first / "train.targets.jsonl")) == 2
    for row in _read_jsonl(first / "train.inputs.jsonl"):
        assert not {"label", "gold_sentence_ids", "evidence"} & row.keys()
    manifest = result["manifest"]
    for filename, metadata in manifest["outputs"].items():
        assert _sha256(first / filename) == metadata["sha256"]
    train[0]["label"] = "REFUTED"
    train[0]["evidence"] = [{"evidence_id": "new", "text": "Paris ở Pháp."}]
    train[1]["label"] = "NOT_ENOUGH_INFO"
    _source(source, {"train": train, "validation": validation})
    second = tmp_path / "second"
    prepare_dataset(source, second)
    assert (second / "train.inputs.jsonl").read_bytes() == original_inputs
    targets = _read_jsonl(second / "train.targets.jsonl")
    assert targets[1]["alignment_status"] == "not_applicable"
    assert targets[1]["gold_sentence_ids"] == []


def test_prepare_rejects_checksum_mismatch_and_wrong_split(tmp_path):
    source = tmp_path / "source"
    _source(source, {"train": [_row("train_1")]})
    with (source / "train.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum/count mismatch"):
        prepare_dataset(source, tmp_path / "out", splits=["train"])
    _source(source, {"train": [_row("train_1", "validation")]})
    with pytest.raises(ValueError, match="split is not train"):
        prepare_dataset(source, tmp_path / "out", splits=["train"])
    assert not (tmp_path / "out").exists()
