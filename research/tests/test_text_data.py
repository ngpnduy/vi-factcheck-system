import unicodedata
from pathlib import Path
import shutil
import pytest

from vifactcheck.config import TrainConfig
from vifactcheck.data import bm25_candidates, has_evidence_supervision, smoke_subset
from vifactcheck.text import align_evidence, normalize, recover_sentences


def sentences():
    raw = "Cầu chưa mở. Có 20 người."
    return raw, recover_sentences(raw, ["Cầu chưa mở .", "Có 20 người ."])


def test_source_offsets_and_segmentation_preserve_numbers_dates_negation():
    raw = "TP. Hồ Chí Minh chưa mở cầu. Giá là 3,5 tỷ. Ngày 20/10/2024."
    recovered = recover_sentences(raw, ["TP. Hồ_Chí_Minh chưa mở cầu .", "Giá là 3,5 tỷ .", "Ngày 20/10/2024 ."])
    assert len(recovered) == 3
    assert all(raw[s["start"]:s["end"]] == s["text"] for s in recovered)
    assert "3,5" in recovered[1]["text"] and "20/10/2024" in recovered[2]["text"]


def test_exact_and_multisentence_gold():
    raw, s = sentences()
    assert align_evidence(raw, s, raw) == {"status": "valid", "gold_sentence_ids": [0, 1]}
    assert align_evidence(raw, s, "Cầu chưa mở.\nCó 20 người.")["gold_sentence_ids"] == [0, 1]
    assert align_evidence(raw, s, "Có 2 người.")["status"] == "unmatched"
    assert align_evidence(raw, s, "Cầu đã mở.")["status"] == "unmatched"


def test_ambiguous_and_missing_gold():
    raw = "Cầu mở. Cầu mở."
    s = recover_sentences(raw, ["Cầu mở .", "Cầu mở ."])
    assert align_evidence(raw, s, "Cầu mở.")["status"] == "ambiguous"
    assert align_evidence(raw, s, "")["status"] == "missing"


def test_unicode_and_whitespace_alignment():
    raw = unicodedata.normalize("NFD", "Cầu  chưa mở.")
    s = recover_sentences(raw, ["Cầu chưa mở ."])
    assert align_evidence(raw, s, "Cầu chưa mở.")["status"] == "valid"
    assert normalize(raw) == "Cầu chưa mở."


def test_short_article_and_ties():
    raw, s = sentences()
    ids, _ = bm25_candidates("không_có_từ_này", s)
    assert ids == [0, 1]


@pytest.mark.parametrize("label,status,gold,expected", [(0, "valid", [0], True), (1, "valid", [0], True), (2, "valid", [0], False), (0, "unmatched", [], False), (0, "valid", [7], False), (0, "valid", [0, 1], False)])
def test_supervision_gating(label, status, gold, expected):
    row = {"label": label, "candidate_ids": [0, 1], "alignment": {"status": status, "gold_sentence_ids": gold}}
    assert has_evidence_supervision(row) is expected


def test_empty_context_rejected():
    with pytest.raises(ValueError):
        recover_sentences("", [])
    with pytest.raises(ValueError):
        bm25_candidates("claim", [])


def test_invalid_training_limits():
    with pytest.raises(ValueError):
        TrainConfig(max_length=512)


@pytest.mark.skipif(not Path(".cache/vncorenlp/VnCoreNLP-1.2.jar").exists() or not shutil.which("java"), reason="Cached VnCoreNLP and Java required; no downloads in tests")
def test_real_vncore_batch_boundaries():
    from vifactcheck.text import VnCoreSegmenter
    texts = ["Ông làm việc cùng Lan", "TP. Hồ Chí Minh có 3,5 triệu người. Ngày 20/10/2024 trời mưa.", "Bà Lan chưa đi."]
    batch = VnCoreSegmenter().segment_many(texts)
    assert len(batch) == 3
    assert len(batch[1]) == 2
    for text, segmented in zip(texts, batch):
        recovered = recover_sentences(text, segmented)
        assert all(text[s["start"]:s["end"]] == s["text"] for s in recovered)
