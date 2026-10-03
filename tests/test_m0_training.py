"""End-to-end training/checkpoint/inference contract, using a tiny local model."""
from __future__ import annotations

import json

import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from factcheck.data.prepare_m0 import prepare_dataset
from factcheck.data.vifactcheck import _write_jsonl
from factcheck.evaluation.evaluate_m0 import evaluate_m0
from factcheck.inference import predict_m0
from factcheck.m0_config import M0Config
from factcheck.models.verifier import M0Verifier
from factcheck.training import train_m0
from factcheck.training.m0_runtime import verify_prepared


class TestSegmenter:
    __test__ = False
    manifest = {"revision": "test-only", "sha256": {}}

    def __init__(self, _path):
        pass

    def __call__(self, text):
        return text

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


def prepared_data(tmp_path):
    source = tmp_path / "processed"
    for split in ("train", "validation"):
        _write_jsonl(source / f"{split}.jsonl", [
            {"id": f"{split}-{index}", "claim": f"Claim {index} is correct.",
             "context": "Claim zero is correct. Claim one is incorrect. We lack evidence.",
             "evidence": [], "label": label, "dataset": "vifactcheck", "split": split}
            for index, label in enumerate(("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"))
        ])
    prepared = tmp_path / "prepared"
    prepare_dataset(source, prepared)
    return prepared


def test_train_partial_accumulation_checkpoint_reload_and_evaluate(tmp_path, monkeypatch):
    prepared = prepared_data(tmp_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nclaim\nzero\none\nis\ncorrect\n.\n")
    tokenizer = BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    model = M0Verifier(BertForSequenceClassification(BertConfig(
        vocab_size=len(tokenizer), hidden_size=12, num_hidden_layers=1,
        num_attention_heads=3, intermediate_size=16, max_position_embeddings=260,
        num_labels=3, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )))
    monkeypatch.setattr(train_m0, "load_components", lambda _config: (model, tokenizer))
    monkeypatch.setattr(train_m0, "VnCoreNLPSegmenter", TestSegmenter)
    monkeypatch.setattr(predict_m0, "VnCoreNLPSegmenter", TestSegmenter)
    config = M0Config(data_dir=str(prepared), output_dir=str(tmp_path / "runs"),
                      warmup_epochs=1, continuation_epochs=0, batch_size=2,
                      gradient_accumulation_steps=2, cpu_threads=1, device="cpu")
    result = train_m0.train(config, 42)
    assert result["status"] == "complete"
    assert result["train_claims"] == result["validation_claims"] == 3
    run = tmp_path / "runs/seed_42"
    prediction_path = run / "reload.jsonl"
    predict_m0.predict(run / "best", prepared / "validation.inputs.jsonl", prediction_path,
                       device="cpu", batch_size=2)
    expected = [json.loads(line) for line in (run / "validation.predictions.jsonl").read_text()
                .splitlines()]
    actual = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    for original, reloaded in zip(expected, actual, strict=True):
        assert original["id"] == reloaded["id"]
        assert original["probabilities"] == pytest.approx(reloaded["probabilities"], abs=1e-7)
        assert original["sentence_ids"] == reloaded["sentence_ids"]
    report = evaluate_m0(prediction_path, prepared / "validation.targets.jsonl",
                         run / "reevaluated.json", prepared / "validation.inputs.jsonl")
    assert report["verdict"]["macro_f1"] == result["best_macro_f1"]
    with pytest.raises(FileExistsError):
        train_m0.train(config, 42)
    with pytest.raises(FileExistsError):
        predict_m0.predict(run / "best", prepared / "validation.inputs.jsonl", prediction_path)


def test_prepared_checksum_and_candidate_protocol_fail_before_training(tmp_path):
    prepared = prepared_data(tmp_path)
    verify_prepared(prepared, ("train", "validation"))
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate_k"] = 5
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="retrieval protocol"):
        verify_prepared(prepared, ("train", "validation"))
    manifest["candidate_k"] = 20
    manifest_path.write_text(json.dumps(manifest))
    with (prepared / "train.inputs.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_prepared(prepared, ("train", "validation"))


def test_config_rejects_wrong_m0_architecture():
    with pytest.raises(ValueError, match="requires"):
        M0Config(pooling="mean_logits")
    with pytest.raises(ValueError, match="requires"):
        M0Config(scope="global_corpus")
    with pytest.raises(ValueError, match="top_k"):
        M0Config(top_k=20)
