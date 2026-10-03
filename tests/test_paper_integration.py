"""Keep the paper reproduction usable alongside M0 and corpus retrieval."""

import importlib
import json
import math
from types import SimpleNamespace

import nltk
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer

from factcheck.classification.llm.inference import _parse_label
from factcheck.classification.llm.prompt import build_inference_prompt
from factcheck.classification.llm.train_unsloth import _records_to_hf_dataset
from factcheck.classification.plm.dataset import FactCheckDataset
from factcheck.classification.plm.inference import run_inference
from factcheck.classification.plm.model import PLMClassifier
from factcheck.classification.plm.train import train_one_epoch
from factcheck.data.schemas import DatasetRecord, VerdictLabel
from factcheck.evaluation.evaluator import evaluate
from factcheck.retrieval import eval_bm25, eval_sbert
from factcheck.retrieval.bm25 import BM25Okapi, evidence_top_n


@pytest.fixture
def paper_data(tmp_path):
    records = [
        DatasetRecord(
            id=f"vifactcheck:validation:{index}", dataset="vifactcheck", split="validation",
            claim="alpha", context="alpha one. beta two. gamma three.", label=label,
            evidence=[{"evidence_id": f"e{index}", "text": "alpha one."}]
            if label != VerdictLabel.NOT_ENOUGH_INFO else [],
        )
        for index, label in enumerate(VerdictLabel)
    ]
    path = tmp_path / "validation.jsonl"
    path.write_text("".join(record.model_dump_json() + "\n" for record in records))
    return path, records


def test_paper_bm25_preserves_okapi_scores_and_scaled_ranking():
    sentences = ["alpha one.", "beta two.", "gamma three."]
    model = BM25Okapi([sentence.split() for sentence in sentences])
    expected = [math.log(2.5) - math.log(1.5), 0.0, 0.0]
    assert model.get_scores(["alpha"]) == pytest.approx(expected)
    assert model.get_batch_scores(["alpha"], [2, 0]) == pytest.approx([0, expected[0]])
    docs, scores = evidence_top_n(sentences, "alpha", top_n=1)
    assert docs == [sentences[0]]
    assert scores == [1.0]


def test_plm_train_reload_predict_and_evaluate_canonical_data(paper_data, tmp_path):
    data_path, records = paper_data
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\none\nbeta\ntwo\ngamma\nthree\n.\n")
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    dataset = FactCheckDataset(data_path, tokenizer, max_length=16)
    assert [int(dataset[index]["label"]) for index in range(3)] == [0, 1, 2]
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=12, num_hidden_layers=1,
        num_attention_heads=3, intermediate_size=16, max_position_embeddings=32,
    )
    model = PLMClassifier(BertModel(config))
    loader = DataLoader(dataset, batch_size=2)
    loss = train_one_epoch(
        model, loader, torch.nn.CrossEntropyLoss(),
        torch.optim.AdamW(model.parameters(), lr=1e-4), torch.device("cpu"),
    )
    assert math.isfinite(loss)
    expected = run_inference(model, loader, dataset.ids, torch.device("cpu"))
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = PLMClassifier(BertModel(config))
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = run_inference(reloaded, loader, dataset.ids, torch.device("cpu"))
    assert actual == expected
    assert [row["id"] for row in actual] == [record.id for record in records]
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in actual))
    result = evaluate(prediction_path, data_path)
    assert result["n_samples"] == 3
    assert 0 <= result["macro_f1"] <= 1


def test_llm_data_and_prompts_work_without_unsloth(paper_data):
    _, records = paper_data
    dataset = _records_to_hf_dataset(records, "gold_evidence")
    assert list(dataset["output"]) == ["0", "1", "2"]
    prompt = build_inference_prompt(records[0], "full_context")
    assert records[0].context in prompt
    assert prompt.endswith("### Response:\n")
    assert [_parse_label(label) for label in dataset["output"]] == list(VerdictLabel)


def test_retrieval_imports_do_not_download_nltk_resources(monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("Import must not download tokenizer resources")

    monkeypatch.setattr(nltk, "download", unexpected_download)
    importlib.reload(eval_bm25)
    importlib.reload(eval_sbert)


def test_paper_hybrid_evaluation_uses_actual_constructor(paper_data, monkeypatch, capsys):
    # A deterministic two-vector encoder exercises the actual BM25 + SBERT code
    # without downloading a transformer or NLTK tokenizer data.
    def tokenize(sentences, **kwargs):
        ids = torch.tensor([[int("alpha" in sentence)] for sentence in sentences])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(input_ids, attention_mask):
        return (torch.nn.functional.one_hot(input_ids, num_classes=2).float(),)

    sbert = SimpleNamespace(device=torch.device("cpu"), tokenizer=tokenize, model=encode)
    monkeypatch.setattr(eval_sbert, "SBERTRetriever", lambda: sbert)
    monkeypatch.setattr(nltk.tokenize, "sent_tokenize", lambda text: text.split(". "))
    eval_sbert.evaluate(paper_data[0], method="hybrid", top_k=1, bm25_candidates=3)
    output = capsys.readouterr().out
    assert "Evaluated 2 samples." in output
    assert "ACC@1 : 1.0000" in output


def test_paper_bm25_handles_no_evaluable_records(tmp_path, capsys):
    data_path = tmp_path / "empty.jsonl"
    data_path.write_text("")
    eval_bm25.evaluate_retrieval(data_path)
    assert "No records with gold evidence" in capsys.readouterr().out
