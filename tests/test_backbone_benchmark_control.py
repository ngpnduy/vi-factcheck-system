import argparse

import pytest

from vifactcheck import LABELS
from vifactcheck.benchmark_cli import add_benchmark_parser
from vifactcheck.benchmark_report import check_predictions, write_report
from vifactcheck.io import read_json, write_json, write_jsonl


@pytest.mark.parametrize("action", ["smoke", "run"])
def test_backbone_benchmark_defaults_to_m0(action):
    parser = argparse.ArgumentParser()
    add_benchmark_parser(parser.add_subparsers(dest="command", required=True))
    args = ["benchmark", action, "--backbone", "mbert", "--output", "out"]
    assert parser.parse_args(args).control == "m0"
    assert parser.parse_args(args + ["--control", "claim-article"]).control == "claim-article"


def test_report_uses_m0_as_paired_control(tmp_path):
    row = {"id": "claim-1", "article_id": "article-1", "label": 0, "candidate_ids": [1, 0]}
    predictions = {
        "M0": {"ranking": [1, 0], "label_id": 1, "probabilities": dict(zip(LABELS, [0.1, 0.8, 0.1]))},
        "M2": {"ranking": [0, 1], "label_id": 0, "probabilities": dict(zip(LABELS, [0.8, 0.1, 0.1]))},
    }
    checkpoints = {}
    for mode, prediction in predictions.items():
        prediction.update(id=row["id"], article_id=row["article_id"], gold_label_id=row["label"],
                          selected_sentences=[{"id": i} for i in prediction["ranking"]])
        prefix = tmp_path / "evaluation" / "seed-42" / f"{mode.lower()}_test"
        write_jsonl(prefix.with_name(prefix.name + "_predictions.jsonl"), [prediction])
        write_json(prefix.with_name(prefix.name + "_metrics.json"), {
            "classification": {"count": 1, "macro_f1": 1 / 3 if mode == "M2" else 0,
                               "accuracy": 1 if mode == "M2" else 0},
            "cost": {"parameters": 1, "total_input_tokens": 8, "prepared_input_seconds_per_claim": 0.1},
            "truncation": {"selected_sentence_truncated_count": 0, "verifier_context_tokens_kept": 5},
            "retrieval": {"count": 1, "mrr": 1},
        })
        checkpoints[f"42/{mode}"] = {
            "selection": {"selected_global_epoch": 1, "selected_stage": "m0" if mode == "M0" else "warmup"},
            "sha256": "test",
            "training_cost": {"training_seconds": 1, "peak_training_allocated_mib": None,
                              "peak_training_reserved_mib": None},
        }
    with pytest.raises(ValueError, match="BM25 order"):
        check_predictions([predictions["M2"]], [row], "M0")
    write_report(tmp_path, {"assets": {"backbone": "mbert"}, "control": "M0", "seeds": [42]},
                 [row], checkpoints, "test")
    report = read_json(tmp_path / "comparison.json")
    assert [item["system"] for item in report["summary"]] == ["M0", "M2"]
    assert report["summary"][1]["paired_macro_f1_delta_mean"] == pytest.approx(1 / 3)
    assert "42" in report["m2_minus_m0_paired_article_bootstrap"]
    assert "Δ Macro-F1 vs M0" in (tmp_path / "REPORT.md").read_text()
