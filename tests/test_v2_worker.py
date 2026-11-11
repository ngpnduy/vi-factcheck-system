"""Exercise the entire staged runner with tiny artifacts, without GPU training."""
import argparse
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from vifactcheck.config import ALL_MODELS, DATASET_REVISION, MODEL_REVISION, TrainConfig
from vifactcheck.io import read_json, sha256, write_json, write_jsonl
from vifactcheck.metrics import classification_metrics


def test_three_seed_worker_tunes_fresh_reuses_winners_and_resumes_without_retraining(tmp_path, monkeypatch):
    import vifactcheck.full as full
    data, model = tmp_path / "data", tmp_path / "model"
    model.mkdir()
    (model / "pytorch_model.bin").write_bytes(b"pinned fixture")
    splits = {}
    for split, count in (("train", 5062), ("dev", 723), ("test", 1447)):
        splits[split] = [{"id": f"{split}:{i}", "article_id": str(i // 3), "claim": f"claim {i}", "context": "context",
                          "label": i % 3, "candidate_ids": [0, 1], "alignment": {"status": "valid", "gold_sentence_ids": [0]}}
                         for i in range(count)]
        write_jsonl(data / f"{split}.jsonl", [])
    assets = {"dataset_revision": DATASET_REVISION, "model_revision": MODEL_REVISION,
              "files": [{"path": "phobert/pytorch_model.bin", "sha256": sha256(model / "pytorch_model.bin")}]}
    manifest = {"assets": assets, "splits": {s: {"valid_count": len(rows)} for s, rows in splits.items()},
                "prepared_file_sha256": {s: sha256(data / f"{s}.jsonl") for s in splits}}
    write_json(data / "manifest.json", manifest)
    jobs, evaluations = [], []
    def load_split(directory, split):
        if split == "test":
            assert len(jobs) == 13, "Test must not be read before all training/configuration selection is final"
        return splits[split]
    monkeypatch.setattr(full, "load_split", load_split)
    monkeypatch.setattr(full, "keep_awake", nullcontext)
    monkeypatch.setattr(full, "require_full_start_space", lambda p: 40 * 1024**3)
    monkeypatch.setattr(full, "require_space", lambda *a: 40 * 1024**3)
    monkeypatch.setattr(full, "profile_long_claims", lambda *a: {"status": "passed"})
    monkeypatch.setattr(full.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(full.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(full, "release_gpu", lambda: None)
    monkeypatch.setattr("vifactcheck.assets.ensure_assets", lambda *a, **k: assets)
    classification = classification_metrics([0, 1, 2], [0, 1, 2])
    retrieval = {"count": 2, "alignment_coverage": 1., **{k: 1. for k in (
        "hit@5", "hit@20", "sentence_recall@5", "sentence_recall@20", "mrr", "complete_gold@5", "complete_gold@20")}}
    metric = {"classification": classification, "retrieval": retrieval,
              "cost": {"prepared_input_seconds_per_claim": .1}, "error_stages": {}, "truncation": {}}
    def prediction(row):
        return {"id": row["id"], "gold_label_id": row["label"], "article_id": row["article_id"],
                "label_id": row["label"], "probabilities": {"SUPPORTED": .2, "REFUTED": .3, "NOT_ENOUGH_INFO": .5}}
    def job(cfg, directory, modes, progress, train, initialization_from=None):
        directory = Path(directory)
        if (directory / "training_complete.json").exists():
            return full.verify_completed_training(directory)
        jobs.append((cfg, tuple(modes), initialization_from))
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "experiment.json", {"protocol_version": 2, "config": cfg.to_dict(), "models": list(modes),
                                                   "subset_ids": {s: [r["id"] for r in splits[s]] for s in ("train", "dev")}})
        parents = []
        if "M1" in modes:
            for name in ("reranker_best.pt", "warmup_best.pt", "warmup_end.pt"):
                (directory / name).write_bytes(f"{cfg.seed}:{cfg.learning_rate}:{name}".encode())
            for name, stage in (("reranker", "reranker"), ("warmup", "warmup")):
                write_json(directory / f"{name}_history.json", [{"stage": stage, "training_seconds": 10.}])
            parents = ["reranker_best.pt", "warmup_best.pt", "warmup_end.pt", "reranker_history.json", "warmup_history.json"]
        result = {}
        for mode in modes:
            score = .8 if cfg.learning_rate == 2e-5 else .7
            if mode == "M2":
                score += {.1: .01, .5: .05, 1.: .03}[cfg.evidence_lambda]
            selected = {"protocol_version": 2, "selected_global_epoch": 4 if mode in ("M1", "M2", "A2") else 5,
                        "selected_stage": "continuation" if mode in ("M1", "M2", "A2") else mode.lower(),
                        "selected_validation_macro_f1": score}
            checks = {"verifier_initial_sha256": f"initial:{cfg.seed}:{cfg.learning_rate}",
                      "reranker_unchanged": mode in ("M1", "A1"), "reranker_final_sha256": "same frozen selector"}
            if mode in ("M1", "M2", "A2"):
                checks["shared_warmup_checkpoint_sha256"] = sha256(Path(initialization_from or directory) / "warmup_end.pt")
            current_metric = {**metric, "classification": {**classification, "macro_f1": score}}
            name = mode.lower()
            (directory / f"{name}_best.pt").write_bytes(mode.encode())
            write_json(directory / f"{name}_dev_metrics.json", current_metric)
            write_jsonl(directory / f"{name}_dev_predictions.jsonl", [prediction(splits["dev"][0])])
            write_json(directory / f"{name}_checks.json", checks)
            write_json(directory / f"{name}_selection.json", selected)
            write_json(directory / f"{name}_history.json", [{"stage": "continuation" if mode in ("M1", "M2", "A2") else name, "training_seconds": 10.}])
            result[mode] = {"metrics": current_metric, "checks": checks, "selection": selected,
                            "checkpoint": str(directory / f"{name}_best.pt"),
                            "training_cost": {"training_seconds": 10., "peak_training_allocated_mib": 100., "peak_training_reserved_mib": 120.}}
        write_json(directory / "summary.json", result)
        full.seal_training(directory, result, parents)
        return result
    monkeypatch.setattr(full, "training_job", job)
    monkeypatch.setattr(full, "system_from_checkpoint", lambda p, device: SimpleNamespace(prediction=prediction, mode=Path(p).name.split("_")[0]))
    monkeypatch.setattr("vifactcheck.cli.predict_record", lambda system, claim, context, segmenter: prediction(splits["dev"][0]))
    def evaluate(system, rows, prefix, progress):
        evaluations.append(system.mode)
        result = {**metric, "classification": classification_metrics([r["label"] for r in rows], [r["label"] for r in rows])}
        write_json(str(prefix) + "_metrics.json", result)
        write_jsonl(str(prefix) + "_predictions.jsonl", [prediction(r) for r in rows])
        return result
    monkeypatch.setattr(full, "evaluate", evaluate)
    args = argparse.Namespace(output=str(tmp_path / "full"), status_dir=str(tmp_path / "status"), data=str(data),
                              model_path=str(model), segmenter_path=str(tmp_path / "vncorenlp"), include_ablations=True,
                              reranker_architecture="listwise")
    completed = full.run_full(args)
    assert completed["protocol_version"] == 2 and completed["models"] == list(ALL_MODELS)
    assert len(jobs) == 13 and len(evaluations) == 15
    selected = read_json(Path(args.output) / "selected_hyperparameters.json")
    assert selected["learning_rate"] == 2e-5 and selected["evidence_lambda"] == .5
    assert all(cfg.reranker_architecture == "listwise" and cfg.protocol_version == 2 for cfg, _, _ in jobs)
    assert all(cfg.learning_rate == 2e-5 and cfg.evidence_lambda == .5 for cfg, _, _ in jobs if cfg.seed in (13, 2026))
    assert [cfg.evidence_lambda for cfg, modes, _ in jobs if cfg.seed == 42 and modes == ("M2",)] == [.1, .5, 1.]
    assert len(read_json(Path(args.output) / "comparison.json")["per_seed"]) == 15
    assert read_json(Path(args.status_dir) / "completion.json") == completed
    full.run_full(args)
    assert len(jobs) == 13 and len(evaluations) == 15, "Completed resume must not retrain or reevaluate"
