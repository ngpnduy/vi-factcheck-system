"""Summarize nine finalized checkpoints without pooling repeated test claims."""
import csv
import io
from pathlib import Path

import numpy as np

from . import LABELS
from .config import EXPECTED_COUNTS
from .io import read_json, read_jsonl, write_json
from .reporting import paired_article_bootstrap

MODELS = ("M0", "M1", "M2")
SEEDS = (42, 13, 2026)


def mean_std(values):
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("Full comparison requires three finite seed scores")
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}


def write_full_report(directory):
    directory = Path(directory)
    plan = read_json(directory / "full_plan.json")
    selected = read_json(directory / "selected_hyperparameters.json")
    rows, metrics, predictions, bootstrap = [], {}, {}, {}
    for seed in SEEDS:
        location = directory / "seeds" / f"seed{seed}"
        summary = read_json(location / "summary.json")
        metadata = read_json(location / "experiment.json")
        if len(metadata["subset_ids"]["train"]) != EXPECTED_COUNTS["train"] or metadata["config"].get("train_per_label") is not None or metadata["config"]["smoke"]:
            raise ValueError("Full report cannot include a smoke or subset experiment")
        if metadata["config"]["seed"] != seed:
            raise ValueError("Seed metadata does not match its directory")
        metrics[seed], predictions[seed] = {}, {}
        for mode in MODELS:
            metric = read_json(location / f"{mode.lower()}_test_metrics.json")
            prediction = read_jsonl(location / f"{mode.lower()}_test_predictions.jsonl")
            if metric["classification"]["count"] != EXPECTED_COUNTS["test"] or len(prediction) != EXPECTED_COUNTS["test"]:
                raise ValueError("All nine models need complete official test predictions")
            if len({p["id"] for p in prediction}) != len(prediction):
                raise ValueError("Test predictions contain duplicate claim IDs")
            metrics[seed][mode], predictions[seed][mode] = metric, prediction
            c, r = metric["classification"], metric["retrieval"]
            rows.append({"model": mode, "seed": seed, "train_count": EXPECTED_COUNTS["train"], "test_count": c["count"],
                         "dev_macro_f1": summary[mode]["metrics"]["classification"]["macro_f1"],
                         "test_macro_f1": c["macro_f1"], "accuracy": c["accuracy"],
                         **{f"{label.lower()}_f1": c["per_label"][label]["f1"] for label in LABELS},
                         **{key: r[key] for key in ("count", "alignment_coverage", "hit@5", "hit@20", "sentence_recall@5", "sentence_recall@20", "mrr", "complete_gold@5", "complete_gold@20")},
                         "inference_seconds_per_claim": metric["cost"]["prepared_input_seconds_per_claim"]})
        for a, b in (("M1", "M0"), ("M2", "M1")):
            bootstrap[f"seed{seed}:{a}-{b}"] = paired_article_bootstrap(predictions[seed][a], predictions[seed][b])
    # Check pairing across seeds too; each seed evaluates the same 1,447 claims.
    baseline = {p["id"]: (p["gold_label_id"], p["article_id"]) for p in predictions[42]["M0"]}
    for seed in SEEDS:
        for mode in MODELS:
            if {p["id"]: (p["gold_label_id"], p["article_id"]) for p in predictions[seed][mode]} != baseline:
                raise ValueError("Seeds/models evaluated different test claims or labels")
    aggregate = {}
    score_keys = ("dev_macro_f1", "test_macro_f1", "accuracy", "supported_f1", "refuted_f1", "not_enough_info_f1", "hit@5", "hit@20", "sentence_recall@5", "sentence_recall@20", "mrr", "complete_gold@5", "complete_gold@20", "inference_seconds_per_claim")
    for mode in MODELS:
        model_rows = [r for r in rows if r["model"] == mode]
        aggregate[mode] = {key: mean_std([r[key] for r in model_rows]) for key in score_keys}
    differences = {}
    for a, b in (("M1", "M0"), ("M2", "M1")):
        by_seed = {str(seed): metrics[seed][a]["classification"]["macro_f1"] - metrics[seed][b]["classification"]["macro_f1"] for seed in SEEDS}
        differences[f"{a}-{b}"] = {"per_seed_macro_f1": by_seed, **mean_std(list(by_seed.values()))}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (directory / "results.csv").write_bytes(buffer.getvalue().encode("utf-8-sig"))
    comparison = {"study": "full-data three-seed experiment", "seeds": list(SEEDS), "selected_hyperparameters": selected,
                  "per_seed": rows, "models": aggregate, "paired_seed_macro_f1_differences": differences,
                  "test_metrics": metrics, "bootstrap_note": "Per-seed article bootstrap intervals are conditional on each fitted model. Three-seed sample SD measures observed training variation; test claims are not independent repeated samples across seeds."}
    write_json(directory / "comparison.json", comparison)
    write_json(directory / "paired_bootstrap.json", bootstrap)
    lines = ["# Full M0/M1/M2 experiment", "", "## Protocol", "",
             "All 5,062 official training claims were used, with all 723 validation and 1,447 test claims. Final models use seeds 13, 42 and 2026.", "",
             f"Seed-42 validation tuning selected learning rate {selected['learning_rate']} and M2 evidence lambda {selected['evidence_lambda']}. Search candidates were LR {{1e-5, 2e-5}} and lambda {{0.1, 0.5, 1.0}}. Test data did not select hyperparameters or checkpoints.", "",
             "Reranker: three epochs, selected by validation MRR. Common weighted verifier initialization: three epochs. M1/M2 continuation: two epochs from the same best initialization checkpoint. M0: five verifier epochs from identical initial verifier weights. Ties retain earlier epochs. Learning-rate trials use M1; lambda trials use M2 and one common selected-LR warmup checkpoint. Seed-42 tuning winners are reused for final M1/M2 evaluation.", "",
             "All models keep separate PhoBERT-base encoders where applicable, 256 pair tokens, 96 claim tokens, BM25 top 20 and selected top five. Adafactor, eight-claim accumulation, mixed precision and activation checkpointing remain enabled. Both M2 encoders are trainable. Lambda applies to M2 evidence loss; it has no effect in M0/M1.", "",
             "## Aggregate scores", "", "Scores are means and sample standard deviations across three seeds, not a vote ensemble.", "",
             "| Model | Dev Macro-F1 | Test Macro-F1 | Test accuracy |", "|---|---:|---:|---:|"]
    def display(stat):
        return f"{stat['mean']:.4f} +/- {stat['sample_std']:.4f}"
    for mode in MODELS:
        lines.append(f"| {mode} | {display(aggregate[mode]['dev_macro_f1'])} | {display(aggregate[mode]['test_macro_f1'])} | {display(aggregate[mode]['accuracy'])} |")
    lines += ["", "## Per-seed scores", "", "| Seed | Model | Dev Macro-F1 | Test Macro-F1 | Accuracy |", "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['model']} | {row['dev_macro_f1']:.4f} | {row['test_macro_f1']:.4f} | {row['accuracy']:.4f} |")
    lines += ["", "## Per-label F1", "", "| Model | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |", "|---|---:|---:|---:|"]
    for mode in MODELS:
        lines.append(f"| {mode} | {display(aggregate[mode]['supported_f1'])} | {display(aggregate[mode]['refuted_f1'])} | {display(aggregate[mode]['not_enough_info_f1'])} |")
    lines += ["", "## Retrieval and alignment", "", "Each record reports its eligible aligned SUPPORTED/REFUTED denominator. NEI and unsuccessful or ambiguous alignment are excluded from evidence metrics. Classification keeps all valid claims.", "",
              "| Seed | Model | Eligible claims | Hit@5 | Hit@20 | Recall@5 | MRR | Complete gold@5 | Alignment coverage |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['model']} | {row['count']} | {row['hit@5']:.4f} | {row['hit@20']:.4f} | {row['sentence_recall@5']:.4f} | {row['mrr']:.4f} | {row['complete_gold@5']:.4f} | {row['alignment_coverage']:.4f} |")
    lines += ["", "## Paired differences", "", "| Comparison | Mean Macro-F1 difference | Sample SD across seeds |", "|---|---:|---:|"]
    for name, record in differences.items():
        lines.append(f"| {name} | {record['mean']:+.4f} | {record['sample_std']:.4f} |")
    lines += ["", "Per-seed test-sample intervals use 2,000 paired article-cluster bootstrap replicates. They are conditional on fitted checkpoints and do not replace training-seed variability.", "",
              "| Seed | Comparison | Macro-F1 difference [95% CI] | REFUTED F1 difference [95% CI] |", "|---|---|---:|---:|"]
    for name, record in bootstrap.items():
        seed_name, pair = name.split(":")
        cells = []
        for key in ("macro_f1", "refuted_f1"):
            stat = record["metrics"][key]
            cells.append(f"{stat['difference']:+.4f} [{stat['ci95'][0]:+.4f}, {stat['ci95'][1]:+.4f}]")
        lines.append(f"| {seed_name.removeprefix('seed')} | {pair} | {' | '.join(cells)} |")
    lines += ["", "## Confusion matrices, diagnostics and costs", ""]
    for seed in SEEDS:
        for mode in MODELS:
            metric = metrics[seed][mode]
            lines += [f"### Seed {seed}: {mode}", "", "Rows are gold labels; columns are SUPPORTED / REFUTED / NOT_ENOUGH_INFO.", "",
                      "| Gold label | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |", "|---|---:|---:|---:|"]
            for label, counts in zip(LABELS, metric["classification"]["confusion_matrix"]):
                lines.append(f"| {label} | {' | '.join(map(str, counts))} |")
            lines += ["", f"Per-label precision/recall/F1: {metric['classification']['per_label']}", "",
                      f"Error stages: {metric['error_stages']}", "", f"Truncation: {metric['truncation']}", "",
                      f"Prepared-input inference cost: {metric['cost']}", ""]
    lines += ["## Training resource measurements", "", "Training times exclude validation and checkpoint writes. Shared preparation executes once; phase costs below must not be double-counted across model configurations.", "",
              "| Job | Phase | Training minutes | Peak allocated MiB | Peak reserved MiB |", "|---|---|---:|---:|---:|"]
    histories = sorted(directory.glob("tuning/*/*_history.json")) + sorted(directory.glob("seeds/*/*_history.json"))
    for path in histories:
        history = read_json(path)
        minutes = sum(e["training_seconds"] for e in history) / 60
        allocated = max((e.get("peak_training_cuda") or {}).get("allocated_mib", 0) for e in history)
        reserved = max((e.get("peak_training_cuda") or {}).get("reserved_mib", 0) for e in history)
        lines.append(f"| {path.parent.relative_to(directory)} | {path.stem.removesuffix('_history')} | {minutes:.1f} | {allocated:.1f} | {reserved:.1f} |")
    lines += ["", "GPU memory figures use the PyTorch allocator and exclude driver/display allocations. Inference timings exclude VnCoreNLP segmentation.", "",
              "## Limitations and reproduction", "", "Official splits contain overlapping source articles; this study does not measure generalization to unseen articles. Three training seeds provide a small estimate of training variability. No A1/A2/D0 ablations or manual contradiction annotation were run, so component-level causal explanations remain untested.", "",
              "This full experiment follows the pilot and uses the same public official test split. The pilot previously exposed test scores; full-study selection nevertheless uses validation alone. It is not a new blind test set.", "",
              f"Pinned data checksums: {plan['data_checksums']}", "", "Exact configurations, source hashes and tuning protocol: `full_plan.json`. Hyperparameter trial scores: `selected_hyperparameters.json`. Per-claim predictions, checkpoints, metrics and behavior checks: `seeds/seed*/`. Parent training sources remain in `tuning/`. Runtime progress and completion times: `status.json` and `completion.json`."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison
