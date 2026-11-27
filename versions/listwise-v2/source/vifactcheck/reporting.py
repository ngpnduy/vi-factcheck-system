"""Report the reduced-data pilot without treating it as a three-seed study."""
from collections import Counter
import csv
import io
from pathlib import Path

import numpy as np

from . import LABELS
from .io import read_json, read_jsonl, write_json


def _scores(matrix):
    diagonal = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(3), where=denominator != 0)
    return np.array([f1.mean(), f1[1]])


def paired_article_bootstrap(left, right, iterations=2000, seed=2026):
    """Resample entire articles, preserving paired claims and multiplicity."""
    def indexed(rows):
        result = {r["id"]: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError("Predictions must have unique, nonempty claim IDs")
        return result
    left, right = indexed(left), indexed(right)
    if left.keys() != right.keys():
        raise ValueError("Compared predictions have different claim IDs")
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive")
    articles = sorted({r["article_id"] for r in left.values()})
    article_index = {article: i for i, article in enumerate(articles)}
    matrices = np.zeros((2, len(articles), 3, 3), dtype=np.int64)
    for claim_id in sorted(left):
        a, b = left[claim_id], right[claim_id]
        if (a["gold_label_id"], a["article_id"]) != (b["gold_label_id"], b["article_id"]):
            raise ValueError("Gold labels/article IDs differ between models")
        for model, row in enumerate((a, b)):
            if row["gold_label_id"] not in (0, 1, 2) or row["label_id"] not in (0, 1, 2):
                raise ValueError("Unknown label in predictions")
            matrices[model, article_index[row["article_id"]], row["gold_label_id"], row["label_id"]] += 1
    point = _scores(matrices[0].sum(0)) - _scores(matrices[1].sum(0))
    rng, differences = np.random.default_rng(seed), []
    for _ in range(iterations):
        indices = rng.integers(len(articles), size=len(articles))
        sampled = matrices[:, indices].sum(axis=1)
        differences.append(_scores(sampled[0]) - _scores(sampled[1]))
    bounds = np.quantile(np.asarray(differences), [0.025, 0.975], axis=0)
    return {"iterations": iterations, "bootstrap_seed": seed, "claim_count": len(left), "article_count": len(articles), "method": "paired article-cluster percentile bootstrap; conditional on this training sample and seed", "metrics": {name: {"difference": float(point[i]), "ci95": bounds[:, i].tolist()} for i, name in enumerate(("macro_f1", "refuted_f1"))}}


def write_pilot_report(directory):
    directory = Path(directory)
    spec = read_json(directory / "pilot_spec.json")
    experiment = read_json(directory / "experiment.json")
    summary = read_json(directory / "summary.json")
    models = ("M0", "M1", "M2")
    test = {m: read_json(directory / f"{m.lower()}_test_metrics.json") for m in models}
    predictions = {m: read_jsonl(directory / f"{m.lower()}_test_predictions.jsonl") for m in models}
    expected = experiment["data_manifest"]["splits"]["test"]["valid_count"]
    if any(len(predictions[m]) != expected or test[m]["classification"]["count"] != expected for m in models):
        raise ValueError("Report requires complete test predictions for all three models")
    bootstrap = {f"{a}-{b}": paired_article_bootstrap(predictions[a], predictions[b]) for a, b in (("M1", "M0"), ("M2", "M1"))}
    write_json(directory / "paired_bootstrap.json", bootstrap)
    histories = {phase: read_json(directory / f"{phase}_history.json") for phase in ("reranker", "warmup", "m0", "m1", "m2")}
    costs = {}
    rows = []
    for model in models:
        phases = [model.lower()] if model == "M0" else ["reranker", "warmup", model.lower()]
        epochs = [epoch for phase in phases for epoch in histories[phase]]
        costs[model] = {"training_seconds": sum(e["training_seconds"] for e in epochs), "peak_training_allocated_mib": max((e.get("peak_training_cuda") or {}).get("allocated_mib", 0) for e in epochs), "peak_training_reserved_mib": max((e.get("peak_training_cuda") or {}).get("reserved_mib", 0) for e in epochs)}
        c, r = test[model]["classification"], test[model]["retrieval"]
        row = {"model": model, "seed": 42, "train_count": len(experiment["subset_ids"]["train"]), "test_count": c["count"], "dev_macro_f1": summary[model]["metrics"]["classification"]["macro_f1"], "test_macro_f1": c["macro_f1"], "accuracy": c["accuracy"], "hit_at_5": r["hit@5"], "hit_at_20": r["hit@20"], "sentence_recall_at_5": r["sentence_recall@5"], "sentence_recall_at_20": r["sentence_recall@20"], "mrr": r["mrr"], "complete_gold_at_5": r["complete_gold@5"], "complete_gold_at_20": r["complete_gold@20"], "retrieval_count": r["count"], **costs[model], "inference_seconds_per_claim": test[model]["cost"]["prepared_input_seconds_per_claim"]}
        rows.append(row)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (directory / "results.csv").write_text(buffer.getvalue(), encoding="utf-8-sig")
    comparison = {"study_type": "reduced-data single-seed pilot", "models": rows, "paired_article_bootstrap": bootstrap, "test_metrics": test, "cost_note": "M1/M2 include their common reranker and initialization costs; do not sum model costs to estimate actual shared workflow time."}
    write_json(directory / "comparison.json", comparison)
    lines = ["# M0/M1/M2: reduced-data pilot study", "", "## Experimental setup", "", f"We trained on {len(experiment['subset_ids']['train'])} official training claims (300 per label), selected by stratified random sampling with seed 42. Sampling did not filter by evidence alignment. Validation used all 723 claims; final evaluation used all 1,447 test claims. This is a single-seed pilot, not the full proposal experiment.", "", f"The training subset contains {spec['train_evidence_supervised_count']} claims eligible for balanced evidence supervision. Unaligned claims and NEI remain in verification training. Hyperparameters were fixed before training: learning rate 2e-5 and evidence lambda 0.5; no hyperparameter search was performed.", "", "Both components use separate PhoBERT-base parameters. Inputs keep the proposal's 256-token pair limit and 96-token claim limit. BM25 retrieves up to 20 candidates; the verifier uses up to five sentences. Short articles retain their actual sentence count. All configurations use Adafactor, eight-claim accumulation, mixed precision and activation checkpointing.", "", "M0 uniformly averages sentence label probabilities from BM25 top five. M1 reranks BM25 top 20 and uses softmax-weighted probabilities with a frozen reranker. M2 uses the same aggregation while verification gradients reach reranker scores, alongside eligible evidence supervision. Claim verification loss is applied after aggregation; gold evidence is never inserted into the candidate set.", "", "The reranker receives three epochs and is selected by validation MRR. The weighted verifier receives three initialization epochs; M1 and M2 branch from the same selected checkpoint for two continuation epochs. M0 receives five epochs from the same initial verifier weights. Final branch checkpoints are selected by validation Macro-F1; checkpoint ties favor the earlier epoch.", "", "## Classification results", "", "| Model | Validation Macro-F1 | Test Macro-F1 | Test accuracy |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['dev_macro_f1']:.4f} | {row['test_macro_f1']:.4f} | {row['accuracy']:.4f} |")
    lines += ["", "| Model | Label | Precision | Recall | F1 | Test support |", "|---|---|---:|---:|---:|---:|"]
    for model in models:
        for label in LABELS:
            scores = test[model]["classification"]["per_label"][label]
            lines.append(f"| {model} | {label} | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['f1']:.4f} | {scores['support']} |")
    lines += ["", "## Evidence retrieval", "", "Metrics use aligned SUPPORTED/REFUTED claims only. NEI and unsuccessful/ambiguous alignment are excluded. Hit counts any retrieved gold sentence; recall and complete-gold coverage account for all annotated sentences.", "", "| Model | Eligible claims | Hit@5 | Hit@20 | Recall@5 | Recall@20 | MRR | Complete gold@5 | Complete gold@20 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['retrieval_count']} | {row['hit_at_5']:.4f} | {row['hit_at_20']:.4f} | {row['sentence_recall_at_5']:.4f} | {row['sentence_recall_at_20']:.4f} | {row['mrr']:.4f} | {row['complete_gold_at_5']:.4f} | {row['complete_gold_at_20']:.4f} |")
    lines += ["", "## Paired differences and uncertainty", "", "Intervals use 2,000 paired article-cluster bootstrap replicates with bootstrap seed 2026. They quantify test-sample uncertainty conditional on this training sample and seed; they do not measure variability across training seeds.", "", "| Comparison | Macro-F1 difference [95% CI] | REFUTED F1 difference [95% CI] |", "|---|---:|---:|"]
    for name, result in bootstrap.items():
        values = [result["metrics"][key] for key in ("macro_f1", "refuted_f1")]
        formatted = [f"{v['difference']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for v in values]
        lines.append(f"| {name} | {' | '.join(formatted)} |")
    lines += ["", "## Runtime and GPU memory", "", "Training times exclude validation and checkpoint serialization. M1/M2 each include their shared preparation cost for comparison; that preparation executed once in the workflow. Inference latency uses prepared segmented input and excludes VnCoreNLP segmentation. Memory values are PyTorch allocator measurements, excluding driver/display allocations.", "", "| Model | Training minutes | Peak allocated MiB | Peak reserved MiB | Inference seconds/claim |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['training_seconds']/60:.1f} | {row['peak_training_allocated_mib']:.1f} | {row['peak_training_reserved_mib']:.1f} | {row['inference_seconds_per_claim']:.4f} |")
    lines += ["", "## Confusion matrices and diagnostic errors", "", "Rows are gold labels; columns are predicted labels, in SUPPORTED / REFUTED / NOT_ENOUGH_INFO order."]
    for model in models:
        lines += ["", f"### {model}", "", "| Gold / predicted | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |", "|---|---:|---:|---:|"]
        for label, counts in zip(LABELS, test[model]["classification"]["confusion_matrix"]):
            lines.append(f"| {label} | {' | '.join(map(str,counts))} |")
        lines += ["", "Error-stage counts: " + "; ".join(f"{key}={value}" for key, value in sorted(test[model]["error_stages"].items())) + "."]
        t = test[model]["truncation"]
        lines += ["", f"Truncation: {t['claim_truncated_count']}/{t['claim_count']} claims, {t['candidate_sentence_truncated_count']}/{t['candidate_pair_count']} candidate sentences, and {t['selected_sentence_truncated_count']}/{t['selected_pair_count']} selected sentences."]
    lines += ["", "## Evidence examples", "", "Examples below are the first correct and first incorrect test prediction, if available, in official test order for each model. They are illustrative examples and are not a manually annotated contradiction sample."]
    for model in models:
        for correct in (True, False):
            row = next((p for p in predictions[model] if (p["label_id"] == p["gold_label_id"]) == correct), None)
            if row:
                lines += ["", f"### {model}: {'correct' if correct else 'incorrect'} example ({row['id']})", "", "Claim: " + row.get("claim", "See the accompanying prediction record."), "", f"Gold: {LABELS[row['gold_label_id']]}; predicted: {row['label']}."]
                for sentence in row["selected_sentences"]:
                    lines.append(f"- Weight {sentence.get('weight',0):.4f}: {sentence.get('original',sentence.get('text',''))}")
    lines += ["", "## Limitations and interpretation", "", "Only 900 balanced training claims and one training seed were used. Hyperparameters were untuned; no between-seed standard deviation is available. These results are preliminary and cannot establish the behavior of full-data, three-seed training. Official splits share source articles, so results do not establish generalization to previously unseen articles.", "", "M1 versus M0 combines changed sentence selection and evidence weighting. M2 versus M1 measures the overall change from joint training; A1 and A2 are needed to isolate the individual mechanisms. Diagnostic retrieval/selection/verification categories are not causal proof. A1, A2, D0 and manual contradiction annotation were not run.", "", "Evidence alignment was conservative. Test alignment status counts: " + str(test["M0"]["retrieval"]["alignment_status_counts"]) + ". No fuzzy alignment or manual repair was used for this pilot. PyTorch may use a nondeterministic attention backward kernel despite seeded execution; independent GPU training is not claimed to be bitwise reproducible.", "", f"Pilot source-article overlaps: {spec['subset_article_overlap']}. The full official-split audit remains in the data manifest.", "", "## Reproduction records", "", f"GPU: {experiment['hardware']['gpu']}. Dataset revision: `{experiment['data_manifest']['assets']['dataset_revision']}`. PhoBERT revision: `{experiment['data_manifest']['assets']['model_revision']}`.", "", "Exact IDs and settings are in `experiment.json` and `pilot_spec.json`; epoch histories, behavioral checks, per-claim test predictions, `results.csv`, `comparison.json` and `paired_bootstrap.json` accompany this report."]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison
