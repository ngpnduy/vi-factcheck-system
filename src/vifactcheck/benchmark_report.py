"""Write benchmark tables and exports suitable for a browser file download."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import zipfile

from . import LABELS
from .io import read_json, read_jsonl, sha256, write_json


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def check_predictions(predictions, rows, mode):
    expected = {row["id"]: row for row in rows}
    if len(expected) != len(rows) or [p["id"] for p in predictions] != [r["id"] for r in rows]:
        raise ValueError("Evaluation predictions do not cover the expected distinct IDs in order")
    for prediction in predictions:
        row = expected[prediction["id"]]
        probabilities = prediction["probabilities"]
        if (prediction["gold_label_id"] != row["label"] or prediction["article_id"] != row["article_id"] or
                tuple(probabilities) != LABELS or
                any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()) or
                not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5) or
                prediction["label_id"] != max(range(3), key=lambda i: probabilities[LABELS[i]])):
            raise ValueError(f"Invalid prediction for {row['id']}")
        ranking = prediction["ranking"]
        if mode == "M0" and (ranking != row["candidate_ids"] or
                              [s["id"] for s in prediction["selected_sentences"]] != ranking[:5]):
            raise ValueError(f"M0 did not use the prepared BM25 order for {row['id']}")
        if mode == "M2" and (sorted(ranking) != sorted(row["candidate_ids"]) or
                              [s["id"] for s in prediction["selected_sentences"]] != ranking[:5]):
            raise ValueError(f"M2 did not select from the prepared BM25 candidates for {row['id']}")
        if mode == "BASELINE" and (ranking or prediction["selected_sentences"]):
            raise ValueError(f"Baseline emitted a sentence ranking for {row['id']}")


def summary_markdown(path, rows, purpose, control):
    control_label = "Baseline" if control == "BASELINE" else control
    lines = [f"# {purpose}", "",
             f"| Backbone | System | Seeds | Macro-F1 (mean ± SD) | Accuracy (mean ± SD) | Δ Macro-F1 vs {control_label} |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        def score(metric):
            sd = row[f"{metric}_sd"]
            return f"{row[f'{metric}_mean']:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")

        delta = row["paired_macro_f1_delta_mean"]
        lines.append(f"| {row['backbone']} | {row['system']} | {row['seed_count']} | {score('macro_f1')} | "
                     f"{score('accuracy')} | {f'{delta:+.4f}' if delta is not None else '—'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output, plan, rows, checkpoints, split):
    from .reporting import paired_article_bootstrap

    per_seed, aggregate, contrasts = [], [], {}
    backbone = plan["assets"]["backbone"]
    control = plan.get("control", "BASELINE")
    modes = (control, "M2")
    for seed in plan["seeds"]:
        predictions_by_mode = {}
        for mode in modes:
            prefix = output / "evaluation" / f"seed-{seed}" / f"{mode.lower()}_{split}"
            metrics = read_json(str(prefix) + "_metrics.json")
            predictions = read_jsonl(str(prefix) + "_predictions.jsonl")
            check_predictions(predictions, rows, mode)
            predictions_by_mode[mode] = predictions
            item = checkpoints[f"{seed}/{mode}"]
            scores, cost = metrics["classification"], metrics["cost"]
            if scores["count"] != len(rows):
                raise ValueError("Metric sample count differs from the evaluation split")
            per_seed.append({
                "backbone": backbone, "system": mode, "seed": seed, "split": split,
                "claims": scores["count"], "macro_f1": scores["macro_f1"], "accuracy": scores["accuracy"],
                "selected_epoch": item["selection"]["selected_global_epoch"],
                "selected_stage": item["selection"]["selected_stage"], "checkpoint_sha256": item["sha256"],
                "parameters": cost["parameters"], "total_input_tokens": cost["total_input_tokens"],
                "context_truncated_pairs": metrics["truncation"]["selected_sentence_truncated_count"],
                "context_tokens_kept": metrics["truncation"]["verifier_context_tokens_kept"],
                "prepared_input_seconds_per_claim": cost["prepared_input_seconds_per_claim"],
                "training_seconds": item["training_cost"]["training_seconds"],
                "peak_training_allocated_mib": item["training_cost"]["peak_training_allocated_mib"],
                "peak_training_reserved_mib": item["training_cost"]["peak_training_reserved_mib"],
                "retrieval_eligible_claims": metrics["retrieval"]["count"] if metrics["retrieval"] else None,
                "retrieval_mrr": metrics["retrieval"]["mrr"] if metrics["retrieval"] else None,
            })
        if split == "test":
            contrasts[str(seed)] = paired_article_bootstrap(predictions_by_mode["M2"], predictions_by_mode[control])
    for mode in modes:
        selected = [row for row in per_seed if row["system"] == mode]
        result = {"backbone": backbone, "system": mode, "split": split, "seed_count": len(selected)}
        for metric in ("macro_f1", "accuracy"):
            values = [row[metric] for row in selected]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_sd"] = stdev(values) if len(values) > 1 else None
        result["paired_macro_f1_delta_mean"] = (mean(c["metrics"]["macro_f1"]["difference"] for c in contrasts.values())
                                                 if mode == "M2" and contrasts else None)
        aggregate.append(result)
    write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "summary.csv", aggregate)
    title = "Smoke check on development data" if split == "dev" else "Results on the official test split"
    summary_markdown(output / "REPORT.md", aggregate, title, control)
    write_json(output / "comparison.json", {"summary": aggregate, "per_seed": per_seed,
                                           f"m2_minus_{control.lower()}_paired_article_bootstrap": contrasts})


def compare_runs(runs, destination):
    summary, details, signatures, backbones = [], [], [], set()
    for run in map(Path, runs):
        plan = read_json(run / "plan.json")
        completion = read_json(run / "completion.json")
        if plan["kind"] != "benchmark" or completion["status"] != "complete" or completion["split"] != "test":
            raise ValueError(f"Only completed test benchmarks can be combined: {run}")
        if completion["report_sha256"] != sha256(run / "comparison.json"):
            raise ValueError(f"Completed report checksum differs: {run}")
        backbone = plan["assets"]["backbone"]
        if backbone in backbones:
            raise ValueError(f"Duplicate backbone: {backbone}")
        backbones.add(backbone)
        control = plan.get("control", "BASELINE")
        signatures.append((plan["data_manifest"]["prepared_file_sha256"], plan["seeds"],
                           plan["hyperparameter_policy"], plan["search"], control,
                           {k: v for k, v in plan["config"].items()
                            if k not in ("model_path", "data_path", "backbone", "device")}))
        report = read_json(run / "comparison.json")
        summary.extend(report["summary"])
        details.extend(report["per_seed"])
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Runs use different controls, data, seeds, training constraints, or tuning policies")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "per_seed.csv", details)
    summary_markdown(destination / "REPORT.md", summary, "Backbone comparison on the official test split", control)
    return {"report": str(destination / "REPORT.md"), "backbones": sorted(backbones)}


def compact_prediction(prediction):
    fields = ("id", "configuration", "label", "label_id", "gold_label_id", "probabilities",
              "ranking", "truncation", "error_stage")
    result = {key: prediction[key] for key in fields if key in prediction}
    result["article_id"] = hashlib.sha256(str(prediction["article_id"]).encode()).hexdigest()
    result["selected_sentences"] = [
        {key: sentence[key] for key in ("id", "start", "end", "weight")}
        for sentence in prediction["selected_sentences"]
    ]
    result["alignment"] = {key: prediction["alignment"].get(key) for key in ("status", "gold_sentence_ids")}
    return result


def export_results(run):
    from .full import exclusive_run

    run = Path(run).resolve()
    if not (run / "plan.json").is_file() and not (run / "failure.json").is_file():
        raise FileNotFoundError(f"No benchmark output at {run}")
    with exclusive_run(run):
        return export_locked(run)


def export_locked(output):
    root_files = {
        "plan.json", "status.json", "failure.json", "completion.json", "resource_probe.json",
        "selected_hyperparameters.json", "selected_checkpoints.json", "tuning_trials.json",
        "job_timings.json", "comparison.json", "summary.csv", "per_seed.csv", "REPORT.md", "checkpoints.sha256",
    }
    job_suffixes = ("_history.json", "_selection.json", "_checks.json", "_metrics.json")
    complete = (output / "completion.json").is_file()
    downloads = output / "download"
    downloads.mkdir(exist_ok=True)
    archive = downloads / ("results.zip" if complete else "diagnostics.zip")
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if not path.is_file() or path.is_symlink():
                continue
            if len(relative.parts) == 1 and (path.name in root_files or
                                             (path.name.startswith("slurm-") and path.suffix == ".log")):
                bundle.write(path, relative.as_posix())
            elif relative.parts[0] in ("jobs", "evaluation"):
                if path.name.endswith("_predictions.jsonl"):
                    compact = "".join(json.dumps(compact_prediction(p), ensure_ascii=False) + "\n"
                                      for p in read_jsonl(path))
                    bundle.writestr(relative.as_posix(), compact)
                elif (path.name.endswith(job_suffixes) or
                      path.name in ("experiment.json", "training_complete.json", "checkpoint_cleanup.json")):
                    bundle.write(path, relative.as_posix())
    temporary.replace(archive)
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    if complete:
        # Also expose selected weight files individually. Links on the run
        # filesystem do not need a second copy of large model weights.
        (downloads / "checkpoint_locations.json").write_bytes((output / "selected_checkpoints.json").read_bytes())
    return {"status": read_json(output / "completion.json")["status"] if complete else "partial",
            "archive": str(archive), "checksum": str(checksum)}
