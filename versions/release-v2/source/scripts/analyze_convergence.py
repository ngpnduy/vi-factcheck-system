"""Audit saved training histories without loading checkpoints or running training.

Example (from the repository root)::

    python scripts/analyze_convergence.py --run reports/experiments/listwise-best5-v2-three-seeds-20261112 --output runs/convergence-audit

Training objectives from different stages are deliberately kept separate. A
finite run can show an observed plateau or overfitting signal, never a proof of
the global optimum. Only dev metrics in the saved history are inspected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any


MODELS = ("M0", "M1", "A1", "M2", "A2")
FORMAT = "vifactcheck-convergence-analysis-v1"
CAUTIONS = [
    "Five verifier epochs and two joint-continuation observations do not establish convergence or a global optimum.",
    "Warmup and continuation can use different objectives and trainable parameters; do not interpret their losses as one stationary optimization curve.",
    "Lower training loss alone does not establish better verdict prediction; use held-out dev Macro-F1 and, when recorded, dev verification NLL.",
    "A dev decline with lower training loss is an overfitting signal, not proof of overfitting from one transition.",
    "No test scores are read or used for selection. A longer run is a new diagnostic protocol and must not overwrite the published five-epoch results.",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def number(value: Any, description: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{description} must be a finite number or null")
    return float(value)


def nested(value: dict, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        result: Any = value
        for key in path:
            if not isinstance(result, dict) or key not in result:
                result = None
                break
            result = result[key]
        if result is not None:
            return result
    return None


def infer_seed(path: Path, root: Path, selected: dict) -> int | None:
    match = re.search(r"(?:^|/)seed(\d+)(?:[-/]|$)", path.relative_to(root).as_posix(), re.IGNORECASE)
    if match:
        return int(match.group(1))
    for parent in (path.parent, root):
        for filename in ("convergence_plan.json", "experiment.json", "config.json"):
            candidate = parent / filename
            if candidate.is_file():
                value = read_json(candidate)
                seed = nested(value, ("config", "seed"), ("seed",))
                if seed is not None:
                    return int(seed)
    if "tuning" in path.relative_to(root).parts:
        return int(selected.get("seed", 42))
    return None


def is_selected_trial(path: Path, root: Path, selected: dict, all_trials: bool) -> bool:
    if all_trials or not selected:
        return True
    relative = path.relative_to(root).as_posix()
    for prefix, key in (("lr_", "learning_rate"), ("lambda_", "evidence_lambda")):
        match = re.search(r"(?:^|/)tuning/" + prefix + r"([^/]+)/", relative)
        if match and key in selected:
            try:
                trial = float(match.group(1))
            except ValueError:
                continue
            if trial != float(selected[key]):
                return False
    return True


def normalize_history(rows: Any, path: Path) -> list[dict]:
    if isinstance(rows, dict):
        rows = rows.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"History must be a nonempty list: {path}")
    result = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid history row {index}: {path}")
        epoch = row.get("global_epoch", row.get("epoch"))
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1 or epoch in seen:
            raise ValueError(f"Global epochs must be positive unique integers: {path}")
        seen.add(epoch)
        stage = row.get("stage", row.get("phase", "unspecified"))
        if not isinstance(stage, str) or not stage:
            raise ValueError(f"Each history row needs a nonempty stage: {path}")
        result.append({
            "global_epoch": epoch,
            "stage": stage,
            "train_objective": number(row.get("average_loss"), "average_loss"),
            "train_verification_loss": number(nested(row, ("loss_components", "verification_nll"), ("average_verification_loss",), ("train_verification_loss",)), "train verification loss"),
            "train_evidence_loss": number(nested(row, ("loss_components", "eligible_evidence_bce"), ("average_evidence_loss",), ("train_evidence_loss",)), "train evidence loss"),
            "train_evidence_weighted_contribution": number(nested(row, ("loss_components", "evidence_weighted_contribution")), "weighted evidence contribution"),
            "train_claim_count": nested(row, ("loss_components", "claim_count")),
            "train_eligible_evidence_count": nested(row, ("loss_components", "eligible_evidence_count")),
            "train_probe_nll": number(nested(row, ("diagnostics", "train_probe", "verification_nll")), "train-probe verification NLL"),
            "train_probe_evidence_bce": number(nested(row, ("diagnostics", "train_probe", "eligible_evidence_bce")), "train-probe evidence BCE"),
            "train_probe_claim_count": nested(row, ("diagnostics", "train_probe", "claim_count")),
            "train_probe_eligible_evidence_count": nested(row, ("diagnostics", "train_probe", "eligible_evidence_count")),
            "dev_macro_f1": number(nested(row, ("validation", "classification", "macro_f1"), ("dev_macro_f1",)), "dev Macro-F1"),
            "dev_nll": number(nested(row, ("diagnostics", "dev", "verification_nll"), ("validation", "verification_nll"), ("validation", "loss", "verification_nll"), ("validation", "loss", "verification"), ("dev_verification_nll",), ("dev_nll",)), "dev verification NLL"),
            "dev_evidence_bce": number(nested(row, ("diagnostics", "dev", "eligible_evidence_bce")), "dev evidence BCE"),
            "dev_eligible_evidence_count": nested(row, ("diagnostics", "dev", "eligible_evidence_count")),
            "dev_claim_count": nested(row, ("diagnostics", "dev", "claim_count"), ("validation", "classification", "count"), ("dev_claim_count",)),
        })
    result.sort(key=lambda row: row["global_epoch"])
    if any(right["global_epoch"] != left["global_epoch"] + 1 for left, right in zip(result, result[1:])):
        raise ValueError(f"History has missing global epochs: {path}")
    return result


def summarize_history(rows: list[dict], min_delta: float, plateau_window: int) -> dict:
    scored = [row for row in rows if row["dev_macro_f1"] is not None]
    best = max(scored, key=lambda row: (row["dev_macro_f1"], -row["global_epoch"])) if scored else None
    # Contiguous segments, not a pooled cross-stage fit.
    segments: list[list[dict]] = []
    for row in rows:
        if not segments or segments[-1][-1]["stage"] != row["stage"]:
            segments.append([])
        segments[-1].append(row)
    stages = []
    for segment in segments:
        losses = [row for row in segment if row["train_objective"] is not None]
        dev = [row for row in segment if row["dev_macro_f1"] is not None]
        loss_delta = (losses[-1]["train_objective"] - losses[-2]["train_objective"]) if len(losses) >= 2 else None
        dev_delta = (dev[-1]["dev_macro_f1"] - dev[-2]["dev_macro_f1"]) if len(dev) >= 2 else None
        signal = "insufficient_within_stage_observations"
        if dev_delta is not None:
            if dev_delta > min_delta:
                signal = "dev_improving_at_stage_boundary"
            elif loss_delta is not None and loss_delta < 0 and dev_delta < -min_delta:
                signal = "training_loss_decreases_but_dev_declines"
            elif len(dev) >= plateau_window and max(row["dev_macro_f1"] for row in dev[-plateau_window:]) - min(row["dev_macro_f1"] for row in dev[-plateau_window:]) <= min_delta:
                signal = "observed_dev_plateau_at_current_settings"
            else:
                signal = "no_clear_latest_dev_gain"
        stages.append({
            "stage": segment[0]["stage"], "first_global_epoch": segment[0]["global_epoch"],
            "last_global_epoch": segment[-1]["global_epoch"], "observations": len(segment),
            "first_train_objective": losses[0]["train_objective"] if losses else None,
            "last_train_objective": losses[-1]["train_objective"] if losses else None,
            "last_train_objective_delta": loss_delta, "last_dev_macro_f1_delta": dev_delta,
            "signal": signal,
            "plateau_window_observed": len(dev) >= plateau_window,
        })
    return {
        "epochs_observed": len(rows),
        "best_dev_global_epoch": best["global_epoch"] if best else None,
        "best_dev_macro_f1": best["dev_macro_f1"] if best else None,
        "best_at_run_boundary": best["global_epoch"] == rows[-1]["global_epoch"] if best else None,
        "final_dev_macro_f1": rows[-1]["dev_macro_f1"],
        "dev_nll_available": any(row["dev_nll"] is not None for row in rows),
        "convergence_established": False,
        "stages": stages,
        "rows": rows,
    }


def build_analysis(run: str | Path, models=("M2",), all_trials=False, min_delta=1e-4, plateau_window=3) -> dict:
    root = Path(run).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--run must be a directory")
    if not models or any(model not in MODELS for model in models):
        raise ValueError(f"models must be drawn from {MODELS}")
    if not math.isfinite(min_delta) or min_delta < 0 or plateau_window < 3:
        raise ValueError("min_delta must be nonnegative and plateau_window at least three")
    selected_path = root / "selected_hyperparameters.json"
    selected = read_json(selected_path) if selected_path.is_file() else {}
    histories = []
    signatures = {}
    excluded = []
    for path in sorted(root.rglob("*_history.json")):
        model = path.name.removesuffix("_history.json").upper()
        if model not in models:
            continue
        relative = path.relative_to(root).as_posix()
        if not is_selected_trial(path, root, selected, all_trials):
            excluded.append(relative)
            continue
        raw = path.read_bytes()
        normalized = normalize_history(read_json(path), path)
        seed = infer_seed(path, root, selected)
        signature = (seed, model, json.dumps(normalized, sort_keys=True))
        if signature in signatures:
            signatures[signature]["aliases"].append(relative)
            continue
        summary = summarize_history(normalized, min_delta, plateau_window)
        summary.update({"history_path": relative, "aliases": [], "history_sha256": hashlib.sha256(raw).hexdigest(), "model": model, "seed": seed})
        selection_path = path.with_name(model.lower() + "_selection.json")
        if selection_path.is_file():
            selection = read_json(selection_path)
            summary["recorded_selection"] = selection
            summary["recorded_selection_matches_history"] = selection.get("selected_global_epoch") == summary["best_dev_global_epoch"] and abs(selection.get("selected_validation_macro_f1", math.inf) - (summary["best_dev_macro_f1"] or 0)) <= 1e-12
        histories.append(summary)
        signatures[signature] = summary
    if not histories:
        raise ValueError(f"No requested model histories found in {root}")
    histories.sort(key=lambda row: (row["seed"] is None, row["seed"] or 0, row["model"], row["history_path"]))
    return {"format": FORMAT, "source_run": str(root), "models": list(models), "all_trials": bool(all_trials),
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "min_delta": min_delta, "plateau_window": plateau_window, "excluded_nonwinning_trials": excluded,
            "cautions": CAUTIONS, "histories": histories}


def markdown_report(analysis: dict) -> str:
    lines = ["# Saved-history convergence audit", "", f"Input: `{analysis['source_run']}`", "",
             "This is a read-only analysis of recorded train/dev histories. It does not train, load weights, or read test scores.", "",
             "| Model | Seed | Epochs | Best dev epoch | Best dev Macro-F1 | Best at final epoch | Final-stage signal |",
             "|---|---:|---:|---:|---:|---|---|"]
    for history in analysis["histories"]:
        score = history["best_dev_macro_f1"]
        text = f"{score:.6f}" if score is not None else "unavailable"
        lines.append(f"| {history['model']} | {history['seed']} | {history['epochs_observed']} | {history['best_dev_global_epoch']} | {text} | {history['best_at_run_boundary']} | {history['stages'][-1]['signal']} |")
    lines += ["", "## Interpretation", ""] + [f"- {caution}" for caution in analysis["cautions"]]
    lines += ["- Existing histories may not contain dev NLL or separate verification/evidence losses; missing values remain blank rather than being reconstructed from Macro-F1.",
              "- New `loss_components` fields describe online training with dropout enabled. `diagnostics.train_probe` and `diagnostics.dev` describe fixed-checkpoint evaluation with dropout disabled; compare their NLL on clearly reported sample counts.",
              "- An observed plateau describes only this finite window, learning rate, objective and data. It is not certification of the best possible model.", "", "## Within-stage loss changes", "",
              "| Model | Seed | Stage | Epoch range | Observations | First objective | Last objective | Latest dev F1 change |",
              "|---|---:|---|---|---:|---:|---:|---:|"]
    def format_value(value):
        return "unavailable" if value is None else f"{value:.6f}"
    for history in analysis["histories"]:
        for stage in history["stages"]:
            lines.append(f"| {history['model']} | {history['seed']} | {stage['stage']} | {stage['first_global_epoch']}-{stage['last_global_epoch']} | {stage['observations']} | {format_value(stage['first_train_objective'])} | {format_value(stage['last_train_objective'])} | {format_value(stage['last_dev_macro_f1_delta'])} |")
    lines += ["", "## Provenance", ""]
    for history in analysis["histories"]:
        lines += [f"- `{history['history_path']}`: SHA-256 `{history['history_sha256']}`; identical copies: {history['aliases'] or 'none'}."]
    if analysis["excluded_nonwinning_trials"]:
        lines += ["", "Nonwinning tuning histories were excluded using the study's recorded dev-selected hyperparameters. Use `--all-trials` to inspect them separately."]
    return "\n".join(lines) + "\n"


def write_analysis(analysis: dict, output: str | Path) -> dict:
    destination = Path(output).resolve()
    source = Path(analysis["source_run"])
    if destination == source:
        raise ValueError("Output must be separate from the input run directory")
    owner = destination / "analysis_manifest.json"
    if destination.exists() and any(destination.iterdir()):
        if not owner.is_file():
            raise ValueError("Refusing to overwrite a nonempty output without an analysis manifest")
        previous = read_json(owner)
        if previous.get("format") != FORMAT or previous.get("source_run") != str(source):
            raise ValueError("Refusing to overwrite an analysis of a different study")
    destination.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (destination / "convergence.json").write_text(payload, encoding="utf-8")
    stream = io.StringIO(newline="")
    fields = ["history_path", "model", "seed", "global_epoch", "stage", "train_objective", "train_verification_loss", "train_evidence_loss", "train_evidence_weighted_contribution", "train_claim_count", "train_eligible_evidence_count", "train_probe_nll", "train_probe_evidence_bce", "train_probe_claim_count", "train_probe_eligible_evidence_count", "dev_macro_f1", "dev_evidence_bce", "dev_eligible_evidence_count", "dev_nll", "dev_claim_count"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for history in analysis["histories"]:
        for row in history["rows"]:
            writer.writerow({"history_path": history["history_path"], "model": history["model"], "seed": history["seed"], **row})
    (destination / "convergence.csv").write_text(stream.getvalue(), encoding="utf-8")
    (destination / "REPORT.md").write_text(markdown_report(analysis), encoding="utf-8")
    manifest = {"format": FORMAT, "source_run": str(source), "files": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in ("convergence.json", "convergence.csv", "REPORT.md")}}
    owner.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", "--model", nargs="+", choices=MODELS, default=["M2"])
    parser.add_argument("--all-trials", action="store_true")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--plateau-window", type=int, default=3)
    args = parser.parse_args()
    analysis = build_analysis(args.run, tuple(args.models), args.all_trials, args.min_delta, args.plateau_window)
    write_analysis(analysis, args.output)
    print(f"Audited {len(analysis['histories'])} histories; results: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
