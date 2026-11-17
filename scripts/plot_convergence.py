"""Plot the CSV exported by analyze_convergence.py; no checkpoint loading."""
import argparse
import csv
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True, help="New output prefix for PNG and SVG")
    args = parser.parse_args(argv)
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["stage"] != "reranker"]
    if not rows:
        raise ValueError("No verifier observations to plot")
    output = Path(args.output)
    targets = [output.with_suffix(suffix) for suffix in (".png", ".svg")]
    if any(target.exists() for target in targets):
        raise FileExistsError("Choose a new plot prefix; existing figures are preserved")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["seed"]), []).append(row)
    has_nll = any(row.get("dev_nll") for row in rows)
    figure, axes = plt.subplots(2 if has_nll else 1, 2, figsize=(12, 8 if has_nll else 4.8), squeeze=False)
    flat = axes.ravel()
    for index, ((model, seed), values) in enumerate(sorted(groups.items())):
        color = f"C{index % 10}"
        values.sort(key=lambda row: int(row["global_epoch"]))
        label = f"{model}, seed {seed}"
        for stage, style in (("warmup", "--"), ("continuation", "-"), ("m0", "-"), ("a1", "-")):
            points = [row for row in values if row["stage"] == stage and row.get("train_objective")]
            if points:
                flat[0].plot([int(r["global_epoch"]) for r in points], [float(r["train_objective"]) for r in points],
                             style, marker="o", color=color, label=f"{label}: {stage}")
        points = [row for row in values if row.get("dev_macro_f1")]
        flat[1].plot([int(r["global_epoch"]) for r in points], [100 * float(r["dev_macro_f1"]) for r in points],
                     marker="o", color=color, label=label)
        if has_nll:
            for field, style, suffix in (("train_probe_nll", "--", "fixed train probe"), ("dev_nll", "-", "dev")):
                points = [row for row in values if row.get(field)]
                flat[2].plot([int(r["global_epoch"]) for r in points], [float(r[field]) for r in points],
                             style, marker="o", color=color, label=f"{label}: {suffix}")
            for field, style, suffix in (("train_evidence_loss", "--", "online train"), ("dev_evidence_bce", "-", "dev")):
                points = [row for row in values if row.get(field)]
                if points:
                    flat[3].plot([int(r["global_epoch"]) for r in points], [float(r[field]) for r in points],
                                 style, marker="o", color=color, label=f"{label}: {suffix}")
    titles = ("Training objective: stages plotted separately", "Held-out dev Macro-F1",
              "Verification NLL: evaluation mode", "Evidence BCE: eligible claims only")
    ylabels = ("Mean online objective", "Macro-F1 (%)", "Mean verification NLL", "Mean balanced BCE")
    for axis, title, ylabel in zip(flat, titles, ylabels):
        axis.set(title=title, xlabel="Global verifier epoch", ylabel=ylabel)
        axis.grid(alpha=.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
        axis.xaxis.get_major_locator().set_params(integer=True)
    figure.suptitle("Convergence diagnostic — train/dev only; a finite horizon does not establish an optimum", fontsize=12)
    figure.tight_layout(rect=(0, .04, 1, .94))
    figure.text(.01, .01, "Warmup and continuation optimize different objectives. Missing historical losses are not reconstructed.", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    for target in targets:
        figure.savefig(target, dpi=180, metadata={"Creator": "ViFactCheck convergence diagnostic"})
    plt.close(figure)
    print("Saved " + ", ".join(str(target) for target in targets))


if __name__ == "__main__":
    main()
