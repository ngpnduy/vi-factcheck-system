"""Command line for the backbone benchmark."""


def add_benchmark_parser(commands):
    command = commands.add_parser("benchmark", help="M0 and listwise M2 on mBERT or XLM-R")
    actions = command.add_subparsers(dest="benchmark_action", required=True)
    for action in ("prepare", "smoke", "run"):
        parser = actions.add_parser(action)
        parser.add_argument("--backbone", choices=("mbert", "xlmr"), required=True)
        parser.add_argument("--cache", default=".cache")
        if action == "prepare":
            continue
        parser.add_argument("--data", default="data/prepared")
        parser.add_argument("--output", required=True)
        parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        parser.add_argument("--control", choices=("m0", "claim-article"), default="m0",
                            help="Control model alongside M2 (default: m0)")
    actions.choices["run"].add_argument(
        "--fixed-hyperparameters", action="store_true",
        help="Use LR 1e-5 and M2 lambda 0.1 without search (automatic with M0)")
    actions.choices["run"].add_argument(
        "--single-seed", action="store_true",
        help="Train and evaluate both systems on seed 42 only")
    export = actions.add_parser("export", help="Pack completed results or partial diagnostics without weights")
    export.add_argument("--run", required=True)
    compare = actions.add_parser("compare", help="Combine completed backbone reports")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output", required=True)


def dispatch_benchmark(args):
    if args.benchmark_action in ("export", "compare"):
        from .benchmark_report import compare_runs, export_results

        return export_results(args.run) if args.benchmark_action == "export" else compare_runs(args.runs, args.output)
    from .benchmark import prepare_backbone, run_benchmark

    if args.benchmark_action == "prepare":
        return prepare_backbone(args.backbone, args.cache)
    return run_benchmark(args)
