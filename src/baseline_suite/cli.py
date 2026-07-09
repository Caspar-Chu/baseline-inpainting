from __future__ import annotations

import argparse
import sys

from baseline_suite.experiments import EXPERIMENT_BUILDERS
from baseline_suite.paths import ensure_dirs
from baseline_suite.registry import available_baselines
from baseline_suite.runner import aggregate_trials, run_experiment


def cmd_make_masks(args: argparse.Namespace) -> int:
    ensure_dirs()
    run_experiment(args.experiment, baselines=[], config_path=args.config, cache_only=True)
    print(f"Masks cached for experiment: {args.experiment}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    baselines = args.baselines or None
    df = run_experiment(args.experiment, baselines, config_path=args.config)
    if args.aggregate and not df.empty and "trial" in df.columns:
        agg = aggregate_trials(df)
        out = df.attrs.get("output_root")
        print(agg.to_string(index=False))
    else:
        print(df.to_string(index=False) if not df.empty else "Done (cache only).")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    print("Experiments:")
    for name in EXPERIMENT_BUILDERS:
        print(f"  - {name}")
    print("Baselines:")
    for name in available_baselines():
        print(f"  - {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baseline-suite",
        description="Baseline augmentation experiment framework",
    )
    parser.add_argument("--config", default=None, help="Path to experiments.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_masks = sub.add_parser("make-masks", help="Generate and cache masks only")
    p_masks.add_argument("experiment", choices=list(EXPERIMENT_BUILDERS.keys()))
    p_masks.set_defaults(func=cmd_make_masks)

    p_run = sub.add_parser("run", help="Run inpainting + evaluation")
    p_run.add_argument("experiment", choices=list(EXPERIMENT_BUILDERS.keys()))
    p_run.add_argument(
        "-b",
        "--baselines",
        nargs="+",
        default=None,
        help=f"Baselines to run (default: all). Choices: {available_baselines()}",
    )
    p_run.add_argument("--aggregate", action="store_true", help="Print trial-averaged metrics")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="List experiments and baselines")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
