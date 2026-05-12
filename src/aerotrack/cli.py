from __future__ import annotations

import argparse
import sys

from aerotrack.config import ConfigError, load_detector_train_config, load_experiment_config
from aerotrack.experiment import prepare_experiment_dir
from aerotrack.preflight import format_preflight, run_preflight, run_training_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerotrack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-experiment")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="check local prerequisites without creating a run directory",
    )

    train_parser = subparsers.add_parser("train-detector")
    train_parser.add_argument("--config", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-experiment":
        try:
            config = load_experiment_config(args.config)
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        result = run_preflight(config)
        print(format_preflight(result))
        if result.exit_code != 0:
            if result.has_gate:
                print("Preflight stopped at a confirmation gate.", file=sys.stderr)
            return result.exit_code
        if args.preflight_only:
            return 0
        run_dir = prepare_experiment_dir(config)
        print(f"Experiment directory prepared: {run_dir}")
        print("Pipeline execution is not enabled until CARRADA data preparation is confirmed.")
        return 0

    if args.command == "train-detector":
        try:
            config = load_detector_train_config(args.config)
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        result = run_training_preflight(config)
        print(format_preflight(result))
        if result.exit_code != 0:
            if result.has_gate:
                print("Training preflight stopped at a confirmation gate.", file=sys.stderr)
            return result.exit_code
        print("Training execution is intentionally not implemented in Stage1 smoke setup.")
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
