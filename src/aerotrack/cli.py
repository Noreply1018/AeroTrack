from __future__ import annotations

import argparse
import sys

from aerotrack.config import ConfigError, load_detector_train_config, load_experiment_config
from aerotrack.pipeline import run_experiment, run_prepare_data
from aerotrack.pipeline import run_detection_stage, run_evaluation_stage, run_tracking_stage, run_visualization_stage
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

    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("--config", required=True)

    for name in ("run-detection", "run-tracking", "evaluate", "visualize"):
        stage_parser = subparsers.add_parser(name)
        stage_parser.add_argument("--config", required=True)

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
        try:
            result = run_experiment(config)
        except ValueError as exc:
            print(f"Pipeline error: {exc}", file=sys.stderr)
            return 2
        print(f"Prepared data: {result.prepared_data.root}")
        print(f"Detections: {result.detections_path}")
        print(f"Tracks: {result.tracks_path}")
        print(f"Summary: {result.summary_path}")
        return 0

    if args.command == "prepare-data":
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
        prepared = run_prepare_data(config)
        print(f"Prepared data: {prepared.root}")
        print(f"Sample index: {prepared.sample_index_path}")
        print(f"Annotations: {prepared.annotations_path}")
        return 0

    if args.command in {"run-detection", "run-tracking", "evaluate", "visualize"}:
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
        try:
            if args.command == "run-detection":
                output = run_detection_stage(config)
                print(f"Detections: {output}")
            elif args.command == "run-tracking":
                output = run_tracking_stage(config)
                print(f"Tracks: {output}")
            elif args.command == "evaluate":
                output = run_evaluation_stage(config)
                print(f"Summary: {output}")
            else:
                output = run_visualization_stage(config)
                print(f"Visualizations: {output}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Stage error: {exc}", file=sys.stderr)
            return 2
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
