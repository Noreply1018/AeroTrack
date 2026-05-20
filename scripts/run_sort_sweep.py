from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from aerotrack.config import load_experiment_config, write_yaml
from aerotrack.metrics import evaluate_tracking
from aerotrack.tracking import run_sort_tracking


SWEEP_FIELDS = [
    "experiment_name",
    "max_age",
    "min_hits",
    "iou_threshold",
    "num_tracks",
    "num_track_rows",
    "mota",
    "tp",
    "fp",
    "fn",
    "idf1_status",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small SORT parameter sweep on existing detections.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = str(config.get("evaluation", {}).get("split", "test"))
    split_sample_ids = _read_split(Path(args.prepared_root) / "splits" / f"{split}.txt")
    iou_threshold = float(config.get("evaluation", {}).get("iou_threshold", 0.5))

    rows = []
    for variant in _variants(config.get("tracker", {})):
        name = f"sort_age{variant['max_age']}_hits{variant['min_hits']}_iou{str(variant['iou_threshold']).replace('.', 'p')}"
        tracks_path = output_dir / name / "tracks.csv"
        metrics_path = output_dir / name / "tracking_metrics.json"
        run_sort_tracking(Path(args.detections), tracks_path, variant)
        metrics = evaluate_tracking(
            Path(args.annotations),
            tracks_path,
            metrics_path,
            split_sample_ids=split_sample_ids,
            iou_threshold=iou_threshold,
        )
        track_rows = _read_csv(tracks_path)
        rows.append(
            {
                "experiment_name": name,
                "max_age": str(variant["max_age"]),
                "min_hits": str(variant["min_hits"]),
                "iou_threshold": f"{float(variant['iou_threshold']):.2f}",
                "num_tracks": str(len({(row["sequence_id"], row["track_id"]) for row in track_rows})),
                "num_track_rows": str(len(track_rows)),
                "mota": f"{float(metrics['mota']):.6f}",
                "tp": str(metrics["tp"]),
                "fp": str(metrics["fp"]),
                "fn": str(metrics["fn"]),
                "idf1_status": str(metrics.get("idf1", {}).get("status", "unavailable")),
            }
        )
        write_yaml(output_dir / name / "tracker.yaml", variant)

    _write_csv(output_dir / "sort_sweep_summary.csv", SWEEP_FIELDS, rows)
    (output_dir / "sort_sweep_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


def _variants(base_tracker: dict[str, Any]) -> list[dict[str, Any]]:
    base = deepcopy(base_tracker)
    candidates = [
        {"max_age": 1, "min_hits": 1, "iou_threshold": 0.10},
        {"max_age": 1, "min_hits": 1, "iou_threshold": 0.30},
        {"max_age": 3, "min_hits": 1, "iou_threshold": 0.30},
        {"max_age": 5, "min_hits": 1, "iou_threshold": 0.30},
        {"max_age": 3, "min_hits": 2, "iou_threshold": 0.30},
        {"max_age": 3, "min_hits": 1, "iou_threshold": 0.50},
    ]
    variants = []
    seen = set()
    for candidate in candidates:
        variant = deepcopy(base)
        variant.update(candidate)
        key = (variant["max_age"], variant["min_hits"], variant["iou_threshold"])
        if key not in seen:
            variants.append(variant)
            seen.add(key)
    return variants


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_split(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
