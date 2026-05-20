from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerotrack.config import write_yaml
from aerotrack.contracts import SUMMARY_FIELDS
from aerotrack.data_prep import PreparedData, prepare_carrada_ra_smoke
from aerotrack.experiment import ensure_experiment_subdirs, experiment_dir, prepare_experiment_dir
from aerotrack.metrics import (
    evaluate_detection,
    evaluate_tracking,
    failure_examples,
    generate_gt_detections,
)
from aerotrack.tracking import run_sort_tracking
from aerotrack.visualization import render_visualizations, write_failure_report


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    prepared_data: PreparedData
    detections_path: Path
    tracks_path: Path
    summary_path: Path


def run_prepare_data(config: dict[str, Any]) -> PreparedData:
    return prepare_carrada_ra_smoke(config)


def run_detection_stage(config: dict[str, Any]) -> Path:
    _require_gt_bbox(config)
    prepared = _require_prepared_data(config)
    run_dir = _stage_run_dir(config)
    detections_path = generate_gt_detections(
        prepared.annotations_path,
        run_dir / "detections" / "detections.csv",
        float(config.get("detector", {}).get("score", 1.0)),
    )
    write_yaml(run_dir / "prepared_data.yaml", {"prepared_root": str(prepared.root), "stage": "detection"})
    return detections_path


def run_tracking_stage(config: dict[str, Any]) -> Path:
    prepared = _require_prepared_data(config)
    run_dir = _stage_run_dir(config)
    detections_path = _require_file(run_dir / "detections" / "detections.csv", "detections.csv")
    tracks_path = run_sort_tracking(detections_path, run_dir / "tracks" / "tracks.csv", config.get("tracker", {}))
    write_yaml(run_dir / "prepared_data.yaml", {"prepared_root": str(prepared.root), "stage": "tracking"})
    return tracks_path


def run_evaluation_stage(config: dict[str, Any]) -> Path:
    prepared = _require_prepared_data(config)
    run_dir = _stage_run_dir(config)
    detections_path = _require_file(run_dir / "detections" / "detections.csv", "detections.csv")
    tracks_path = _require_file(run_dir / "tracks" / "tracks.csv", "tracks.csv")
    split = str(config.get("evaluation", {}).get("split", "test"))
    split_sample_ids = _read_split(prepared.root / "splits" / f"{split}.txt")
    iou_threshold = float(config.get("evaluation", {}).get("iou_threshold", 0.5))
    detection_metrics = evaluate_detection(
        prepared.annotations_path,
        detections_path,
        run_dir / "metrics" / "detection_metrics.json",
        split_sample_ids=split_sample_ids,
        iou_threshold=iou_threshold,
    )
    tracking_metrics = evaluate_tracking(
        prepared.annotations_path,
        tracks_path,
        run_dir / "metrics" / "tracking_metrics.json",
        split_sample_ids=split_sample_ids,
        iou_threshold=iou_threshold,
    )
    summary_path = _write_summary(run_dir, config, split, prepared, len(split_sample_ids), detection_metrics, tracking_metrics)
    write_failure_report(
        run_dir,
        failure_examples(
            prepared.annotations_path,
            detections_path,
            tracks_path,
            split_sample_ids=split_sample_ids,
            iou_threshold=iou_threshold,
        ),
    )
    return summary_path


def run_visualization_stage(config: dict[str, Any]) -> Path:
    prepared = _require_prepared_data(config)
    run_dir = _stage_run_dir(config)
    detections_path = _require_file(run_dir / "detections" / "detections.csv", "detections.csv")
    tracks_path = _require_file(run_dir / "tracks" / "tracks.csv", "tracks.csv")
    classes = {int(item["id"]): str(item["name"]) for item in config["dataset"].get("classes", [])}
    render_visualizations(
        prepared.root,
        run_dir,
        prepared.annotations_path,
        detections_path,
        tracks_path,
        classes,
        max_frames_per_sequence=int(config.get("visualization", {}).get("max_frames_per_sequence", 20)),
    )
    return run_dir / "visualizations"


def run_experiment(config: dict[str, Any]) -> PipelineResult:
    _require_gt_bbox(config)
    prepared = prepare_carrada_ra_smoke(config)
    run_dir = prepare_experiment_dir(config)
    classes = {int(item["id"]): str(item["name"]) for item in config["dataset"].get("classes", [])}
    detections_path = generate_gt_detections(
        prepared.annotations_path,
        run_dir / "detections" / "detections.csv",
        float(config.get("detector", {}).get("score", 1.0)),
    )
    tracks_path = run_sort_tracking(detections_path, run_dir / "tracks" / "tracks.csv", config.get("tracker", {}))
    split = str(config.get("evaluation", {}).get("split", "test"))
    split_sample_ids = _read_split(prepared.root / "splits" / f"{split}.txt")
    iou_threshold = float(config.get("evaluation", {}).get("iou_threshold", 0.5))
    detection_metrics = evaluate_detection(
        prepared.annotations_path,
        detections_path,
        run_dir / "metrics" / "detection_metrics.json",
        split_sample_ids=split_sample_ids,
        iou_threshold=iou_threshold,
    )
    tracking_metrics = evaluate_tracking(
        prepared.annotations_path,
        tracks_path,
        run_dir / "metrics" / "tracking_metrics.json",
        split_sample_ids=split_sample_ids,
        iou_threshold=iou_threshold,
    )
    failures = failure_examples(
        prepared.annotations_path,
        detections_path,
        tracks_path,
        split_sample_ids=split_sample_ids,
        iou_threshold=iou_threshold,
    )
    if config.get("visualization", {}).get("enabled", True):
        render_visualizations(
            prepared.root,
            run_dir,
            prepared.annotations_path,
            detections_path,
            tracks_path,
            classes,
            max_frames_per_sequence=int(config.get("visualization", {}).get("max_frames_per_sequence", 20)),
        )
        write_failure_report(run_dir, failures)
    else:
        write_failure_report(run_dir, failures)
    summary_path = _write_summary(
        run_dir,
        config,
        split,
        prepared,
        len(split_sample_ids),
        detection_metrics,
        tracking_metrics,
    )
    write_yaml(run_dir / "prepared_data.yaml", {"prepared_root": str(prepared.root), "split": split})
    return PipelineResult(run_dir, prepared, detections_path, tracks_path, summary_path)


def _require_gt_bbox(config: dict[str, Any]) -> None:
    detector_source = config.get("detector", {}).get("source")
    if detector_source != "gt_bbox":
        raise ValueError(
            f"Stage1 gt_bbox smoke pipeline only supports detector.source=gt_bbox; got {detector_source!r}. "
            "YOLO inference is reserved for the next stage."
        )


def _stage_run_dir(config: dict[str, Any]) -> Path:
    return ensure_experiment_subdirs(experiment_dir(config))


def _require_prepared_data(config: dict[str, Any]) -> PreparedData:
    prepared = _prepared_from_config(config)
    missing = [
        path
        for path in [prepared.sample_index_path, prepared.annotations_path, prepared.classes_path, prepared.conversion_records_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Prepared data is missing; run `uv run python scripts/prepare_data.py --config ...` first. "
            f"Missing: {', '.join(str(path) for path in missing)}"
        )
    return prepared


def _prepared_from_config(config: dict[str, Any]) -> PreparedData:
    repo_root = Path(config.get("repo_root", ".")).resolve()
    prepared_root = Path(config["dataset"].get("prepared_root", "data/processed/carrada_ra_smoke"))
    if not prepared_root.is_absolute():
        prepared_root = repo_root / prepared_root
    sample_count = _csv_row_count(prepared_root / "sample_index.csv")
    sequence_count = _csv_unique_count(prepared_root / "sample_index.csv", "sequence_id")
    return PreparedData(
        root=prepared_root,
        sample_index_path=prepared_root / "sample_index.csv",
        annotations_path=prepared_root / "annotations.csv",
        classes_path=prepared_root / "classes.yaml",
        conversion_records_path=prepared_root / "conversion_records.csv",
        split_paths={split: prepared_root / "splits" / f"{split}.txt" for split in ("train", "val", "test")},
        num_sequences=sequence_count,
        num_frames=sample_count,
    )


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} is missing at {path}; run the preceding stage first.")
    return path


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _csv_unique_count(path: Path, field: str) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return len({row[field] for row in csv.DictReader(handle)})


def _read_split(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _write_summary(
    run_dir: Path,
    config: dict[str, Any],
    split: str,
    prepared: PreparedData,
    split_frames: int,
    detection_metrics: Any,
    tracking_metrics: dict[str, Any],
) -> Path:
    path = run_dir / "metrics" / "summary.csv"
    row = {
        "experiment_name": str(config["experiment_name"]),
        "detection_source": str(config.get("detector", {}).get("source", "")),
        "split": split,
        "num_sequences": str(prepared.num_sequences),
        "num_frames": str(split_frames),
        "precision": f"{detection_metrics.precision:.6f}",
        "recall": f"{detection_metrics.recall:.6f}",
        "f1": f"{detection_metrics.f1:.6f}",
        "map50": f"{detection_metrics.map50:.6f}",
        "mota": f"{float(tracking_metrics['mota']):.6f}",
        "idf1_status": str(tracking_metrics.get("idf1", {}).get("status", "unavailable")),
        "notes": "ID metrics unavailable in Stage1 smoke.",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return path
