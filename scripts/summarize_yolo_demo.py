from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_FIELDS = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]


SHOWCASE_FILES = [
    "results.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "confusion_matrix.png",
    "labels.jpg",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize YOLO local demo outputs for presentation.")
    parser.add_argument(
        "--run-dir",
        default="runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu",
        help="Ultralytics training run directory.",
    )
    parser.add_argument(
        "--prediction-dir",
        default="runs/yolo_local_demo/carrada_ra_cpu10_showcase_pred",
        help="Ultralytics showcase prediction directory.",
    )
    parser.add_argument(
        "--output",
        default="runs/yolo_local_demo/presentation_summary.csv",
        help="Output CSV summary path.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    prediction_dir = Path(args.prediction_dir)
    output = Path(args.output)
    write_summary(run_dir, prediction_dir, output)
    print(f"Summary: {output}")
    return 0


def write_summary(run_dir: Path, prediction_dir: Path, output: Path) -> None:
    rows = metric_rows(run_dir / "results.csv") + artifact_rows(run_dir, prediction_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "name", "value", "path"])
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows found: {results_csv}")
    latest = _normalized_row(rows[-1])
    return [
        {
            "section": "metric",
            "name": field,
            "value": latest[field],
            "path": str(results_csv),
        }
        for field in METRIC_FIELDS
    ]


def artifact_rows(run_dir: Path, prediction_dir: Path) -> list[dict[str, str]]:
    rows = [
        {
            "section": "artifact",
            "name": rel,
            "value": "exists" if (run_dir / rel).exists() else "missing",
            "path": str(run_dir / rel),
        }
        for rel in SHOWCASE_FILES
    ]
    prediction_images = sorted(prediction_dir.glob("*.jpg"))
    rows.append(
        {
            "section": "artifact",
            "name": "showcase_prediction_images",
            "value": str(len(prediction_images)),
            "path": str(prediction_dir),
        }
    )
    prediction_labels = sorted((prediction_dir / "labels").glob("*.txt")) if (prediction_dir / "labels").exists() else []
    rows.append(
        {
            "section": "artifact",
            "name": "showcase_prediction_label_files",
            "value": str(len(prediction_labels)),
            "path": str(prediction_dir / "labels"),
        }
    )
    return rows


def _normalized_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


if __name__ == "__main__":
    raise SystemExit(main())
