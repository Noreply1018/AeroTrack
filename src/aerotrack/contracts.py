from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SAMPLE_INDEX_FIELDS = [
    "sample_id",
    "sequence_id",
    "frame_id",
    "split",
    "representation",
    "image_path",
    "label_path",
]

ANNOTATION_FIELDS = [
    "sequence_id",
    "frame_id",
    "object_id",
    "class_id",
    "x1",
    "y1",
    "x2",
    "y2",
]

CONVERSION_RECORD_FIELDS = [
    "sample_id",
    "sequence_id",
    "frame_id",
    "object_id",
    "raw_label",
    "class_id",
    "class_name",
    "bbox_source",
    "image_source",
    "image_path",
    "label_path",
    "notes",
]

DETECTION_FIELDS = [
    "sequence_id",
    "frame_id",
    "class_id",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
]

SUMMARY_FIELDS = [
    "experiment_name",
    "detection_source",
    "split",
    "num_sequences",
    "num_frames",
    "precision",
    "recall",
    "f1",
    "map50",
    "mota",
    "idf1_status",
    "notes",
]

TRACK_FIELDS = [
    "sequence_id",
    "frame_id",
    "track_id",
    "class_id",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
]


@dataclass(frozen=True)
class CsvValidation:
    path: Path
    required_fields: tuple[str, ...]
    actual_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_fields


def validate_csv_fields(path: str | Path, required_fields: Iterable[str]) -> CsvValidation:
    csv_path = Path(path)
    required = tuple(required_fields)
    if not csv_path.exists():
        return CsvValidation(csv_path, required, tuple(), required)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = []

    actual = tuple(header)
    missing = tuple(field for field in required if field not in actual)
    return CsvValidation(csv_path, required, actual, missing)


def invalid_bbox(row: dict[str, str]) -> str | None:
    try:
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
    except (KeyError, TypeError, ValueError) as exc:
        return f"bbox coordinates are not numeric: {exc}"

    if x2 < x1 or y2 < y1:
        return "bbox has negative width or height"
    return None
