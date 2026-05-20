import csv
from pathlib import Path

from aerotrack.tracking import run_sort_tracking


def test_sort_tracking_keeps_constant_velocity_track(tmp_path: Path) -> None:
    detections = tmp_path / "detections.csv"
    with detections.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence_id", "frame_id", "class_id", "score", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"sequence_id": "s1", "frame_id": "000001", "class_id": "0", "score": "1", "x1": "0", "y1": "0", "x2": "10", "y2": "10"},
                {"sequence_id": "s1", "frame_id": "000002", "class_id": "0", "score": "1", "x1": "2", "y1": "0", "x2": "12", "y2": "10"},
                {"sequence_id": "s1", "frame_id": "000003", "class_id": "0", "score": "1", "x1": "4", "y1": "0", "x2": "14", "y2": "10"},
            ]
        )
    tracks = tmp_path / "tracks.csv"

    run_sort_tracking(detections, tracks, {"max_age": 3, "min_hits": 1, "iou_threshold": 0.3})

    rows = list(csv.DictReader(tracks.open("r", encoding="utf-8", newline="")))
    assert [row["track_id"] for row in rows] == ["1", "1", "1"]
