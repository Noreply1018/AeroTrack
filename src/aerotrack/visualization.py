from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "gt": (0, 255, 0),
    "detections": (255, 210, 0),
    "tracks": (0, 180, 255),
}


def render_visualizations(
    prepared_root: Path,
    run_dir: Path,
    annotations_path: Path,
    detections_path: Path,
    tracks_path: Path,
    classes: dict[int, str],
    *,
    max_frames_per_sequence: int,
) -> None:
    _render_rows(prepared_root, run_dir / "visualizations" / "gt", _read_csv(annotations_path), classes, "gt", max_frames_per_sequence)
    _render_rows(
        prepared_root,
        run_dir / "visualizations" / "detections",
        _read_csv(detections_path),
        classes,
        "detections",
        max_frames_per_sequence,
    )
    track_rows = _read_csv(tracks_path)
    _render_rows(prepared_root, run_dir / "visualizations" / "tracks", track_rows, classes, "tracks", max_frames_per_sequence)
    _render_sequence_frames(prepared_root, run_dir / "visualizations" / "sequences", track_rows, classes, max_frames_per_sequence)
    for rel in ["visualizations/failures"]:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


def write_failure_report(run_dir: Path, failures: dict[str, object]) -> Path:
    target = run_dir / "visualizations" / "failures" / "failure_examples.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return target


def _render_rows(
    prepared_root: Path,
    output_dir: Path,
    rows: list[dict[str, str]],
    classes: dict[int, str],
    layer: str,
    max_frames_per_sequence: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["sequence_id"], row["frame_id"]), []).append(row)
    counts: dict[str, int] = {}
    for (sequence_id, frame_id), frame_rows in sorted(grouped.items()):
        if counts.get(sequence_id, 0) >= max_frames_per_sequence:
            continue
        source = prepared_root / "images" / sequence_id / f"{frame_id}.png"
        if not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        for row in frame_rows:
            _draw_row(draw, row, classes, layer)
        image.save(output_dir / f"{sequence_id}_{frame_id}.png")
        counts[sequence_id] = counts.get(sequence_id, 0) + 1


def _render_sequence_frames(
    prepared_root: Path,
    output_dir: Path,
    rows: list[dict[str, str]],
    classes: dict[int, str],
    max_frames_per_sequence: int,
) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sequence_id"], []).append(row)
    for sequence_id, sequence_rows in grouped.items():
        target_dir = output_dir / sequence_id
        _render_rows(prepared_root, target_dir, sequence_rows, classes, "tracks", max_frames_per_sequence)


def _draw_row(draw: ImageDraw.ImageDraw, row: dict[str, str], classes: dict[int, str], layer: str) -> None:
    xy = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
    color = COLORS[layer]
    draw.rectangle(xy, outline=color, width=2)
    class_name = classes.get(int(row["class_id"]), row["class_id"])
    if layer == "detections":
        label = f"{class_name} {float(row.get('score', 0.0)):.2f}"
    elif layer == "tracks":
        label = f"#{row.get('track_id', '?')} {class_name}"
    else:
        label = class_name
    draw.text((xy[0], max(0.0, xy[1] - 10)), label, fill=color)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
