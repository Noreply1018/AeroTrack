from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from aerotrack.contracts import DETECTION_FIELDS, TRACK_FIELDS
from aerotrack.metrics import iou


@dataclass
class TrackState:
    track_id: int
    last_frame_number: int
    last_box: dict[str, str]
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    misses: int = 0
    hits: int = 1

    def predicted_box(self, frame_number: int) -> dict[str, str]:
        delta = max(1, frame_number - self.last_frame_number)
        box = dict(self.last_box)
        for key, value in zip(("x1", "y1", "x2", "y2"), self.velocity, strict=True):
            box[key] = f"{float(self.last_box[key]) + value * delta:.6f}"
        return box


def run_sort_tracking(detections_path: Path, output_path: Path, tracker_config: dict[str, Any]) -> Path:
    detections = _read_csv(detections_path)
    min_confidence = float(tracker_config.get("min_confidence", 0.0))
    max_age = int(tracker_config.get("max_age", 3))
    min_hits = int(tracker_config.get("min_hits", 1))
    iou_threshold = float(tracker_config.get("iou_threshold", 0.3))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in detections:
        if float(row.get("score", 0.0)) < min_confidence:
            continue
        key = (row["sequence_id"], row["class_id"])
        grouped.setdefault(key, []).append(row)

    output_rows: list[dict[str, str]] = []
    next_track_id = 1
    for (sequence_id, class_id), rows in sorted(grouped.items()):
        active: list[TrackState] = []
        frames: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            frames.setdefault(row["frame_id"], []).append(row)
        for frame_id in sorted(frames, key=_frame_number):
            frame_number = _frame_number(frame_id)
            for track in active:
                track.misses = frame_number - track.last_frame_number
            active = [track for track in active if track.misses <= max_age]
            detections = sorted(frames[frame_id], key=lambda item: float(item["score"]), reverse=True)
            matches, unmatched_tracks, unmatched_detections = _assign(active, detections, frame_number, iou_threshold)
            for track_index, det_index in matches:
                track = active[track_index]
                det = detections[det_index]
                track.velocity = _box_velocity(track.last_box, det, max(1, frame_number - track.last_frame_number))
                track.last_frame_number = frame_number
                track.last_box = det
                track.misses = 0
                track.hits += 1
                if track.hits >= min_hits:
                    output_rows.append(_track_row(sequence_id, class_id, frame_id, track.track_id, det))
            for det_index in unmatched_detections:
                det = detections[det_index]
                track = TrackState(next_track_id, frame_number, det)
                next_track_id += 1
                active.append(track)
                if track.hits >= min_hits:
                    output_rows.append(_track_row(sequence_id, class_id, frame_id, track.track_id, det))
            for track_index in unmatched_tracks:
                active[track_index].misses += 1
    _write_csv(output_path, TRACK_FIELDS, output_rows)
    return output_path


def _assign(
    tracks: list[TrackState],
    detections: list[dict[str, str]],
    frame_number: int,
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not tracks or not detections:
        return [], set(range(len(tracks))), set(range(len(detections)))
    cost = np.ones((len(tracks), len(detections)), dtype=float)
    for track_index, track in enumerate(tracks):
        predicted = track.predicted_box(frame_number)
        for det_index, det in enumerate(detections):
            cost[track_index, det_index] = 1.0 - iou(predicted, det)
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))
    for track_index, det_index in zip(row_ind, col_ind, strict=True):
        if 1.0 - cost[track_index, det_index] < iou_threshold:
            continue
        matches.append((track_index, det_index))
        unmatched_tracks.discard(track_index)
        unmatched_detections.discard(det_index)
    return matches, unmatched_tracks, unmatched_detections


def _box_velocity(previous: dict[str, str], current: dict[str, str], delta: int) -> tuple[float, float, float, float]:
    return tuple((float(current[key]) - float(previous[key])) / delta for key in ("x1", "y1", "x2", "y2"))


def _track_row(
    sequence_id: str,
    class_id: str,
    frame_id: str,
    track_id: int,
    det: dict[str, str],
) -> dict[str, str]:
    return {
        "sequence_id": sequence_id,
        "frame_id": frame_id,
        "track_id": str(track_id),
        "class_id": class_id,
        "score": det["score"],
        "x1": det["x1"],
        "y1": det["y1"],
        "x2": det["x2"],
        "y2": det["y2"],
    }


def _frame_number(frame_id: str) -> int:
    try:
        return int(frame_id)
    except ValueError:
        return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
