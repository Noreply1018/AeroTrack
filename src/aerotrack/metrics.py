from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import motmetrics as mm

from aerotrack.contracts import ANNOTATION_FIELDS, DETECTION_FIELDS, TRACK_FIELDS


UNAVAILABLE_METRIC = {
    "status": "unavailable",
    "value": None,
    "reason": "ID metrics are disabled until IDF1, ID switches, and fragmentation calculations are wired into evaluation.",
}


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    map50: float
    tp: int
    fp: int
    fn: int


def generate_gt_detections(annotations_path: Path, output_path: Path, score: float) -> Path:
    rows = _read_csv(annotations_path)
    detections = [
        {
            "sequence_id": row["sequence_id"],
            "frame_id": row["frame_id"],
            "class_id": row["class_id"],
            "score": f"{score:.6f}",
            "x1": row["x1"],
            "y1": row["y1"],
            "x2": row["x2"],
            "y2": row["y2"],
        }
        for row in rows
    ]
    _write_csv(output_path, DETECTION_FIELDS, detections)
    return output_path


def evaluate_detection(
    annotations_path: Path,
    detections_path: Path,
    output_path: Path,
    *,
    split_sample_ids: set[str],
    iou_threshold: float,
) -> DetectionMetrics:
    gt_rows = _filter_split(_read_csv(annotations_path), split_sample_ids)
    det_rows = _filter_split(_read_csv(detections_path), split_sample_ids)
    tp, fp, fn = match_detection_counts(gt_rows, det_rows, iou_threshold=iou_threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    map50 = mean_average_precision_50(gt_rows, det_rows, iou_threshold=iou_threshold)
    metrics = DetectionMetrics(precision, recall, f1, map50, tp, fp, fn)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": metrics.map50,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou_threshold": iou_threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def mean_average_precision_50(
    gt_rows: list[dict[str, str]],
    det_rows: list[dict[str, str]],
    *,
    iou_threshold: float,
) -> float:
    class_ids = sorted({row["class_id"] for row in gt_rows} | {row["class_id"] for row in det_rows})
    if not class_ids:
        return 0.0
    aps = []
    for class_id in class_ids:
        class_gt = [row for row in gt_rows if row["class_id"] == class_id]
        class_det = [row for row in det_rows if row["class_id"] == class_id]
        aps.append(_average_precision(class_gt, class_det, iou_threshold))
    return sum(aps) / len(aps)


def _average_precision(gt_rows: list[dict[str, str]], det_rows: list[dict[str, str]], iou_threshold: float) -> float:
    if not gt_rows:
        return 0.0
    gt_by_frame: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in gt_rows:
        gt_by_frame[(row["sequence_id"], row["frame_id"])].append(row)
    matched: dict[tuple[str, str], set[int]] = defaultdict(set)
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for det in sorted(det_rows, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        key = (det["sequence_id"], det["frame_id"])
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(gt_by_frame.get(key, [])):
            if index in matched[key]:
                continue
            score = iou(det, gt)
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[key].add(best_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    if not tp_flags:
        return 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    cum_tp = 0
    cum_fp = 0
    for tp, fp in zip(tp_flags, fp_flags, strict=True):
        cum_tp += tp
        cum_fp += fp
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / len(gt_rows))
    ap = 0.0
    previous_recall = 0.0
    for recall, precision in zip(recalls, precisions, strict=True):
        ap += (recall - previous_recall) * max(p for r, p in zip(recalls, precisions, strict=True) if r >= recall)
        previous_recall = recall
    return ap


def match_detection_counts(
    gt_rows: list[dict[str, str]],
    det_rows: list[dict[str, str]],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    groups: dict[tuple[str, str, str], tuple[list[dict[str, str]], list[dict[str, str]]]] = {}
    for row in gt_rows:
        key = (row["sequence_id"], row["frame_id"], row["class_id"])
        groups.setdefault(key, ([], []))[0].append(row)
    for row in det_rows:
        key = (row["sequence_id"], row["frame_id"], row["class_id"])
        groups.setdefault(key, ([], []))[1].append(row)

    tp = fp = fn = 0
    for gt_group, det_group in groups.values():
        matched_gt: set[int] = set()
        for det in sorted(det_group, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            best_index = -1
            best_iou = 0.0
            for index, gt in enumerate(gt_group):
                if index in matched_gt:
                    continue
                score = iou(det, gt)
                if score > best_iou:
                    best_iou = score
                    best_index = index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched_gt.add(best_index)
                tp += 1
            else:
                fp += 1
        fn += len(gt_group) - len(matched_gt)
    return tp, fp, fn


def evaluate_tracking(
    annotations_path: Path,
    tracks_path: Path,
    output_path: Path,
    *,
    split_sample_ids: set[str],
    iou_threshold: float,
) -> dict[str, Any]:
    gt_rows = _filter_split(_read_csv(annotations_path), split_sample_ids)
    track_rows = _filter_split(_read_csv(tracks_path), split_sample_ids)
    tp, fp, fn = match_detection_counts(gt_rows, _tracks_as_detections(track_rows), iou_threshold=iou_threshold)
    mota = _motmetrics_mota(gt_rows, track_rows, iou_threshold)
    result = {
        "mota": mota,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "idf1": UNAVAILABLE_METRIC,
        "id_switches": UNAVAILABLE_METRIC,
        "track_fragmentation": UNAVAILABLE_METRIC,
        "metadata": {
            "backend": "motmetrics",
            "motmetrics_version": getattr(mm, "__version__", "unknown"),
            "iou_threshold": iou_threshold,
            "grouping": "sequence_id/class_id",
            "id_metrics_status": "unavailable",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _motmetrics_mota(gt_rows: list[dict[str, str]], track_rows: list[dict[str, str]], iou_threshold: float) -> float:
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted({(row["sequence_id"], row["frame_id"]) for row in gt_rows + track_rows})
    gt_id_map: dict[str, int] = {}
    track_id_map: dict[str, int] = {}
    for sequence_id, frame_id in frames:
        gt_frame = [row for row in gt_rows if row["sequence_id"] == sequence_id and row["frame_id"] == frame_id]
        track_frame = [row for row in track_rows if row["sequence_id"] == sequence_id and row["frame_id"] == frame_id]
        gt_ids = [_mapped_id(gt_id_map, f"{row['sequence_id']}:{row['object_id']}") for row in gt_frame]
        track_ids = [_mapped_id(track_id_map, f"{row['sequence_id']}:{row['track_id']}") for row in track_frame]
        distances = mm.distances.iou_matrix(
            [_xyxy_to_xywh(row) for row in gt_frame],
            [_xyxy_to_xywh(row) for row in track_frame],
            max_iou=1.0 - iou_threshold,
        )
        acc.update(gt_ids, track_ids, distances)
    summary = mm.metrics.create().compute(acc, metrics=["mota"], name="smoke")
    value = summary.loc["smoke", "mota"]
    return float(value) if value == value else 0.0


def _mapped_id(mapping: dict[str, int], key: str) -> int:
    if key not in mapping:
        mapping[key] = len(mapping) + 1
    return mapping[key]


def failure_examples(
    annotations_path: Path,
    detections_path: Path,
    tracks_path: Path,
    *,
    split_sample_ids: set[str],
    iou_threshold: float,
) -> dict[str, Any]:
    gt_rows = _filter_split(_read_csv(annotations_path), split_sample_ids)
    det_rows = _filter_split(_read_csv(detections_path), split_sample_ids)
    track_rows = _filter_split(_read_csv(tracks_path), split_sample_ids)
    misses = _missed_frames(gt_rows, det_rows, iou_threshold)
    false_alarms = _false_alarm_frames(gt_rows, det_rows, iou_threshold)
    track_empty = sorted(
        {
            f"{row['sequence_id']}_{row['frame_id']}"
            for row in gt_rows
            if not any(t["sequence_id"] == row["sequence_id"] and t["frame_id"] == row["frame_id"] for t in track_rows)
        }
    )
    return {
        "missed_frames": misses,
        "false_alarm_frames": false_alarms,
        "tracking_empty_annotated_frames": track_empty,
        "id_switch_examples": UNAVAILABLE_METRIC,
        "fragmentation_examples": UNAVAILABLE_METRIC,
    }


def iou(a: dict[str, str], b: dict[str, str]) -> float:
    ax1, ay1, ax2, ay2 = (float(a[key]) for key in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (float(b[key]) for key in ("x1", "y1", "x2", "y2"))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _xyxy_to_xywh(row: dict[str, str]) -> list[float]:
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _filter_split(rows: list[dict[str, str]], sample_ids: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if f"{row['sequence_id']}_{row['frame_id']}" in sample_ids]


def _tracks_as_detections(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{field: row[field] for field in DETECTION_FIELDS} for row in rows]


def _missed_frames(gt_rows: list[dict[str, str]], det_rows: list[dict[str, str]], threshold: float) -> list[str]:
    missed: set[str] = set()
    for gt in gt_rows:
        candidates = [
            det
            for det in det_rows
            if det["sequence_id"] == gt["sequence_id"] and det["frame_id"] == gt["frame_id"] and det["class_id"] == gt["class_id"]
        ]
        if not any(iou(gt, det) >= threshold for det in candidates):
            missed.add(f"{gt['sequence_id']}_{gt['frame_id']}")
    return sorted(missed)


def _false_alarm_frames(gt_rows: list[dict[str, str]], det_rows: list[dict[str, str]], threshold: float) -> list[str]:
    false_alarms: set[str] = set()
    for det in det_rows:
        candidates = [
            gt
            for gt in gt_rows
            if gt["sequence_id"] == det["sequence_id"] and gt["frame_id"] == det["frame_id"] and gt["class_id"] == det["class_id"]
        ]
        if not any(iou(gt, det) >= threshold for gt in candidates):
            false_alarms.add(f"{det['sequence_id']}_{det['frame_id']}")
    return sorted(false_alarms)
