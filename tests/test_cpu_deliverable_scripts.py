from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_read_split_ignores_blank_lines(tmp_path):
    split_path = tmp_path / "test.txt"
    split_path.write_text("a\n\nb\n", encoding="utf-8")
    run_sort_sweep = _load_script("run_sort_sweep")

    assert run_sort_sweep._read_split(split_path) == {"a", "b"}


def test_showcase_helpers_read_real_summary_shape(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "detection_source",
                "split",
                "num_frames",
                "precision",
                "recall",
                "f1",
                "map50",
                "mota",
                "idf1_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "detection_source": "gt_bbox",
                "split": "test",
                "num_frames": "264",
                "precision": "1.000000",
                "recall": "1.000000",
                "f1": "1.000000",
                "map50": "1.000000",
                "mota": "0.520000",
                "idf1_status": "unavailable",
            }
        )
    failure_report = {
        "id_switch_examples": {"status": "unavailable"},
        "fragmentation_examples": {"status": "unavailable"},
    }
    make_showcase_slides = _load_script("make_showcase_slides")

    summary = make_showcase_slides._read_summary(summary_path)

    assert summary["detection_source"] == "gt_bbox"
    assert make_showcase_slides._fmt(summary["mota"]) == "0.520"
    assert make_showcase_slides._status(failure_report["id_switch_examples"]) == "unavailable"
    assert make_showcase_slides._status(json.loads("{}")) == "unavailable"


def test_showcase_colorize_ra_returns_rgb_image():
    make_showcase_slides = _load_script("make_showcase_slides")
    gray = Image.new("L", (2, 1))
    gray.putpixel((0, 0), 0)
    gray.putpixel((1, 0), 255)

    colorized = make_showcase_slides._colorize_ra(gray)

    assert colorized.mode == "RGB"
    assert colorized.getpixel((0, 0)) != colorized.getpixel((1, 0))


def test_showcase_tracking_frame_selection_uses_same_track_id():
    make_showcase_slides = _load_script("make_showcase_slides")
    rows = [
        {"sequence_id": "s", "frame_id": "000001", "track_id": "a", "x1": "0", "y1": "0", "x2": "2", "y2": "2"},
        {"sequence_id": "s", "frame_id": "000002", "track_id": "a", "x1": "0", "y1": "0", "x2": "2", "y2": "2"},
        {"sequence_id": "s", "frame_id": "000003", "track_id": "a", "x1": "0", "y1": "0", "x2": "2", "y2": "2"},
        {"sequence_id": "s", "frame_id": "000004", "track_id": "a", "x1": "0", "y1": "0", "x2": "2", "y2": "2"},
        {"sequence_id": "s", "frame_id": "000001", "track_id": "b", "x1": "0", "y1": "0", "x2": "4", "y2": "4"},
    ]

    frames = make_showcase_slides._select_tracking_frames(rows, limit=4)

    assert [frame["frame_id"] for frame in frames] == ["000001", "000002", "000003", "000004"]


def test_sort_sweep_variants_override_base_and_write_csv(tmp_path):
    run_sort_sweep = _load_script("run_sort_sweep")
    variants = run_sort_sweep._variants({"max_age": 9, "min_hits": 9, "iou_threshold": 0.9, "min_confidence": 0.25})

    assert variants[0]["max_age"] == 1
    assert variants[0]["min_hits"] == 1
    assert variants[0]["iou_threshold"] == 0.10
    assert variants[0]["min_confidence"] == 0.25
    assert len({(item["max_age"], item["min_hits"], item["iou_threshold"]) for item in variants}) == len(variants)

    csv_path = tmp_path / "out" / "summary.csv"
    run_sort_sweep._write_csv(csv_path, ["name", "mota"], [{"name": "a", "mota": "0.500"}])

    assert csv_path.read_text(encoding="utf-8").splitlines() == ["name,mota", "a,0.500"]


def test_identity_audit_helpers_enrich_conversion_fields():
    audit_identity = _load_script("audit_identity_stability")

    assert audit_identity._gaps([1, 2, 5]) == [
        {"after_frame": "000002", "before_frame": "000005", "missing_frames": 2}
    ]

    row = {"sequence_id": "seq", "frame_id": "000001", "object_id": "obj", "class_id": "2"}
    lookup = {
        ("seq", "000001", "obj"): {
            "raw_label": "3",
            "bbox_source": "range_angle.box",
        }
    }

    enriched = audit_identity._with_conversion_fields(row, lookup)

    assert enriched["raw_label"] == "3"
    assert enriched["bbox_source"] == "range_angle.box"
    assert row.get("raw_label") is None
