from pathlib import Path

from aerotrack.contracts import (
    CONVERSION_RECORD_FIELDS,
    DETECTION_FIELDS,
    SAMPLE_INDEX_FIELDS,
    invalid_bbox,
    validate_csv_fields,
)


def test_validate_csv_fields_reports_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("sequence_id,frame_id,class_id\n", encoding="utf-8")

    result = validate_csv_fields(csv_path, DETECTION_FIELDS)

    assert not result.ok
    assert "score" in result.missing_fields
    assert "x1" in result.missing_fields


def test_invalid_bbox_detects_negative_area() -> None:
    row = {"x1": "10", "y1": "5", "x2": "9", "y2": "8"}

    assert invalid_bbox(row) == "bbox has negative width or height"


def test_csv_contract_fields_keep_expected_order() -> None:
    assert SAMPLE_INDEX_FIELDS == [
        "sample_id",
        "sequence_id",
        "frame_id",
        "split",
        "representation",
        "image_path",
        "label_path",
    ]
    assert CONVERSION_RECORD_FIELDS == [
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
