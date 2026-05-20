from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CARRADA object_id stability in prepared annotations.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--conversion-records")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    output_path = Path(args.output)
    rows = _read_csv(annotations_path)
    conversion_lookup = _conversion_lookup(Path(args.conversion_records)) if args.conversion_records else {}
    rows = [_with_conversion_fields(row, conversion_lookup) for row in rows]
    by_object: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_object[(row["sequence_id"], row["object_id"])].append(row)

    object_reports = []
    issues = []
    for (sequence_id, object_id), object_rows in sorted(by_object.items()):
        frame_ids = sorted(int(row["frame_id"]) for row in object_rows)
        class_counts = Counter(row["class_id"] for row in object_rows)
        raw_counts = Counter(row["raw_label"] for row in object_rows)
        bbox_sources = Counter(row["bbox_source"] for row in object_rows)
        gaps = _gaps(frame_ids)
        report = {
            "sequence_id": sequence_id,
            "object_id": object_id,
            "frame_count": len(frame_ids),
            "first_frame": f"{frame_ids[0]:06d}",
            "last_frame": f"{frame_ids[-1]:06d}",
            "class_ids": dict(class_counts),
            "raw_labels": dict(raw_counts),
            "bbox_sources": dict(bbox_sources),
            "continuous": not gaps,
            "gap_count": len(gaps),
            "gaps": gaps[:20],
        }
        if len(class_counts) > 1:
            issues.append({"type": "class_id_changes", **report})
        if len(raw_counts) > 1:
            issues.append({"type": "raw_label_changes", **report})
        object_reports.append(report)

    summary = {
        "status": "pass" if not issues else "warning",
        "annotations_path": str(annotations_path),
        "num_annotation_rows": len(rows),
        "num_sequences": len({row["sequence_id"] for row in rows}),
        "num_objects": len(object_reports),
        "objects_with_class_id_changes": sum(1 for item in object_reports if len(item["class_ids"]) > 1),
        "objects_with_raw_label_changes": sum(1 for item in object_reports if len(item["raw_labels"]) > 1),
        "objects_with_frame_gaps": sum(1 for item in object_reports if item["gap_count"] > 0),
        "interpretation": (
            "No class/raw-label changes were found for the same sequence_id/object_id in the prepared annotations. "
            "Frame gaps are reported as visibility gaps, not identity instability by themselves."
        ),
    }
    output = {"summary": summary, "issues": issues, "objects": object_reports}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not issues else 1


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _conversion_lookup(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (row["sequence_id"], row["frame_id"], row["object_id"]): row
        for row in _read_csv(path)
    }


def _with_conversion_fields(row: dict[str, str], lookup: dict[tuple[str, str, str], dict[str, str]]) -> dict[str, str]:
    enriched = dict(row)
    conversion = lookup.get((row["sequence_id"], row["frame_id"], row["object_id"]), {})
    enriched["raw_label"] = conversion.get("raw_label", row.get("raw_label", "unknown"))
    enriched["bbox_source"] = conversion.get("bbox_source", row.get("bbox_source", "unknown"))
    return enriched


def _gaps(frame_ids: list[int]) -> list[dict[str, str | int]]:
    gaps = []
    for previous, current in zip(frame_ids, frame_ids[1:], strict=False):
        if current - previous > 1:
            gaps.append(
                {
                    "after_frame": f"{previous:06d}",
                    "before_frame": f"{current:06d}",
                    "missing_frames": current - previous - 1,
                }
            )
    return gaps


if __name__ == "__main__":
    raise SystemExit(main())
