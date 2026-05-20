from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from aerotrack.config import write_yaml
from aerotrack.contracts import ANNOTATION_FIELDS, CONVERSION_RECORD_FIELDS, SAMPLE_INDEX_FIELDS


SPLIT_NAMES = {"train": "train", "validation": "val", "val": "val", "test": "test"}


@dataclass(frozen=True)
class PreparedData:
    root: Path
    sample_index_path: Path
    annotations_path: Path
    classes_path: Path
    conversion_records_path: Path
    split_paths: dict[str, Path]
    num_sequences: int
    num_frames: int


def prepare_carrada_ra_smoke(config: dict[str, Any]) -> PreparedData:
    repo_root = Path(config.get("repo_root", ".")).resolve()
    dataset = config["dataset"]
    carrada_root = carrada_dataset_root(_repo_path(repo_root, dataset.get("root", "data/carrada")))
    prepared_root = _repo_path(repo_root, dataset.get("prepared_root", "data/processed/carrada_ra_smoke"))
    _ensure_carrada_files(carrada_root)

    annotations = json.loads((carrada_root / "annotations_instance_oriented.json").read_text(encoding="utf-8"))
    seq_ref = json.loads((carrada_root / "data_seq_ref.json").read_text(encoding="utf-8"))
    classes = _classes(dataset)
    raw_mapping = {int(k): int(v) for k, v in dataset.get("raw_label_mapping", {1: 0, 2: 1, 3: 2}).items()}
    selected = _select_sequences(seq_ref, int(dataset.get("splits", {}).get("max_sequences", 2)))

    _make_dirs(prepared_root)
    sample_rows: list[dict[str, str]] = []
    annotation_rows: list[dict[str, str]] = []
    conversion_rows: list[dict[str, str]] = []
    split_samples: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for sequence_id, split in selected:
        seq_dir = carrada_root / sequence_id
        image_sources = sorted((seq_dir / "range_angle_processed").glob("*.npy"))
        object_labels = _object_raw_labels(seq_ref[sequence_id])
        frame_annotations = _frame_annotations(
            sequence_id,
            annotations.get(sequence_id, {}),
            object_labels,
            raw_mapping,
            classes,
        )
        for npy_path in image_sources:
            frame_id = npy_path.stem
            sample_id = f"{sequence_id}_{frame_id}"
            rel_image = Path("images") / sequence_id / f"{frame_id}.png"
            rel_label = Path("labels") / sequence_id / f"{frame_id}.txt"
            image_path = prepared_root / rel_image
            label_path = prepared_root / rel_label
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            _write_png(npy_path, image_path)
            frame_rows = frame_annotations.get(frame_id, [])
            _write_yolo_label(label_path, frame_rows, image_path)
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "split": split,
                    "representation": "range_angle",
                    "image_path": str(rel_image),
                    "label_path": str(rel_label),
                }
            )
            split_samples[split].append(sample_id)
            for row in frame_rows:
                annotation_rows.append({field: row[field] for field in ANNOTATION_FIELDS})
                conversion_rows.append(
                    {
                        "sample_id": sample_id,
                        "sequence_id": sequence_id,
                        "frame_id": frame_id,
                        "object_id": row["object_id"],
                        "raw_label": row["raw_label"],
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "bbox_source": row["bbox_source"],
                        "image_source": str(npy_path.relative_to(carrada_root)),
                        "image_path": str(rel_image),
                        "label_path": str(rel_label),
                        "notes": row.get("notes", ""),
                    }
                )

    _write_csv(prepared_root / "sample_index.csv", SAMPLE_INDEX_FIELDS, sample_rows)
    _write_csv(prepared_root / "annotations.csv", ANNOTATION_FIELDS, annotation_rows)
    _write_csv(prepared_root / "conversion_records.csv", CONVERSION_RECORD_FIELDS, conversion_rows)
    write_yaml(prepared_root / "classes.yaml", {"classes": [{"id": k, "name": v} for k, v in classes.items()]})
    split_paths = _write_splits(prepared_root, split_samples)
    _render_gt_checks(prepared_root, annotation_rows, classes, max_images=20)
    return PreparedData(
        root=prepared_root,
        sample_index_path=prepared_root / "sample_index.csv",
        annotations_path=prepared_root / "annotations.csv",
        classes_path=prepared_root / "classes.yaml",
        conversion_records_path=prepared_root / "conversion_records.csv",
        split_paths=split_paths,
        num_sequences=len(selected),
        num_frames=len(sample_rows),
    )


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def carrada_dataset_root(root: Path) -> Path:
    if (root / "annotations_instance_oriented.json").exists() and (root / "data_seq_ref.json").exists():
        return root
    return root / "Carrada"


def _ensure_carrada_files(root: Path) -> None:
    required = [root / "annotations_instance_oriented.json", root / "data_seq_ref.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CARRADA files: {', '.join(missing)}")


def _make_dirs(root: Path) -> None:
    for rel in [
        "images",
        "labels",
        "splits",
        "visual_checks/gt",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def _classes(dataset: dict[str, Any]) -> dict[int, str]:
    return {int(item["id"]): str(item["name"]) for item in dataset.get("classes", [])}


def _select_sequences(seq_ref: dict[str, Any], max_sequences: int) -> list[tuple[str, str]]:
    entries = [(sid, SPLIT_NAMES[str(meta.get("split", "")).lower()]) for sid, meta in seq_ref.items()]
    entries = [(sid, split) for sid, split in entries if split in {"train", "val", "test"}]
    if max_sequences <= 0:
        return entries
    selected: list[tuple[str, str]] = []
    for wanted in ("train", "test"):
        for entry in entries:
            if entry[1] == wanted and entry not in selected:
                selected.append(entry)
                break
        if len(selected) >= max_sequences:
            return selected
    for entry in entries:
        if entry not in selected:
            selected.append(entry)
        if len(selected) >= max_sequences:
            break
    return selected


def _object_raw_labels(seq_meta: dict[str, Any]) -> dict[str, int]:
    instances = [str(item) for item in seq_meta.get("instances", [])]
    labels = [int(item) for item in seq_meta.get("labels", [])]
    return dict(zip(instances, labels, strict=False))


def _frame_annotations(
    sequence_id: str,
    sequence_annotations: dict[str, Any],
    object_labels: dict[str, int],
    raw_mapping: dict[int, int],
    classes: dict[int, str],
) -> dict[str, list[dict[str, str]]]:
    by_frame: dict[str, list[dict[str, str]]] = {}
    for object_id, object_frames in sequence_annotations.items():
        raw_label = object_labels.get(str(object_id))
        if raw_label is None:
            continue
        class_id = raw_mapping.get(raw_label)
        if class_id is None:
            continue
        for frame_id, representations in object_frames.items():
            ra = representations.get("range_angle", {})
            bbox, source = _bbox_from_range_angle(ra)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            by_frame.setdefault(str(frame_id), []).append(
                {
                    "sequence_id": sequence_id,
                    "frame_id": str(frame_id),
                    "object_id": str(object_id),
                    "class_id": str(class_id),
                    "x1": f"{x1:.3f}",
                    "y1": f"{y1:.3f}",
                    "x2": f"{x2:.3f}",
                    "y2": f"{y2:.3f}",
                    "raw_label": str(raw_label),
                    "class_name": classes.get(class_id, str(class_id)),
                    "bbox_source": source,
                    "notes": "",
                }
            )
    return by_frame


def _bbox_from_range_angle(ra: dict[str, Any]) -> tuple[tuple[float, float, float, float] | None, str]:
    box = ra.get("box")
    if box and len(box) == 2:
        return (float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1])), "range_angle.box"
    dense = ra.get("dense")
    if dense:
        xs = [float(point[0]) for point in dense]
        ys = [float(point[1]) for point in dense]
        return (min(xs), min(ys), max(xs), max(ys)), "range_angle.dense_bbox"
    return None, "unavailable"


def _write_png(source: Path, target: Path) -> None:
    array = np.load(source)
    clean = np.nan_to_num(array.astype("float32"), copy=False)
    min_value = float(clean.min())
    max_value = float(clean.max())
    if max_value > min_value:
        scaled = (clean - min_value) / (max_value - min_value)
    else:
        scaled = np.zeros_like(clean, dtype="float32")
    image = (scaled * 255.0).clip(0, 255).astype("uint8")
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(target)


def _write_yolo_label(path: Path, rows: list[dict[str, str]], image_path: Path) -> None:
    width, height = Image.open(image_path).size
    lines: list[str] = []
    for row in rows:
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = max(0.0, x2 - x1) / width
        bh = max(0.0, y2 - y1) / height
        lines.append(f"{row['class_id']} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_splits(root: Path, split_samples: dict[str, list[str]]) -> dict[str, Path]:
    split_paths: dict[str, Path] = {}
    for split, samples in split_samples.items():
        path = root / "splits" / f"{split}.txt"
        path.write_text("\n".join(samples) + ("\n" if samples else ""), encoding="utf-8")
        split_paths[split] = path
    return split_paths


def _render_gt_checks(root: Path, annotation_rows: list[dict[str, str]], classes: dict[int, str], max_images: int) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in annotation_rows:
        grouped.setdefault((row["sequence_id"], row["frame_id"]), []).append(row)
    for index, ((sequence_id, frame_id), rows) in enumerate(sorted(grouped.items())):
        if index >= max_images:
            break
        source = root / "images" / sequence_id / f"{frame_id}.png"
        if not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        for row in rows:
            xy = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            draw.rectangle(xy, outline=(0, 255, 0), width=2)
            draw.text((xy[0], max(0.0, xy[1] - 10)), classes.get(int(row["class_id"]), row["class_id"]), fill=(0, 255, 0))
        target = root / "visual_checks" / "gt" / f"{sequence_id}_{frame_id}.png"
        image.save(target)
