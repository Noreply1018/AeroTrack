from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CANVAS = (1600, 1000)
BACKGROUND = (255, 255, 255)
TEXT = (24, 28, 34)
MUTED = (85, 92, 105)
ACCENT = (0, 122, 255)
GT = (0, 210, 80)
DET = (238, 184, 0)
TRACK = (0, 170, 255)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create presentation-friendly AeroTrack showcase slides.")
    parser.add_argument("--showcase-dir", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--failure-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    prepared_root = Path(args.prepared_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = _read_csv(Path(args.annotations))
    detections = _read_csv(Path(args.detections))
    tracks = _read_csv(Path(args.tracks))
    primary_frame = _select_showcase_frames(detections, limit=1)[0]
    tracking_frames = _select_tracking_frames(tracks, limit=4)
    summary = _read_summary(Path(args.summary))
    failures = json.loads(Path(args.failure_report).read_text(encoding="utf-8"))

    _make_large_triptych(prepared_root, output_dir, primary_frame, annotations, detections, tracks)
    _make_tracking_focus(prepared_root, output_dir, tracking_frames, tracks)
    _make_metrics_slide(output_dir, summary)
    _make_failure_slide(output_dir, failures)
    _write_selected_frames(Path(args.showcase_dir) / "selected_frames.csv", [primary_frame, *tracking_frames])
    return 0


def _make_large_triptych(
    prepared_root: Path,
    output_dir: Path,
    frame: dict[str, str],
    annotations: list[dict[str, str]],
    detections: list[dict[str, str]],
    tracks: list[dict[str, str]],
) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    label_font = _font(27)
    body_font = _font(23)
    draw.text((64, 44), "AeroTrack CPU Diagnostic Loop", fill=TEXT, font=title_font)
    draw.text((64, 108), "Full Range-Angle frame plus zoomed target region. Detector source: gt_bbox, not YOLO.", fill=MUTED, font=body_font)

    panels = [
        ("GT annotation", _rows_for_frame(annotations, frame), GT),
        ("gt_bbox detection", _rows_for_frame(detections, frame), DET),
        ("SORT track output", _rows_for_frame(tracks, frame), TRACK),
    ]
    source = _source_image(prepared_root, frame)
    target_box = _primary_box(_rows_for_frame(detections, frame))
    x_positions = [66, 566, 1066]
    for x, (label, rows, color) in zip(x_positions, panels, strict=True):
        full = source.copy()
        _draw_boxes(full, rows, color)
        full = full.resize((300, 300), Image.Resampling.NEAREST)
        zoom = _crop_zoom(source, target_box, rows, color)
        image.paste(full, (x, 210))
        image.paste(zoom, (x, 540))
        draw.rectangle((x, 210, x + 300, 510), outline=(205, 212, 224), width=2)
        draw.rectangle((x, 540, x + 420, 900), outline=color, width=4)
        draw.text((x, 166), label, fill=TEXT, font=label_font)
        draw.text((x, 914), _panel_detail(label, rows), fill=MUTED, font=_font(18))

    draw.text((64, 950), "Use this slide to explain the data -> diagnostic detection -> tracking handoff. Boxes are enlarged for readability.", fill=MUTED, font=_font(20))
    image.save(output_dir / "slide_01_large_triptych.png")


def _make_tracking_focus(prepared_root: Path, output_dir: Path, frames: list[dict[str, str]], tracks: list[dict[str, str]]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    label_font = _font(22)
    body_font = _font(22)
    draw.text((64, 44), "SORT Tracking Across Frames", fill=TEXT, font=title_font)
    draw.text((64, 108), "Zoomed target views make the track boxes and IDs readable in presentation.", fill=MUTED, font=body_font)
    x_positions = [70, 450, 830, 1210]
    for x, frame in zip(x_positions, frames, strict=False):
        source = _source_image(prepared_root, frame)
        rows = _rows_for_frame(tracks, frame)
        box = _primary_box(rows)
        zoom = _crop_zoom(source, box, rows, TRACK)
        image.paste(zoom.resize((300, 300), Image.Resampling.NEAREST), (x, 245))
        draw.rectangle((x, 245, x + 300, 545), outline=TRACK, width=3)
        draw.text((x, 575), f"Frame {frame['frame_id']}", fill=TEXT, font=label_font)
        draw.text((x, 605), _panel_detail("SORT track output", rows), fill=MUTED, font=_font(18))
    draw.line((64, 700, 1536, 700), fill=(224, 228, 235), width=2)
    draw.text((64, 748), "What to say", fill=ACCENT, font=_font(28))
    draw.text((64, 800), "SORT consumes the same detections frame by frame and emits track IDs for sequence-level review.", fill=TEXT, font=body_font)
    draw.text((64, 840), "Identity stability has been audited; IDF1/ID-switch/fragmentation metrics still need evaluation wiring.", fill=TEXT, font=body_font)
    image.save(output_dir / "slide_02_tracking_sequence.png")


def _make_metrics_slide(output_dir: Path, summary: dict[str, str]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    header_font = _font(28)
    body_font = _font(26)
    draw.text((64, 46), "CPU Diagnostic Metrics", fill=TEXT, font=title_font)
    draw.text((64, 112), "Metrics are from gt_bbox + SORT smoke diagnostics. They verify the pipeline, not YOLO quality.", fill=MUTED, font=_font(23))
    rows = [
        ("Detector source", summary["detection_source"]),
        ("Evaluation split", summary["split"]),
        ("Evaluation frames", summary["num_frames"]),
        (
            "Precision / Recall / F1 / mAP50",
            f"{_fmt(summary['precision'])} / {_fmt(summary['recall'])} / {_fmt(summary['f1'])} / {_fmt(summary['map50'])}",
        ),
        ("MOTA", _fmt(summary["mota"])),
        ("ID metrics", summary["idf1_status"]),
    ]
    _draw_table(draw, rows, (160, 230), (1280, 520), header_font, body_font)
    draw.text((64, 820), "Note: detection boxes are converted from GT annotations, so detection scores are expected to be ideal.", fill=TEXT, font=_font(24))
    draw.text((64, 860), "A real YOLO baseline still requires inference dependencies, weights, and a YOLO adapter.", fill=TEXT, font=_font(24))
    image.save(output_dir / "slide_03_metrics.png")


def _make_failure_slide(output_dir: Path, failures: dict[str, object]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    header_font = _font(28)
    body_font = _font(26)
    draw.text((64, 46), "Failure Report Check", fill=TEXT, font=title_font)
    draw.text((64, 112), "For gt_bbox diagnostics, missed and false-alarm frame lists are empty; ID failure examples are not enabled.", fill=MUTED, font=_font(23))
    rows = [
        ("Missed frames", str(len(failures.get("missed_frames", [])))),
        ("False-alarm frames", str(len(failures.get("false_alarm_frames", [])))),
        ("Annotated frames without tracks", str(len(failures.get("tracking_empty_annotated_frames", [])))),
        ("ID switch examples", _status(failures.get("id_switch_examples"))),
        ("Fragmentation examples", _status(failures.get("fragmentation_examples"))),
    ]
    _draw_table(draw, rows, (160, 250), (1280, 430), header_font, body_font)
    draw.text((64, 770), "Boundary", fill=ACCENT, font=header_font)
    draw.text((64, 820), "This proves the failure report is generated and records no detection misses/false alarms in this diagnostic run.", fill=TEXT, font=_font(24))
    draw.text((64, 860), "It does not mean a real YOLO detector would have zero failures.", fill=TEXT, font=_font(24))
    image.save(output_dir / "slide_04_failure_report.png")


def _draw_boxes(image: Image.Image, rows: list[dict[str, str]], color: tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    for row in rows:
        box = _box(row)
        draw.rectangle(box, outline=color, width=2)


def _crop_zoom(
    source: Image.Image,
    focus_box: tuple[float, float, float, float],
    rows: list[dict[str, str]],
    color: tuple[int, int, int],
) -> Image.Image:
    x1, y1, x2, y2 = focus_box
    width, height = source.size
    pad = 34
    crop = (
        max(0, int(x1) - pad),
        max(0, int(y1) - pad),
        min(width, int(x2) + pad),
        min(height, int(y2) + pad),
    )
    region = source.crop(crop)
    shifted = []
    for row in rows:
        r = dict(row)
        bx1, by1, bx2, by2 = _box(row)
        r.update({"x1": str(bx1 - crop[0]), "y1": str(by1 - crop[1]), "x2": str(bx2 - crop[0]), "y2": str(by2 - crop[1])})
        shifted.append(r)
    _draw_boxes(region, shifted, color)
    return region.resize((420, 360), Image.Resampling.NEAREST)


def _draw_table(
    draw: ImageDraw.ImageDraw,
    rows: list[tuple[str, str]],
    origin: tuple[int, int],
    size: tuple[int, int],
    header_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    w, h = size
    draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=8, outline=(205, 212, 224), width=2)
    draw.rectangle((x0, y0, x0 + w, y0 + 70), fill=(245, 247, 250))
    draw.text((x0 + 40, y0 + 20), "Item", fill=TEXT, font=header_font)
    draw.text((x0 + 680, y0 + 20), "Result", fill=TEXT, font=header_font)
    y = y0 + 90
    for label, value in rows:
        draw.text((x0 + 40, y), label, fill=TEXT, font=body_font)
        draw.text((x0 + 680, y), value, fill=TEXT, font=body_font)
        y += 70


def _select_showcase_frames(detections: list[dict[str, str]], *, limit: int) -> list[dict[str, str]]:
    ranked = sorted(detections, key=lambda row: (_area(row), row["sequence_id"], row["frame_id"]), reverse=True)
    selected = []
    seen = set()
    for row in ranked:
        key = (row["sequence_id"], row["frame_id"])
        if key in seen:
            continue
        selected.append({"sequence_id": row["sequence_id"], "frame_id": row["frame_id"]})
        seen.add(key)
        if len(selected) == limit:
            break
    return selected


def _select_tracking_frames(tracks: list[dict[str, str]], *, limit: int) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in tracks:
        groups.setdefault((row["sequence_id"], row["track_id"]), []).append(row)
    best_run: list[dict[str, str]] = []
    best_area = -1.0
    for rows in groups.values():
        for run in _consecutive_runs(sorted(rows, key=lambda row: int(row["frame_id"]))):
            if len(run) < limit:
                continue
            area = max(_area(row) for row in run)
            if len(run) > len(best_run) or (len(run) == len(best_run) and area > best_area):
                best_run = run
                best_area = area
    if not best_run:
        best_run = sorted(tracks, key=lambda row: _area(row), reverse=True)[:limit]
    window = max((best_run[index : index + limit] for index in range(0, max(1, len(best_run) - limit + 1))), key=lambda rows: max(_area(row) for row in rows))
    return [{"sequence_id": row["sequence_id"], "frame_id": row["frame_id"]} for row in window]


def _consecutive_runs(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    runs: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    previous: int | None = None
    for row in rows:
        frame = int(row["frame_id"])
        if previous is None or frame == previous + 1:
            current.append(row)
        else:
            runs.append(current)
            current = [row]
        previous = frame
    if current:
        runs.append(current)
    return runs


def _panel_detail(label: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return "no object in selected frame"
    row = max(rows, key=_area)
    class_name = {"0": "pedestrian", "1": "cyclist", "2": "car"}.get(row.get("class_id", ""), f"class {row.get('class_id', '?')}")
    if "track" in label:
        return f"track ID {row.get('track_id', '?')} / {class_name}"
    if row.get("score"):
        return f"{class_name} / score {float(row['score']):.2f}"
    return class_name


def _source_image(prepared_root: Path, frame: dict[str, str]) -> Image.Image:
    gray = Image.open(prepared_root / "images" / frame["sequence_id"] / f"{frame['frame_id']}.png").convert("L")
    return _colorize_ra(gray)


def _colorize_ra(gray: Image.Image) -> Image.Image:
    enhanced = ImageOps.autocontrast(gray)
    return ImageOps.colorize(enhanced, black="#101820", mid="#2364aa", white="#fff2a8")


def _rows_for_frame(rows: list[dict[str, str]], frame: dict[str, str]) -> list[dict[str, str]]:
    return [row for row in rows if row["sequence_id"] == frame["sequence_id"] and row["frame_id"] == frame["frame_id"]]


def _primary_box(rows: list[dict[str, str]]) -> tuple[float, float, float, float]:
    if not rows:
        return (96.0, 96.0, 160.0, 160.0)
    return _box(max(rows, key=_area))


def _box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def _area(row: dict[str, str]) -> float:
    x1, y1, x2, y2 = _box(row)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if not rows:
        raise SystemExit(f"No summary row found: {path}")
    return rows[0]


def _write_selected_frames(path: Path, frames: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "frame_id"])
        writer.writeheader()
        writer.writerows(frames)


def _fmt(value: str) -> str:
    return f"{float(value):.3f}"


def _status(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("status", "unavailable"))
    return "unavailable"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    ]:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
