from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import numpy as np
except ImportError:  # pragma: no cover - the PNG fallback keeps the script usable.
    np = None


CANVAS = (1600, 1000)
BACKGROUND = (255, 255, 255)
TEXT = (24, 28, 34)
MUTED = (85, 92, 105)
ACCENT = (0, 122, 255)
GT = (0, 210, 80)
DET = (238, 184, 0)
TRACK = (0, 170, 255)
PANEL_BORDER = (205, 212, 224)
GRID = (54, 68, 84)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create presentation-friendly AeroTrack showcase slides.")
    parser.add_argument("--showcase-dir", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--failure-report", required=True)
    parser.add_argument("--cpu10-summary")
    parser.add_argument("--sort-sweep-summary")
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
    cpu10_summary = _read_summary(Path(args.cpu10_summary)) if args.cpu10_summary else None
    sort_sweep = _read_csv(Path(args.sort_sweep_summary)) if args.sort_sweep_summary else []

    source_lookup = _read_source_lookup(prepared_root)

    _make_large_triptych(prepared_root, output_dir, primary_frame, annotations, detections, tracks, source_lookup)
    _make_tracking_focus(prepared_root, output_dir, tracking_frames, tracks, source_lookup)
    _make_metrics_slide(output_dir, summary)
    _make_failure_slide(output_dir, failures)
    _make_single_target_slide(prepared_root, output_dir, primary_frame, annotations, detections, tracks, source_lookup)
    _make_track_strip_slide(prepared_root, output_dir, tracking_frames, tracks, source_lookup)
    if sort_sweep:
        _make_sort_sweep_slide(output_dir, sort_sweep)
    if cpu10_summary:
        _make_scale_comparison_slide(output_dir, summary, cpu10_summary)
    _write_selected_frames(Path(args.showcase_dir) / "selected_frames.csv", [primary_frame, *tracking_frames])
    return 0


def _make_large_triptych(
    prepared_root: Path,
    output_dir: Path,
    frame: dict[str, str],
    annotations: list[dict[str, str]],
    detections: list[dict[str, str]],
    tracks: list[dict[str, str]],
    source_lookup: dict[tuple[str, str], str],
) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    label_font = _font(27)
    body_font = _font(23)
    draw.text((64, 44), "AeroTrack CPU Diagnostic Loop", fill=TEXT, font=title_font)
    draw.text((64, 108), "High-contrast Range-Angle render with readable target zooms. Detector source: gt_bbox, not YOLO.", fill=MUTED, font=body_font)

    panels = [
        ("GT annotation", _rows_for_frame(annotations, frame), GT),
        ("gt_bbox detection", _rows_for_frame(detections, frame), DET),
        ("SORT track output", _rows_for_frame(tracks, frame), TRACK),
    ]
    source = _source_image(prepared_root, frame, source_lookup)
    target_box = _primary_box(_rows_for_frame(detections, frame))
    x_positions = [66, 566, 1066]
    for x, (label, rows, color) in zip(x_positions, panels, strict=True):
        full = _render_panel(source, rows, (340, 340), color)
        zoom = _crop_zoom(source, target_box, rows, color, output_size=(420, 360))
        image.paste(full, (x, 210))
        image.paste(zoom, (x, 565))
        draw.rectangle((x, 210, x + 340, 550), outline=PANEL_BORDER, width=2)
        draw.rectangle((x, 565, x + 420, 925), outline=color, width=4)
        draw.text((x, 166), label, fill=TEXT, font=label_font)
        draw.text((x, 934), _panel_detail(label, rows), fill=MUTED, font=_font(18))

    draw.text((64, 950), "Use this slide to explain the data -> diagnostic detection -> tracking handoff. Zooms are rendered from the source frame.", fill=MUTED, font=_font(20))
    image.save(output_dir / "slide_01_large_triptych.png")


def _make_tracking_focus(
    prepared_root: Path,
    output_dir: Path,
    frames: list[dict[str, str]],
    tracks: list[dict[str, str]],
    source_lookup: dict[tuple[str, str], str],
) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    label_font = _font(22)
    body_font = _font(22)
    draw.text((64, 44), "SORT Tracking Across Frames", fill=TEXT, font=title_font)
    draw.text((64, 108), "Zoomed target views make the track boxes and IDs readable in presentation.", fill=MUTED, font=body_font)
    x_positions = [70, 450, 830, 1210]
    for x, frame in zip(x_positions, frames, strict=False):
        source = _source_image(prepared_root, frame, source_lookup)
        rows = _rows_for_frame(tracks, frame)
        box = _primary_box(rows)
        zoom = _crop_zoom(source, box, rows, TRACK, output_size=(300, 300))
        image.paste(zoom, (x, 245))
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


def _make_single_target_slide(
    prepared_root: Path,
    output_dir: Path,
    frame: dict[str, str],
    annotations: list[dict[str, str]],
    detections: list[dict[str, str]],
    tracks: list[dict[str, str]],
    source_lookup: dict[tuple[str, str], str],
) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((64, 44), "Single Target Detail", fill=TEXT, font=_font(46))
    draw.text((64, 108), "A readable target inspection view: full RA context, magnified RA target, and camera reference.", fill=MUTED, font=_font(23))
    source = _source_image(prepared_root, frame, source_lookup)
    rows = [
        ("GT", _rows_for_frame(annotations, frame), GT),
        ("Detection", _rows_for_frame(detections, frame), DET),
        ("Track", _rows_for_frame(tracks, frame), TRACK),
    ]
    target_box = _primary_box(rows[1][1])
    full = _render_panel(source, [], (430, 430), ACCENT)
    display_layers = [
        (_rows_for_frame(tracks, frame), TRACK, 6),
        (_rows_for_frame(detections, frame), DET, 2),
        (_rows_for_frame(annotations, frame), GT, -2),
    ]
    for layer_rows, color, expand in display_layers:
        scaled = [_expanded_row(_scaled_row(row, 430 / source.width, 430 / source.height), expand) for row in layer_rows]
        _draw_boxes(full, scaled, color, width=4)
    image.paste(full, (80, 220))
    draw.rectangle((80, 220, 510, 650), outline=PANEL_BORDER, width=2)
    zoom = _crop_zoom_layers(source, target_box, display_layers, output_size=(560, 430))
    image.paste(zoom, (560, 220))
    draw.rectangle((560, 220, 1120, 650), outline=ACCENT, width=4)
    camera = _camera_image(prepared_root, frame)
    if camera:
        camera = ImageOps.contain(camera.convert("RGB"), (360, 270), Image.Resampling.LANCZOS)
        camera_panel = Image.new("RGB", (360, 270), (245, 247, 250))
        camera_panel.paste(camera, ((360 - camera.width) // 2, (270 - camera.height) // 2))
        image.paste(camera_panel, (1180, 220))
        draw.rectangle((1180, 220, 1540, 490), outline=PANEL_BORDER, width=2)
        draw.text((1180, 508), "Camera reference", fill=MUTED, font=_font(19))
    else:
        draw.rectangle((1180, 220, 1540, 490), outline=PANEL_BORDER, width=2)
        draw.text((1215, 340), "Camera reference unavailable", fill=MUTED, font=_font(19))

    draw.text((80, 675), "Full RA frame", fill=TEXT, font=_font(23))
    draw.text((560, 675), "Magnified target region", fill=TEXT, font=_font(23))
    legend_y = 720
    for label, _, color in rows:
        draw.rectangle((560, legend_y, 590, legend_y + 20), fill=color)
        draw.text((605, legend_y - 4), label, fill=TEXT, font=_font(23))
        legend_y += 45
    draw.text((80, 790), f"Frame {frame['sequence_id']} / {frame['frame_id']}", fill=TEXT, font=_font(24))
    draw.text((80, 842), "RA color map is clipped and contrast-enhanced for presentation only.", fill=MUTED, font=_font(21))
    draw.text((80, 874), "Detection and tracking boxes still use the original CARRADA Range-Angle coordinates.", fill=MUTED, font=_font(21))
    image.save(output_dir / "slide_05_single_target_detail.png")


def _make_track_strip_slide(
    prepared_root: Path,
    output_dir: Path,
    frames: list[dict[str, str]],
    tracks: list[dict[str, str]],
    source_lookup: dict[tuple[str, str], str],
) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((64, 44), "Track Strip: Same ID Over Time", fill=TEXT, font=_font(46))
    draw.text((64, 108), "A compact strip view for explaining sequence-level tracking without crowding the slide.", fill=MUTED, font=_font(23))
    extended = _extend_track_window(tracks, frames, limit=6)
    x = 70
    for frame in extended:
        source = _source_image(prepared_root, frame, source_lookup)
        rows = _rows_for_frame(tracks, frame)
        zoom = _crop_zoom(source, _primary_box(rows), rows, TRACK, output_size=(220, 260))
        image.paste(zoom, (x, 260))
        draw.rectangle((x, 260, x + 220, 520), outline=TRACK, width=3)
        draw.text((x, 545), frame["frame_id"], fill=TEXT, font=_font(21))
        draw.text((x, 575), _panel_detail("SORT track output", rows), fill=MUTED, font=_font(17))
        x += 245
    draw.line((64, 700, 1536, 700), fill=(224, 228, 235), width=2)
    draw.text((64, 752), "Use this slide when you need more than four frames but still want each target box readable.", fill=TEXT, font=_font(24))
    image.save(output_dir / "slide_06_track_strip.png")


def _make_sort_sweep_slide(output_dir: Path, rows: list[dict[str, str]]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((64, 44), "SORT Parameter Sweep", fill=TEXT, font=_font(46))
    draw.text((64, 108), "MOTA changes with association threshold and track confirmation settings on the CPU diagnostic run.", fill=MUTED, font=_font(23))
    chart = (160, 230, 1380, 690)
    _draw_bar_chart(
        draw,
        [(f"age {r['max_age']}\nhit {r['min_hits']}\niou {r['iou_threshold']}", float(r["mota"])) for r in rows],
        chart,
        "MOTA",
        TRACK,
    )
    draw.text((160, 855), "Interpretation: with gt_bbox detections fixed, changes mainly reflect SORT association behavior.", fill=TEXT, font=_font(24))
    image.save(output_dir / "slide_07_sort_sweep.png")


def _make_scale_comparison_slide(output_dir: Path, smoke: dict[str, str], cpu10: dict[str, str]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((64, 44), "CPU Scale-Up Check", fill=TEXT, font=_font(46))
    draw.text((64, 108), "The same diagnostic loop was also run on a larger 10-sequence processed set.", fill=MUTED, font=_font(23))
    values = [
        ("test frames", float(smoke["num_frames"]), float(cpu10["num_frames"])),
        ("MOTA", float(smoke["mota"]), float(cpu10["mota"])),
    ]
    x0, y0 = 180, 250
    for idx, (label, smoke_value, cpu_value) in enumerate(values):
        y = y0 + idx * 240
        max_value = max(smoke_value, cpu_value, 1.0)
        draw.text((x0, y), label, fill=TEXT, font=_font(28))
        _bar_pair(draw, x0 + 260, y + 5, smoke_value, cpu_value, max_value)
    legend_x = 1160
    draw.rectangle((legend_x, 250, legend_x + 30, 280), fill=DET)
    draw.text((legend_x + 45, 245), "smoke", fill=TEXT, font=_font(24))
    draw.rectangle((legend_x, 295, legend_x + 30, 325), fill=TRACK)
    draw.text((legend_x + 45, 290), "cpu10", fill=TEXT, font=_font(24))
    draw.text((180, 780), "Detection metrics remain ideal because the source is gt_bbox.", fill=TEXT, font=_font(22))
    draw.text((180, 815), "The scale-up proves CPU data conversion and archiving capacity.", fill=TEXT, font=_font(22))
    image.save(output_dir / "slide_08_scale_comparison.png")


def _draw_boxes(image: Image.Image, rows: list[dict[str, str]], color: tuple[int, int, int], *, width: int = 2) -> None:
    draw = ImageDraw.Draw(image)
    for row in rows:
        box = _box(row)
        draw.rectangle(box, outline=color, width=width)


def _draw_grid(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for x in range(width // 4, width, width // 4):
        draw.line((x, 0, x, height), fill=(*GRID, 60), width=1)
    for y in range(height // 4, height, height // 4):
        draw.line((0, y, width, y), fill=(*GRID, 60), width=1)


def _render_panel(
    source: Image.Image,
    rows: list[dict[str, str]],
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> Image.Image:
    original_w, original_h = source.size
    panel = source.resize(size, Image.Resampling.BICUBIC)
    _draw_grid(panel)
    scaled_rows = [_scaled_row(row, size[0] / original_w, size[1] / original_h) for row in rows]
    _draw_boxes(panel, scaled_rows, color, width=4)
    return panel


def _crop_zoom(
    source: Image.Image,
    focus_box: tuple[float, float, float, float],
    rows: list[dict[str, str]],
    color: tuple[int, int, int],
    *,
    output_size: tuple[int, int] = (420, 360),
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
    zoom = region.resize(output_size, Image.Resampling.BICUBIC)
    _draw_grid(zoom)
    scale_x = output_size[0] / max(1, region.width)
    scale_y = output_size[1] / max(1, region.height)
    _draw_boxes(zoom, [_scaled_row(row, scale_x, scale_y) for row in shifted], color, width=5)
    return zoom


def _crop_zoom_layers(
    source: Image.Image,
    focus_box: tuple[float, float, float, float],
    layers: list[tuple[list[dict[str, str]], tuple[int, int, int], int]],
    *,
    output_size: tuple[int, int],
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
    zoom = region.resize(output_size, Image.Resampling.BICUBIC)
    _draw_grid(zoom)
    scale_x = output_size[0] / max(1, region.width)
    scale_y = output_size[1] / max(1, region.height)
    for rows, color, expand in layers:
        shifted = []
        for row in rows:
            r = dict(row)
            bx1, by1, bx2, by2 = _box(row)
            r.update(
                {
                    "x1": str(bx1 - crop[0]),
                    "y1": str(by1 - crop[1]),
                    "x2": str(bx2 - crop[0]),
                    "y2": str(by2 - crop[1]),
                }
            )
            shifted.append(r)
        display_rows = [_expanded_row(_scaled_row(row, scale_x, scale_y), expand) for row in shifted]
        _draw_boxes(zoom, display_rows, color, width=5)
    return zoom


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


def _draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    values: list[tuple[str, float]],
    box: tuple[int, int, int, int],
    ylabel: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    draw.line((x1, y2, x2, y2), fill=(170, 178, 190), width=2)
    draw.line((x1, y1, x1, y2), fill=(170, 178, 190), width=2)
    max_value = max(value for _, value in values) or 1.0
    bar_width = int((x2 - x1) / max(1, len(values)) * 0.55)
    gap = int((x2 - x1) / max(1, len(values)))
    for index, (label, value) in enumerate(values):
        cx = x1 + index * gap + gap // 2
        height = int((y2 - y1 - 40) * value / max_value)
        draw.rectangle((cx - bar_width // 2, y2 - height, cx + bar_width // 2, y2), fill=color)
        draw.text((cx - 38, y2 - height - 32), f"{value:.3f}", fill=TEXT, font=_font(18))
        lines = label.split("\n")
        for offset, line in enumerate(lines):
            draw.text((cx - 42, y2 + 12 + offset * 18), line, fill=MUTED, font=_font(14))
    draw.text((x1 - 72, y1 + 10), ylabel, fill=MUTED, font=_font(20))


def _bar_pair(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    smoke_value: float,
    cpu_value: float,
    max_value: float,
) -> None:
    width = 560
    smoke_w = int(width * smoke_value / max_value)
    cpu_w = int(width * cpu_value / max_value)
    draw.rectangle((x, y, x + smoke_w, y + 45), fill=DET)
    draw.text((x + smoke_w + 15, y + 8), f"{smoke_value:.3g}", fill=TEXT, font=_font(22))
    draw.rectangle((x, y + 65, x + cpu_w, y + 110), fill=TRACK)
    draw.text((x + cpu_w + 15, y + 73), f"{cpu_value:.3g}", fill=TEXT, font=_font(22))


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


def _extend_track_window(tracks: list[dict[str, str]], frames: list[dict[str, str]], *, limit: int) -> list[dict[str, str]]:
    if not frames:
        return []
    first_rows = _rows_for_frame(tracks, frames[0])
    if not first_rows:
        return frames[:limit]
    track_id = first_rows[0]["track_id"]
    sequence_id = frames[0]["sequence_id"]
    same_track = sorted(
        [row for row in tracks if row["sequence_id"] == sequence_id and row["track_id"] == track_id],
        key=lambda row: int(row["frame_id"]),
    )
    start = int(frames[0]["frame_id"])
    window = [row for row in same_track if int(row["frame_id"]) >= start][:limit]
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


def _read_source_lookup(prepared_root: Path) -> dict[tuple[str, str], str]:
    records_path = prepared_root / "conversion_records.csv"
    if not records_path.exists():
        return {}
    lookup: dict[tuple[str, str], str] = {}
    for row in _read_csv(records_path):
        lookup[(row["sequence_id"], row["frame_id"])] = row.get("image_source", "")
    return lookup


def _source_image(prepared_root: Path, frame: dict[str, str], source_lookup: dict[tuple[str, str], str]) -> Image.Image:
    npy_path = _npy_source_path(prepared_root, frame, source_lookup)
    if npy_path is not None and np is not None:
        return _colorize_ra_array(np.load(npy_path))
    gray = Image.open(prepared_root / "images" / frame["sequence_id"] / f"{frame['frame_id']}.png").convert("L")
    return _colorize_ra(gray)


def _npy_source_path(prepared_root: Path, frame: dict[str, str], source_lookup: dict[tuple[str, str], str]) -> Path | None:
    rel = source_lookup.get((frame["sequence_id"], frame["frame_id"]))
    candidates: list[Path] = []
    if rel:
        candidates.append(prepared_root.parents[1] / "carrada" / "Carrada" / rel)
    candidates.extend(
        [
            prepared_root.parents[1]
            / "carrada"
            / "Carrada"
            / frame["sequence_id"]
            / folder
            / f"{frame['frame_id']}.npy"
            for folder in ("range_angle_processed", "range_angle_numpy", "range_angle_raw")
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _camera_image(prepared_root: Path, frame: dict[str, str]) -> Image.Image | None:
    camera_path = (
        prepared_root.parents[1]
        / "carrada"
        / "Carrada"
        / frame["sequence_id"]
        / "camera_images"
        / f"{frame['frame_id']}.jpg"
    )
    if not camera_path.exists():
        return None
    return Image.open(camera_path)


def _colorize_ra(gray: Image.Image) -> Image.Image:
    enhanced = ImageOps.autocontrast(gray, cutoff=1)
    return ImageOps.colorize(enhanced, black="#101820", mid="#2364aa", white="#fff2a8")


def _colorize_ra_array(array: "np.ndarray") -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    values = values[np.isfinite(values)] if np.isfinite(values).any() else values
    if values.size == 0:
        return Image.new("RGB", (256, 256), "#101820")
    low, high = np.percentile(values, [2.0, 99.5])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    scaled = np.clip((np.asarray(array, dtype=np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)
    scaled = np.sqrt(scaled)
    gray = Image.fromarray((scaled * 255.0).astype("uint8"), mode="L")
    return ImageOps.colorize(gray, black="#101820", mid="#1f7a8c", white="#fff4b8")


def _rows_for_frame(rows: list[dict[str, str]], frame: dict[str, str]) -> list[dict[str, str]]:
    return [row for row in rows if row["sequence_id"] == frame["sequence_id"] and row["frame_id"] == frame["frame_id"]]


def _primary_box(rows: list[dict[str, str]]) -> tuple[float, float, float, float]:
    if not rows:
        return (96.0, 96.0, 160.0, 160.0)
    return _box(max(rows, key=_area))


def _box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def _scaled_row(row: dict[str, str], scale_x: float, scale_y: float) -> dict[str, str]:
    scaled = dict(row)
    x1, y1, x2, y2 = _box(row)
    scaled.update(
        {
            "x1": str(x1 * scale_x),
            "y1": str(y1 * scale_y),
            "x2": str(x2 * scale_x),
            "y2": str(y2 * scale_y),
        }
    )
    return scaled


def _expanded_row(row: dict[str, str], pixels: float) -> dict[str, str]:
    expanded = dict(row)
    x1, y1, x2, y2 = _box(row)
    expanded.update(
        {
            "x1": str(x1 - pixels),
            "y1": str(y1 - pixels),
            "x2": str(x2 + pixels),
            "y2": str(y2 + pixels),
        }
    )
    return expanded


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
