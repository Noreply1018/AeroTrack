from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


TEXT = (24, 28, 34)
MUTED = (85, 92, 105)
ACCENT = (0, 122, 255)
DET = (238, 184, 0)
GT = (0, 210, 80)
TRACK = (0, 170, 255)
GRID = (54, 68, 84)
PANEL_BORDER = (205, 212, 224)
BACKGROUND = (247, 250, 252)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PPT-ready AeroTrack assets.")
    parser.add_argument("--ppt-dir", default="ppt")
    parser.add_argument("--showcase-slides-dir", default="runs/carrada_ra_gtbbox_sort_smoke/showcase/slides")
    parser.add_argument("--yolo-run-dir", default="runs/yolo_final_demo/carrada_ra_cpu10_yolov8n_e30_cpu")
    parser.add_argument("--yolo-pred-dir", default="runs/yolo_final_demo/carrada_ra_cpu10_showcase_pred")
    parser.add_argument("--data-dir", default="data/processed/carrada_ra_cpu10")
    parser.add_argument("--cpu10-summary", default="final/tables/cpu10_summary.csv")
    parser.add_argument("--server30-summary", default="final/tables/server30_summary.csv")
    parser.add_argument("--sort-sweep-summary", default="final/tables/cpu10_sort_sweep_summary.csv")
    args = parser.parse_args()

    ppt_dir = Path(args.ppt_dir)
    assets_dir = ppt_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    showcase_dir = Path(args.showcase_slides_dir)
    yolo_run_dir = Path(args.yolo_run_dir)
    yolo_pred_dir = Path(args.yolo_pred_dir)
    data_dir = Path(args.data_dir)

    _copy_image(assets_dir / "01_ra_detection_tracking_overview.png", showcase_dir / "slide_01_large_triptych.png")
    _copy_image(assets_dir / "02_gt_visual_check.png", _pick_final_gt_visual_check(Path("final/figures/data_conversion")))
    _make_labels_distribution(assets_dir / "03_labels_distribution.jpg", data_dir)
    _copy_image(assets_dir / "04_yolo_training_results.png", _crop_yolo_results(yolo_run_dir / "results.png"))
    _copy_image(assets_dir / "05_yolo_pr_curve.png", _crop_pr_curve(yolo_run_dir / "BoxPR_curve.png"))
    _make_yolo_prediction_overview(assets_dir / "06_yolo_prediction_overview.png", yolo_pred_dir)
    _copy_image(assets_dir / "07_tracking_sequence.png", showcase_dir / "slide_02_tracking_sequence.png")
    _copy_image(assets_dir / "08_track_strip.png", showcase_dir / "slide_06_track_strip.png")
    _make_sort_sweep(assets_dir / "09_sort_sweep.png", Path(args.sort_sweep_summary))
    _make_scale_comparison(assets_dir / "10_scale_comparison.png", Path(args.cpu10_summary), Path(args.server30_summary))
    _copy_image(assets_dir / "11_failure_report.png", showcase_dir / "slide_04_failure_report.png")
    _copy_image(assets_dir / "12_prediction_sample.jpg", _pick_final_prediction_sample(Path("final/figures/yolo_predictions")))
    _copy_image(assets_dir / "13_tracking_sample.png", showcase_dir / "slide_05_single_target_detail.png")
    return 0


def _copy_image(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _pick_gt_visual_check(data_dir: Path) -> Path:
    candidates = sorted((data_dir / "visual_checks" / "gt").glob("*.png"))
    if not candidates:
        raise FileNotFoundError("No GT visual check images found")
    return candidates[0]


def _pick_final_gt_visual_check(final_dir: Path) -> Path:
    candidates = sorted(final_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError("No final GT visual check images found")
    return candidates[0]


def _pick_prediction_sample(yolo_pred_dir: Path) -> Path:
    candidates = sorted(yolo_pred_dir.glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError("No YOLO predictions found")
    return candidates[0]


def _pick_final_prediction_sample(final_dir: Path) -> Path:
    candidates = sorted(final_dir.glob("prediction_*.jpg"))
    if not candidates:
        raise FileNotFoundError("No final YOLO prediction samples found")
    return candidates[0]


def _crop_yolo_results(source: Path) -> Path:
    image = Image.open(source).convert("RGB")
    canvas = Image.new("RGB", (1600, 900), "white")
    fitted = ImageOps.contain(image, (1500, 820), Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((1600 - fitted.width) // 2, (900 - fitted.height) // 2))
    return _save_temp(canvas, "yolo_results_crop.png")


def _crop_pr_curve(source: Path) -> Path:
    image = Image.open(source).convert("RGB")
    canvas = Image.new("RGB", (1600, 900), "white")
    fitted = ImageOps.contain(image, (1500, 820), Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((1600 - fitted.width) // 2, (900 - fitted.height) // 2))
    return _save_temp(canvas, "yolo_pr_curve_crop.png")


def _make_labels_distribution(target: Path, data_dir: Path) -> None:
    rows = _read_csv(data_dir / "annotations.csv")
    counts = Counter(row["class_id"] for row in rows)
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((64, 44), "Label distribution", fill=TEXT, font=_font(44))
    draw.text((64, 108), "Current CPU10 subset is dominated by car targets; this is useful to state explicitly in the talk.", fill=MUTED, font=_font(23))
    bars = [("car", counts.get("2", 0)), ("pedestrian", counts.get("0", 0)), ("cyclist", counts.get("1", 0))]
    max_value = max((value for _, value in bars), default=1)
    x0, y0 = 170, 240
    for index, (label, value) in enumerate(bars):
        y = y0 + index * 180
        width = int(980 * value / max_value)
        draw.text((x0 - 120, y + 10), label, fill=TEXT, font=_font(30))
        draw.rectangle((x0, y, x0 + width, y + 52), fill=[DET, TRACK, GT][index])
        draw.text((x0 + width + 18, y + 10), str(value), fill=TEXT, font=_font(26))
    draw.text((64, 760), "This slide should be used to explain class imbalance, not as a decorative backup.", fill=TEXT, font=_font(24))
    image.save(target)


def _make_sort_sweep(target: Path, summary_path: Path) -> None:
    rows = _read_csv(summary_path)
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((64, 44), "SORT parameter sweep", fill=TEXT, font=_font(44))
    draw.text((64, 108), "GT-driven diagnostics show how IOU, max_age, and min_hits shift MOTA.", fill=MUTED, font=_font(23))
    values = [(f"age={row['max_age']} hit={row['min_hits']} iou={row['iou_threshold']}", float(row["mota"])) for row in rows]
    _draw_bar_chart(draw, values, (160, 220, 1380, 650), TRACK)
    draw.text((160, 760), "Use single-line labels so the slide stays readable on a projector.", fill=TEXT, font=_font(24))
    image.save(target)


def _make_scale_comparison(target: Path, cpu10_path: Path, server30_path: Path) -> None:
    rows = [_read_csv(cpu10_path)[0], _read_csv(server30_path)[0]]
    smoke_path = cpu10_path.parent.parent / "smoke_summary.csv"
    if smoke_path.exists():
        rows.insert(0, _read_csv(smoke_path)[0])
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((64, 44), "Scale comparison", fill=TEXT, font=_font(44))
    draw.text((64, 108), "Smoke, CPU10, and server30 share the same gt_bbox + SORT diagnostic loop.", fill=MUTED, font=_font(23))
    y0 = 220
    max_frames = max(float(row["num_frames"]) for row in rows)
    max_mota = max(float(row["mota"]) for row in rows)
    for idx, row in enumerate(rows):
        y = y0 + idx * 185
        label = _scale_label(row)
        draw.text((160, y + 6), label, fill=TEXT, font=_font(26))
        draw.rectangle((360, y, 360 + int(840 * float(row["num_frames"]) / max_frames), y + 38), fill=[DET, TRACK, (42, 183, 169)][idx])
        draw.text((1215, y + 4), str(int(float(row["num_frames"]))), fill=TEXT, font=_font(24))
        draw.rectangle((360, y + 74, 360 + int(840 * float(row["mota"]) / max_mota), y + 112), fill=[DET, TRACK, (42, 183, 169)][idx])
        draw.text((1215, y + 78), f"{float(row['mota']):.3f}".rstrip("0").rstrip("."), fill=TEXT, font=_font(24))
    draw.text((160, 760), "The point here is scale and archiving capacity, not end-to-end detector accuracy.", fill=TEXT, font=_font(24))
    image.save(target)


def _make_yolo_prediction_overview(target: Path, yolo_pred_dir: Path) -> None:
    candidates = []
    for source in sorted(yolo_pred_dir.glob("*.jpg")):
        label_path = yolo_pred_dir / "labels" / f"{source.stem}.txt"
        if label_path.exists():
            boxes = _read_yolo_boxes(label_path, Image.open(source).size)
            candidates.append((len(boxes), source, label_path))
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    selected = candidates[:6]
    if not selected:
        raise FileNotFoundError("No prediction samples found")
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    tile_w, tile_h = 500, 320
    x_positions = [64, 550, 1036]
    y_positions = [70, 430]
    for index, (_, source, label_path) in enumerate(selected):
        x = x_positions[index % 3]
        y = y_positions[index // 3]
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        src = Image.open(source).convert("RGB")
        boxes = _read_yolo_boxes(label_path, src.size)
        _draw_boxes(src, boxes, DET, width=3)
        fitted = ImageOps.contain(src, (tile_w, 260), Image.Resampling.LANCZOS)
        tile.paste(fitted, ((tile_w - fitted.width) // 2, 0))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((12, 268), source.name, fill=TEXT, font=_font(16))
        tile_draw.text((12, 292), f"{len(boxes)} boxes", fill=MUTED, font=_font(14))
        image.paste(tile, (x, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline=PANEL_BORDER, width=2)
    draw.text((64, 22), "YOLO prediction overview", fill=TEXT, font=_font(34))
    draw.text((64, 54), "Selected labeled predictions for presentation use.", fill=MUTED, font=_font(20))
    image.save(target)


def _draw_bar_chart(draw: ImageDraw.ImageDraw, values: list[tuple[str, float]], box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    draw.line((x1, y2, x2, y2), fill=(170, 178, 190), width=2)
    draw.line((x1, y1, x1, y2), fill=(170, 178, 190), width=2)
    max_value = max((value for _, value in values), default=1.0)
    bar_gap = (y2 - y1) // max(1, len(values))
    for index, (label, value) in enumerate(values):
        y = y1 + index * bar_gap + 18
        width = int((x2 - x1 - 180) * value / max_value)
        draw.text((x1 - 5, y - 4), label, fill=MUTED, font=_font(18))
        draw.rectangle((x1 + 160, y, x1 + 160 + width, y + 34), fill=color)
        draw.text((x1 + 170 + width, y - 2), f"{value:.3f}".rstrip("0").rstrip("."), fill=TEXT, font=_font(20))


def _draw_boxes(image: Image.Image, boxes: list[tuple[float, float, float, float]], color: tuple[int, int, int], *, width: int) -> None:
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=color, width=width)


def _scale_label(row: dict[str, str]) -> str:
    name = row.get("experiment_name", "").lower()
    if "server30" in name:
        return "server30"
    if "cpu10" in name:
        return "CPU10"
    if "smoke" in name:
        return "smoke"
    return "run"


def _read_yolo_boxes(label_path: Path, size: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    width, height = size
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = map(float, parts[:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _save_temp(image: Image.Image, name: str) -> Path:
    path = Path("/tmp") / f"aerotrack_{name}"
    image.save(path)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
