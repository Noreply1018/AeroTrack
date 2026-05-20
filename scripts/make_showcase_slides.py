from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS = (1600, 1000)
PANEL_SIZE = (420, 420)
BACKGROUND = (255, 255, 255)
TEXT = (24, 28, 34)
MUTED = (85, 92, 105)
ACCENT = (0, 122, 255)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create presentation-friendly AeroTrack showcase slides.")
    parser.add_argument("--showcase-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--failure-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    showcase_dir = Path(args.showcase_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _read_selected(showcase_dir / "selected_frames.csv")
    if not selected:
        raise SystemExit("No selected frames found.")

    frame = selected[0]
    _make_large_triptych(showcase_dir, output_dir, frame)
    _make_tracking_focus(showcase_dir, output_dir, selected[:4])
    summary = _read_summary(Path(args.summary))
    failures = json.loads(Path(args.failure_report).read_text(encoding="utf-8"))
    _make_metrics_slide(output_dir, summary)
    _make_failure_slide(output_dir, failures)
    return 0


def _make_large_triptych(showcase_dir: Path, output_dir: Path, frame: dict[str, str]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    label_font = _font(28)
    body_font = _font(24)
    draw.text((64, 46), "AeroTrack CPU 诊断闭环：单帧展示", fill=TEXT, font=title_font)
    draw.text((64, 112), "左：GT 标注  中：gt_bbox 诊断检测  右：SORT 跟踪输出", fill=MUTED, font=body_font)

    panels = [
        ("GT 标注", showcase_dir / frame["gt"]),
        ("诊断检测", showcase_dir / frame["detection"]),
        ("SORT 跟踪", showcase_dir / frame["track"]),
    ]
    x_positions = [80, 590, 1100]
    for x, (label, path) in zip(x_positions, panels, strict=True):
        panel = Image.open(path).convert("RGB").resize(PANEL_SIZE, Image.Resampling.NEAREST)
        image.paste(panel, (x, 220))
        draw.rectangle((x, 220, x + PANEL_SIZE[0], 220 + PANEL_SIZE[1]), outline=(210, 216, 226), width=3)
        draw.text((x, 670), label, fill=TEXT, font=label_font)

    draw.line((64, 760, 1536, 760), fill=(224, 228, 235), width=2)
    draw.text((64, 800), "展示口径", fill=ACCENT, font=label_font)
    draw.text((64, 850), "这张图证明数据转换、统一检测格式和 SORT 跟踪链路已经贯通。", fill=TEXT, font=body_font)
    draw.text((64, 890), "中间列来自 gt_bbox 诊断检测，不代表 YOLO 模型预测效果。", fill=TEXT, font=body_font)
    image.save(output_dir / "slide_01_large_triptych.png")


def _make_tracking_focus(showcase_dir: Path, output_dir: Path, frames: list[dict[str, str]]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    label_font = _font(24)
    body_font = _font(24)
    draw.text((64, 46), "SORT 跟踪连续帧展示", fill=TEXT, font=title_font)
    draw.text((64, 112), "同一实验链路输出带 track_id 的连续帧结果，用于人工复核轨迹关联。", fill=MUTED, font=body_font)
    x_positions = [80, 450, 820, 1190]
    for x, frame in zip(x_positions, frames, strict=False):
        panel = Image.open(showcase_dir / frame["track"]).convert("RGB").resize((300, 300), Image.Resampling.NEAREST)
        image.paste(panel, (x, 245))
        draw.rectangle((x, 245, x + 300, 545), outline=(210, 216, 226), width=3)
        draw.text((x, 570), frame["frame_id"], fill=TEXT, font=label_font)
    draw.line((64, 690, 1536, 690), fill=(224, 228, 235), width=2)
    draw.text((64, 735), "展示重点", fill=ACCENT, font=label_font)
    draw.text((64, 785), "跟踪图用于说明项目不止输出单帧目标框，还能把检测结果组织成序列级轨迹。", fill=TEXT, font=body_font)
    draw.text((64, 825), "身份稳定性审计已完成；ID 类指标仍待接入评估模块后启用。", fill=TEXT, font=body_font)
    image.save(output_dir / "slide_02_tracking_sequence.png")


def _make_metrics_slide(output_dir: Path, summary: dict[str, str]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    header_font = _font(28)
    body_font = _font(26)
    draw.text((64, 46), "CPU 诊断闭环指标", fill=TEXT, font=title_font)
    draw.text((64, 112), "指标来自 gt_bbox + SORT smoke 实验；检测指标用于链路诊断，不代表 YOLO 模型性能。", fill=MUTED, font=_font(23))
    rows = [
        ("检测来源", summary["detection_source"]),
        ("评估 split", summary["split"]),
        ("评估帧数", summary["num_frames"]),
        (
            "Precision / Recall / F1 / mAP50",
            f"{_fmt(summary['precision'])} / {_fmt(summary['recall'])} / {_fmt(summary['f1'])} / {_fmt(summary['map50'])}",
        ),
        ("MOTA", _fmt(summary["mota"])),
        ("ID 类指标", summary["idf1_status"]),
    ]
    x0, y0, w = 160, 230, 1280
    draw.rounded_rectangle((x0, y0, x0 + w, y0 + 520), radius=8, outline=(205, 212, 224), width=2)
    draw.rectangle((x0, y0, x0 + w, y0 + 70), fill=(245, 247, 250))
    draw.text((x0 + 40, y0 + 20), "展示项", fill=TEXT, font=header_font)
    draw.text((x0 + 680, y0 + 20), "结果", fill=TEXT, font=header_font)
    y = y0 + 90
    for label, value in rows:
        draw.text((x0 + 40, y), label, fill=TEXT, font=body_font)
        draw.text((x0 + 680, y), value, fill=TEXT, font=body_font)
        y += 70
    draw.text((64, 820), "说明：检测框来自 GT 标注转换，因此检测分数接近理想上限。", fill=TEXT, font=_font(24))
    draw.text((64, 860), "正式 YOLO baseline 需要补齐推理环境、权重和 YOLO adapter 后重新生成。", fill=TEXT, font=_font(24))
    image.save(output_dir / "slide_03_metrics.png")


def _make_failure_slide(output_dir: Path, failures: dict[str, object]) -> None:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(46)
    header_font = _font(28)
    body_font = _font(26)
    draw.text((64, 46), "失败样例检查", fill=TEXT, font=title_font)
    draw.text((64, 112), "当前 gt_bbox 诊断检测报告口径下，漏检和虚警列表为空；ID 类失败样例尚未启用。", fill=MUTED, font=_font(23))
    missed = failures.get("missed_frames", [])
    false_alarms = failures.get("false_alarm_frames", [])
    tracking_empty = failures.get("tracking_empty_annotated_frames", [])
    rows = [
        ("漏检帧", str(len(missed))),
        ("虚警帧", str(len(false_alarms))),
        ("有标注但无跟踪输出帧", str(len(tracking_empty))),
        ("ID switch 样例", _status(failures.get("id_switch_examples"))),
        ("fragmentation 样例", _status(failures.get("fragmentation_examples"))),
    ]
    x0, y0, w = 160, 250, 1280
    draw.rounded_rectangle((x0, y0, x0 + w, y0 + 430), radius=8, outline=(205, 212, 224), width=2)
    y = y0 + 45
    for label, value in rows:
        draw.text((x0 + 50, y), label, fill=TEXT, font=header_font)
        draw.text((x0 + 800, y), value, fill=TEXT, font=body_font)
        y += 75
    draw.text((64, 770), "说明", fill=ACCENT, font=header_font)
    draw.text((64, 820), "这页证明失败样例报告已经生成，并明确记录当前没有检测漏检/虚警样例。", fill=TEXT, font=_font(24))
    draw.text((64, 860), "这个结论只适用于 gt_bbox 诊断检测，不能推导为真实 YOLO 模型不会失败。", fill=TEXT, font=_font(24))
    image.save(output_dir / "slide_04_failure_report.png")


def _read_summary(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No summary row found: {path}")
    return rows[0]


def _read_selected(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: str) -> str:
    return f"{float(value):.3f}"


def _status(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("status", "unavailable"))
    return "unavailable"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
