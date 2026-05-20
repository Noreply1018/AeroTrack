from __future__ import annotations

import argparse
import csv
import os
import tempfile
import shutil
from typing import NamedTuple
from pathlib import Path


FINAL_DIRS = [
    "figures/data_conversion",
    "figures/diagnostic_pipeline",
    "figures/tracking",
    "figures/yolo_training",
    "figures/yolo_predictions",
    "reports",
    "tables",
    "commands",
]


YOLO_FIGURES = [
    "results.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "labels.jpg",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_labels.jpg",
    "val_batch2_pred.jpg",
    "train_batch0.jpg",
    "train_batch1.jpg",
    "train_batch2.jpg",
]


SHOWCASE_SLIDES = [
    "slide_01_large_triptych.png",
    "slide_02_tracking_sequence.png",
    "slide_03_metrics.png",
    "slide_04_failure_report.png",
    "slide_05_single_target_detail.png",
    "slide_06_track_strip.png",
    "slide_07_sort_sweep.png",
    "slide_08_scale_comparison.png",
]


class CopiedArtifact(NamedTuple):
    section: str
    name: str
    source: Path
    target: Path
    status: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the final AeroTrack presentation package.")
    parser.add_argument("--output-dir", default="final")
    parser.add_argument("--yolo-run-dir", default="runs/yolo_final_demo/carrada_ra_cpu10_yolov8n_e30_cpu")
    parser.add_argument("--yolo-pred-dir", default="runs/yolo_final_demo/carrada_ra_cpu10_showcase_pred")
    parser.add_argument("--smoke-run-dir", default="runs/carrada_ra_gtbbox_sort_smoke")
    parser.add_argument("--cpu10-run-dir", default="runs/carrada_ra_gtbbox_sort_cpu10")
    parser.add_argument("--server30-run-dir", default="runs/carrada_ra_gtbbox_sort_server30")
    parser.add_argument("--cpu10-data-dir", default="data/processed/carrada_ra_cpu10")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    yolo_run_dir = Path(args.yolo_run_dir).resolve()
    yolo_pred_dir = Path(args.yolo_pred_dir).resolve()

    artifacts = build_final_showcase(
        output_dir=output_dir,
        yolo_run_dir=yolo_run_dir,
        yolo_pred_dir=yolo_pred_dir,
        smoke_run_dir=Path(args.smoke_run_dir),
        cpu10_run_dir=Path(args.cpu10_run_dir),
        server30_run_dir=Path(args.server30_run_dir),
        cpu10_data_dir=Path(args.cpu10_data_dir),
    )
    print(f"Final package: {output_dir}")
    print(f"Copied/recorded artifacts: {len(artifacts)}")
    return 0


def build_final_showcase(
    *,
    output_dir: Path,
    yolo_run_dir: Path,
    yolo_pred_dir: Path,
    smoke_run_dir: Path,
    cpu10_run_dir: Path,
    server30_run_dir: Path,
    cpu10_data_dir: Path,
) -> list[CopiedArtifact]:
    yolo_metrics = _read_yolo_metrics(yolo_run_dir / "results.csv")
    diagnostic_summary_paths = {
        "smoke": smoke_run_dir / "metrics" / "summary.csv",
        "cpu10": cpu10_run_dir / "metrics" / "summary.csv",
        "server30": server30_run_dir / "metrics" / "summary.csv",
    }
    diagnostic_rows = _read_diagnostic_summaries(diagnostic_summary_paths)
    sort_sweep_rows = _read_required_csv("sort sweep summary", cpu10_run_dir / "analysis" / "sort_sweep" / "sort_sweep_summary.csv")
    data_summary = _data_summary(cpu10_data_dir)
    _require_non_empty("yolo metrics", yolo_metrics, yolo_run_dir / "results.csv")
    _require_non_empty("data conversion checks", list((cpu10_data_dir / "visual_checks" / "gt").glob("*.png")), cpu10_data_dir / "visual_checks" / "gt")
    _require_non_empty("yolo predictions", list(yolo_pred_dir.glob("*.jpg")), yolo_pred_dir)

    with tempfile.TemporaryDirectory(prefix="aerotrack_final_", dir=output_dir.parent) as tmp_dir:
        staging_dir = Path(tmp_dir)
        for rel in FINAL_DIRS:
            (staging_dir / rel).mkdir(parents=True, exist_ok=True)

        artifacts: list[CopiedArtifact] = []
        artifacts.extend(_copy_data_conversion(cpu10_data_dir, staging_dir))
        artifacts.extend(_copy_diagnostic_materials(smoke_run_dir, cpu10_run_dir, server30_run_dir, staging_dir))
        artifacts.extend(_copy_yolo_materials(yolo_run_dir, yolo_pred_dir, staging_dir))

        _write_csv(staging_dir / "tables" / "yolo_training_metrics.csv", yolo_metrics)
        _write_csv(staging_dir / "tables" / "diagnostic_experiment_summary.csv", diagnostic_rows)
        _write_csv(staging_dir / "tables" / "sort_sweep_summary.csv", sort_sweep_rows)
        _write_csv(staging_dir / "tables" / "data_summary.csv", [data_summary])

        _write_readme(staging_dir, yolo_run_dir, yolo_pred_dir, yolo_metrics, diagnostic_rows, data_summary)
        _write_project_summary(staging_dir, yolo_metrics, diagnostic_rows, data_summary)
        _write_yolo_report(staging_dir, yolo_run_dir, yolo_pred_dir, yolo_metrics)
        _write_showcase_report(staging_dir, diagnostic_rows, data_summary, sort_sweep_rows)
        _write_commands(staging_dir, yolo_run_dir, yolo_pred_dir)

        final_artifacts = _retarget_artifacts(artifacts, staging_dir, output_dir)
        _write_artifact_manifest(staging_dir / "tables" / "artifact_manifest.csv", final_artifacts)
        _replace_directory(output_dir, staging_dir)
        return final_artifacts


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_data_conversion(cpu10_data_dir: Path, output_dir: Path) -> list[CopiedArtifact]:
    artifacts: list[CopiedArtifact] = []
    visual_dir = cpu10_data_dir / "visual_checks" / "gt"
    for index, source in enumerate(sorted(visual_dir.glob("*.png"))[:8], start=1):
        target = output_dir / "figures" / "data_conversion" / f"gt_visual_check_{index:02d}_{source.name}"
        artifacts.append(_copy_file("data_conversion", "GT 可视化抽查", source, target))
    for name in ["sample_index.csv", "annotations.csv", "conversion_records.csv", "classes.yaml"]:
        source = cpu10_data_dir / name
        target = output_dir / "tables" / f"cpu10_{name}"
        artifacts.append(_copy_file("data_conversion", name, source, target))
    return artifacts


def _copy_diagnostic_materials(
    smoke_run_dir: Path,
    cpu10_run_dir: Path,
    server30_run_dir: Path,
    output_dir: Path,
) -> list[CopiedArtifact]:
    artifacts: list[CopiedArtifact] = []
    slides_dir = smoke_run_dir / "showcase" / "slides"
    for slide in SHOWCASE_SLIDES:
        source = slides_dir / slide
        target = output_dir / "figures" / "diagnostic_pipeline" / slide
        artifacts.append(_copy_file("diagnostic_pipeline", slide, source, target))

    for index, source in enumerate(sorted((cpu10_run_dir / "visualizations" / "tracks").glob("*.png"))[:12], start=1):
        target = output_dir / "figures" / "tracking" / f"cpu10_track_{index:02d}_{source.name}"
        artifacts.append(_copy_file("tracking", "CPU10 SORT 跟踪图", source, target))

    summary_sources = {
        "smoke_summary.csv": smoke_run_dir / "metrics" / "summary.csv",
        "cpu10_summary.csv": cpu10_run_dir / "metrics" / "summary.csv",
        "server30_summary.csv": server30_run_dir / "metrics" / "summary.csv",
        "cpu10_sort_sweep_summary.csv": cpu10_run_dir / "analysis" / "sort_sweep" / "sort_sweep_summary.csv",
        "cpu10_identity_stability_audit.json": cpu10_run_dir / "analysis" / "identity_stability_audit.json",
    }
    for target_name, source in summary_sources.items():
        artifacts.append(_copy_file("diagnostic_pipeline", target_name, source, output_dir / "tables" / target_name))
    return artifacts


def _copy_yolo_materials(yolo_run_dir: Path, yolo_pred_dir: Path, output_dir: Path) -> list[CopiedArtifact]:
    artifacts: list[CopiedArtifact] = []
    for name in YOLO_FIGURES:
        source = yolo_run_dir / name
        target = output_dir / "figures" / "yolo_training" / name
        artifacts.append(_copy_file("yolo_training", name, source, target))

    for name in ["results.csv", "args.yaml"]:
        source = yolo_run_dir / name
        artifacts.append(_copy_file("yolo_training", name, source, output_dir / "tables" / f"yolo_{name}"))

    prediction_images = sorted(yolo_pred_dir.glob("*.jpg"))
    _require_non_empty("yolo predictions", prediction_images, yolo_pred_dir)
    for index, source in enumerate(prediction_images, start=1):
        target = output_dir / "figures" / "yolo_predictions" / f"prediction_{index:02d}_{source.name}"
        artifacts.append(_copy_file("yolo_predictions", "YOLO 预测示例", source, target))
    return artifacts


def _copy_file(section: str, name: str, source: Path, target: Path) -> CopiedArtifact:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Required artifact missing: {source}")
    shutil.copy2(source, target)
    return CopiedArtifact(section, name, source, target, "copied")


def _retarget_artifacts(
    artifacts: list[CopiedArtifact],
    staging_dir: Path,
    final_dir: Path,
) -> list[CopiedArtifact]:
    retargeted: list[CopiedArtifact] = []
    for artifact in artifacts:
        try:
            relative_target = artifact.target.relative_to(staging_dir)
            target = final_dir / relative_target
        except ValueError:
            target = artifact.target
        retargeted.append(
            CopiedArtifact(
                artifact.section,
                artifact.name,
                artifact.source,
                target,
                artifact.status,
            )
        )
    return retargeted


def _replace_directory(target_dir: Path, source_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    os.replace(source_dir, target_dir)


def _read_yolo_metrics(results_csv: Path) -> list[dict[str, str]]:
    rows = _read_csv(results_csv)
    normalized = [{key.strip(): value.strip() for key, value in row.items()} for row in rows]
    return normalized


def _read_diagnostic_summaries(paths: dict[str, Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scale, path in paths.items():
        for row in _read_required_csv(f"{scale} diagnostic summary", path):
            item = {key.strip(): value.strip() for key, value in row.items()}
            item["scale"] = scale
            rows.append(item)
    return rows


def _read_required_csv(label: str, path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    _require_non_empty(label, rows, path)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_non_empty(label: str, rows: list[object], *sources: Path) -> None:
    if rows:
        return
    joined = ", ".join(str(source) for source in sources)
    raise FileNotFoundError(f"Required {label} missing or empty: {joined}")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_artifact_manifest(path: Path, artifacts: list[CopiedArtifact]) -> None:
    rows = [
        {
            "section": artifact.section,
            "name": artifact.name,
            "status": artifact.status,
            "source": str(artifact.source),
            "target": str(artifact.target),
        }
        for artifact in artifacts
    ]
    _write_csv(path, rows)


def _data_summary(cpu10_data_dir: Path) -> dict[str, str]:
    sample_rows = _read_required_csv("sample index", cpu10_data_dir / "sample_index.csv")
    annotation_rows = _read_required_csv("annotations", cpu10_data_dir / "annotations.csv")
    split_counts: dict[str, int] = {}
    for row in sample_rows:
        split = row.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    return {
        "dataset": "carrada_ra_cpu10",
        "total_images": str(len(sample_rows)),
        "train_images": str(split_counts.get("train", 0)),
        "val_images": str(split_counts.get("val", 0)),
        "test_images": str(split_counts.get("test", 0)),
        "annotations": str(len(annotation_rows)),
        "sequences": str(len({row.get("sequence_id", "") for row in sample_rows if row.get("sequence_id")})),
        "classes": "pedestrian, cyclist, car",
    }


def _latest_yolo_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[-1] if rows else {}


def _fmt(value: str | None, digits: int = 4) -> str:
    if value is None or value == "":
        return "待生成"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def _metric(row: dict[str, str], key: str) -> str:
    return row.get(key, row.get(f" {key}", ""))


def _write_readme(
    output_dir: Path,
    yolo_run_dir: Path,
    yolo_pred_dir: Path,
    yolo_metrics: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
    data_summary: dict[str, str],
) -> None:
    latest = _latest_yolo_row(yolo_metrics)
    copied_predictions = len(list((output_dir / "figures" / "yolo_predictions").glob("*.jpg")))
    content = f"""# AeroTrack 结项展示包

本目录汇总了本项目从数据转换、检测训练、SORT 跟踪到展示报告的结项材料。内容面向 PPT 制作，图片和表格已经按主题归档。

## 目录

- `figures/data_conversion/`：CARRADA Range-Angle 转换后的 GT 标注抽查图。
- `figures/yolo_training/`：YOLO 训练曲线、PR/F1 曲线、混淆矩阵、验证批次图。
- `figures/yolo_predictions/`：YOLO 推理展示图，当前复制 `{copied_predictions}` 张。
- `figures/diagnostic_pipeline/`：GT 诊断检测 + SORT 跟踪闭环的大图。
- `figures/tracking/`：SORT 连续帧跟踪可视化。
- `tables/`：训练指标、诊断实验指标、数据规模和归档清单。
- `reports/`：中文展示报告和 YOLO 训练说明。
- `commands/`：复现实验命令。

## 当前真实结果口径

- 数据集：`{data_summary.get("dataset", "carrada_ra_cpu10")}`，共 `{data_summary.get("total_images", "0")}` 张 RA PNG，`{data_summary.get("annotations", "0")}` 个标注框。
- YOLO 训练目录：`{yolo_run_dir}`。
- YOLO 预测目录：`{yolo_pred_dir}`。
- YOLO 当前已记录 epoch：`{latest.get("epoch", "待生成")}`。
- YOLO 当前 mAP50：`{_fmt(_metric(latest, "metrics/mAP50(B)"))}`。
- 诊断闭环实验数量：`{len(diagnostic_rows)}`。

## 展示边界

`YOLO` 章节来自真实 Ultralytics 训练输出；`diagnostic_pipeline` 章节来自 `gt_bbox` 诊断检测，用来展示下游 SORT、评估和可视化闭环。两者不能混写成同一个指标结论。
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def _write_project_summary(
    output_dir: Path,
    yolo_metrics: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
    data_summary: dict[str, str],
) -> None:
    latest = _latest_yolo_row(yolo_metrics)
    best_map50 = _best_metric(yolo_metrics, "metrics/mAP50(B)")
    best_map5095 = _best_metric(yolo_metrics, "metrics/mAP50-95(B)")
    content = f"""# AeroTrack 项目展示说明

## 项目目标

AeroTrack 面向 CARRADA 雷达 Range-Angle 数据，构建从数据转换、目标检测、SORT 多目标跟踪、指标评估到可视化归档的实验闭环。最终展示材料强调工程链路、真实训练产物和可复核图表。

## 数据处理

本次展示包使用 `{data_summary.get("dataset")}` 作为本地 YOLO 训练与演示数据。该子集包含 `{data_summary.get("sequences")}` 条 sequence、`{data_summary.get("total_images")}` 张图片和 `{data_summary.get("annotations")}` 个目标框，划分为 train/val/test = `{data_summary.get("train_images")}` / `{data_summary.get("val_images")}` / `{data_summary.get("test_images")}`。

## YOLO 检测结果

YOLOv8n 已在本地 CPU 环境下继续训练。当前归档到的最新 epoch 为 `{latest.get("epoch", "待生成")}`，最新 mAP50 为 `{_fmt(_metric(latest, "metrics/mAP50(B)"))}`，当前最佳 mAP50 为 `{_fmt(best_map50)}`，当前最佳 mAP50-95 为 `{_fmt(best_map5095)}`。

这个结果可以用于说明 RA 图已经能接入 YOLO 训练、验证和推理流程；如果指标仍偏低，展示时应表述为“本地 CPU 条件下的工程验证和可视化结果”，而不是完整高精度 baseline。

## 跟踪与诊断闭环

项目同时保留 `gt_bbox` 诊断检测结果，用于展示下游 SORT 跟踪、评估和可视化闭环。该部分的检测指标来自 GT 框转换，不能代表 YOLO 模型效果。它的价值是证明：一旦检测器输出统一格式的 `detections.csv`，后续跟踪、评估和图像归档可以稳定工作。

## PPT 建议结构

1. 项目目标与技术路线。
2. CARRADA RA 数据转换和标注格式。
3. YOLO 训练曲线和验证样例。
4. YOLO 推理成功展示图。
5. SORT 连续帧跟踪展示。
6. 指标表和 SORT 参数消融。
7. 当前局限和后续改进。

## 可以宣称

- 已完成 CARRADA Range-Angle PNG 和 YOLO 标签转换。
- 已完成本地 YOLOv8n 训练、验证和推理产物归档。
- 已完成 `gt_bbox` 诊断检测到 SORT 跟踪的闭环展示。
- 已生成可用于结项 PPT 的图片、表格和中文说明文档。

## 不能宣称

- `gt_bbox` 诊断检测指标代表 YOLO 模型性能。
- 当前 CPU 训练模型已经达到工程部署精度。
- IDF1、ID switches、fragmentation 已经完成正式评估；当前项目仍标记为 unavailable。
"""
    (output_dir / "project_summary.md").write_text(content, encoding="utf-8")


def _write_yolo_report(output_dir: Path, yolo_run_dir: Path, yolo_pred_dir: Path, yolo_metrics: list[dict[str, str]]) -> None:
    latest = _latest_yolo_row(yolo_metrics)
    rows = "\n".join(
        f"| {row.get('epoch', '')} | {_fmt(row.get('train/box_loss'))} | {_fmt(row.get('train/cls_loss'))} | {_fmt(row.get('metrics/precision(B)'))} | {_fmt(row.get('metrics/recall(B)'))} | {_fmt(row.get('metrics/mAP50(B)'))} | {_fmt(row.get('metrics/mAP50-95(B)'))} |"
        for row in yolo_metrics
    )
    if not rows:
        rows = "| 待生成 | 待生成 | 待生成 | 待生成 | 待生成 | 待生成 | 待生成 |"
    content = f"""# YOLO 训练报告

## 训练来源

- 训练目录：`{yolo_run_dir}`
- 预测目录：`{yolo_pred_dir}`
- 模型：YOLOv8n
- 输入：CARRADA Range-Angle PNG
- 设备：CPU

## 最新指标

| 指标 | 值 |
| --- | ---: |
| 最新 epoch | {latest.get("epoch", "待生成")} |
| precision | {_fmt(_metric(latest, "metrics/precision(B)"))} |
| recall | {_fmt(_metric(latest, "metrics/recall(B)"))} |
| mAP50 | {_fmt(_metric(latest, "metrics/mAP50(B)"))} |
| mAP50-95 | {_fmt(_metric(latest, "metrics/mAP50-95(B)"))} |
| train box loss | {_fmt(latest.get("train/box_loss"))} |
| train cls loss | {_fmt(latest.get("train/cls_loss"))} |
| val box loss | {_fmt(latest.get("val/box_loss"))} |
| val cls loss | {_fmt(latest.get("val/cls_loss"))} |

## Epoch 明细

| epoch | train box loss | train cls loss | precision | recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## 展示建议

优先使用 `figures/yolo_training/results.png` 展示训练趋势，再使用 `figures/yolo_predictions/` 中的预测图说明 YOLO 已经完成推理可视化。若最终 mAP 仍偏低，应明确说明这是 CPU 约束下的小规模训练结果，重点用于证明训练链路和展示产物完整。
"""
    (output_dir / "reports" / "yolo_training_report.md").write_text(content, encoding="utf-8")


def _write_showcase_report(
    output_dir: Path,
    diagnostic_rows: list[dict[str, str]],
    data_summary: dict[str, str],
    sort_sweep_rows: list[dict[str, str]],
) -> None:
    diagnostic_table = "\n".join(
        f"| {row.get('experiment_name', '')} | {row.get('num_sequences', '')} | {row.get('num_frames', '')} | {_fmt(row.get('precision'), 3)} | {_fmt(row.get('recall'), 3)} | {_fmt(row.get('map50'), 3)} | {_fmt(row.get('mota'), 3)} | {row.get('idf1_status', '')} |"
        for row in diagnostic_rows
    )
    sort_table = "\n".join(
        f"| {row.get('max_age', '')} | {row.get('min_hits', '')} | {row.get('iou_threshold', '')} | {row.get('num_tracks', '')} | {_fmt(row.get('mota'), 3)} | {row.get('tp', '')} | {row.get('fp', '')} | {row.get('fn', '')} |"
        for row in sort_sweep_rows[:8]
    )
    content = f"""# 最终展示报告

## 展示主线

本展示包建议围绕“数据转换 -> YOLO 检测训练 -> SORT 跟踪闭环 -> 指标与可视化归档”展开。`figures/diagnostic_pipeline/slide_01_large_triptych.png` 适合作为主图，用一页解释 RA 图、目标框和轨迹输出之间的关系。

## 数据转换成果

`{data_summary.get("dataset")}` 包含 `{data_summary.get("total_images")}` 张 RA PNG，标注框 `{data_summary.get("annotations")}` 个。`figures/data_conversion/` 中的 GT 可视化图可用于说明标注框已经正确叠加在 RA 图上。

## 诊断闭环指标

| 实验 | sequence 数 | test 帧数 | precision | recall | mAP50 | MOTA | IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{diagnostic_table}

这些指标来自 `gt_bbox` 诊断检测，检测框由 GT 标注转换得到，因此检测指标接近理想值是预期现象。该章节用于证明后处理链路可运行，不能作为 YOLO 精度结论。

## SORT 参数消融

| max_age | min_hits | IOU 阈值 | 轨迹数 | MOTA | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{sort_table}

SORT 消融可用于说明跟踪结果受关联阈值和轨迹确认策略影响。当前 IDF1、ID switches 和 fragmentation 仍未接入正式评估，应在答辩中保持 unavailable 口径。

## 推荐图片

- `figures/yolo_training/results.png`：YOLO 训练曲线总览。
- `figures/yolo_training/BoxPR_curve.png`：PR 曲线。
- `figures/yolo_predictions/`：YOLO 推理展示图。
- `figures/diagnostic_pipeline/slide_02_tracking_sequence.png`：连续帧 SORT 跟踪。
- `figures/diagnostic_pipeline/slide_08_scale_comparison.png`：smoke 与 cpu10 规模对比。
"""
    (output_dir / "reports" / "final_showcase_report.md").write_text(content, encoding="utf-8")


def _write_commands(output_dir: Path, yolo_run_dir: Path, yolo_pred_dir: Path) -> None:
    source_list = yolo_pred_dir.parent / "carrada_ra_cpu10_showcase_sources.txt"
    content = f"""# 可复现实验命令

## 生成 Ultralytics 数据配置

```bash
uv run python scripts/prepare_ultralytics_data.py \\
  --prepared-root /home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10 \\
  --container-path /home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10
```

## 本地 CPU YOLO 继续训练

```bash
uv run --extra yolo yolo detect train \\
  model=/home/lh/projects/AeroTrack/runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/weights/best.pt \\
  data=/home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10/ultralytics/yolo_data.yaml \\
  imgsz=256 epochs=30 batch=4 device=cpu \\
  project=/home/lh/projects/AeroTrack/runs/yolo_final_demo \\
  name=carrada_ra_cpu10_yolov8n_e30_cpu exist_ok=True
```

## 继续训练

```bash
uv run --extra yolo yolo detect train \\
  model=/home/lh/projects/AeroTrack/runs/yolo_final_demo/carrada_ra_cpu10_yolov8n_e30_cpu/weights/best.pt \\
  data=/home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10/ultralytics/yolo_data.yaml \\
  imgsz=256 epochs=60 batch=4 device=cpu \\
  project=/home/lh/projects/AeroTrack/runs/yolo_final_demo \\
  name=carrada_ra_cpu10_yolov8n_e60_cpu exist_ok=True
```

## YOLO 展示预测

```bash
uv run --extra yolo yolo detect predict \\
  model={yolo_run_dir}/weights/best.pt \\
  source={source_list} \\
  imgsz=256 conf=0.001 save=True save_txt=True save_conf=True device=cpu \\
  project={yolo_pred_dir.parent} \\
  name={yolo_pred_dir.name} exist_ok=True
```

## 生成 final 展示包

```bash
uv run python scripts/build_final_showcase.py
```
"""
    (output_dir / "commands" / "reproducible_commands.md").write_text(content, encoding="utf-8")


def _best_metric(rows: list[dict[str, str]], key: str) -> str:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, "")))
        except ValueError:
            continue
    return str(max(values)) if values else ""


if __name__ == "__main__":
    raise SystemExit(main())
