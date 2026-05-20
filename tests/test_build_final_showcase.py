from __future__ import annotations

import csv
import importlib.util
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


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)


def _make_fixture(tmp_path: Path):
    yolo_run = tmp_path / "runs" / "yolo"
    yolo_pred = tmp_path / "runs" / "pred"
    smoke = tmp_path / "runs" / "smoke"
    cpu10 = tmp_path / "runs" / "cpu10"
    server30 = tmp_path / "runs" / "server30"
    data = tmp_path / "data" / "cpu10"
    output = tmp_path / "final"

    _write_csv(
        yolo_run / "results.csv",
        [
            {
                "epoch": "1",
                "metrics/mAP50(B)": "0.1",
                "metrics/mAP50-95(B)": "0.01",
                "metrics/precision(B)": "0.2",
                "metrics/recall(B)": "0.3",
                "train/box_loss": "1.0",
                "train/cls_loss": "2.0",
                "val/box_loss": "1.5",
                "val/cls_loss": "2.5",
            }
        ],
    )
    (yolo_run / "args.yaml").write_text("imgsz: 256\n", encoding="utf-8")
    for name in [
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
    ]:
        _png(yolo_run / name)
    for name in ["000001.jpg", "000002.jpg", "000003.jpg"]:
        _png(yolo_pred / name)

    summary_row = {
        "experiment_name": "demo",
        "detection_source": "gt_bbox",
        "split": "test",
        "num_sequences": "1",
        "num_frames": "2",
        "precision": "1.0",
        "recall": "1.0",
        "f1": "1.0",
        "map50": "1.0",
        "mota": "0.5",
        "idf1_status": "unavailable",
        "notes": "demo",
    }
    for run in [smoke, cpu10, server30]:
        _write_csv(run / "metrics" / "summary.csv", [summary_row])
    _write_csv(
        cpu10 / "analysis" / "sort_sweep" / "sort_sweep_summary.csv",
        [
            {
                "max_age": "1",
                "min_hits": "1",
                "iou_threshold": "0.3",
                "num_tracks": "2",
                "mota": "0.5",
                "tp": "2",
                "fp": "0",
                "fn": "0",
            }
        ],
    )
    (cpu10 / "analysis" / "identity_stability_audit.json").parent.mkdir(parents=True, exist_ok=True)
    (cpu10 / "analysis" / "identity_stability_audit.json").write_text('{"status":"pass"}\n', encoding="utf-8")

    for name in [
        "slide_01_large_triptych.png",
        "slide_02_tracking_sequence.png",
        "slide_03_metrics.png",
        "slide_04_failure_report.png",
        "slide_05_single_target_detail.png",
        "slide_06_track_strip.png",
        "slide_07_sort_sweep.png",
        "slide_08_scale_comparison.png",
    ]:
        _png(smoke / "showcase" / "slides" / name)
    _png(cpu10 / "visualizations" / "tracks" / "demo_000001.png")

    _write_csv(
        data / "sample_index.csv",
        [
            {"sequence_id": "seq", "frame_id": "000001", "split": "train"},
            {"sequence_id": "seq", "frame_id": "000002", "split": "test"},
        ],
    )
    _write_csv(data / "annotations.csv", [{"sequence_id": "seq", "frame_id": "000001"}])
    (data / "conversion_records.csv").write_text("a,b\n", encoding="utf-8")
    (data / "classes.yaml").write_text("names: [car]\n", encoding="utf-8")
    _png(data / "visual_checks" / "gt" / "demo.png")

    return {
        "yolo_run": yolo_run,
        "yolo_pred": yolo_pred,
        "smoke": smoke,
        "cpu10": cpu10,
        "server30": server30,
        "data": data,
        "output": output,
    }


def _build(build_final_showcase, fixture: dict[str, Path]):
    return build_final_showcase.build_final_showcase(
        output_dir=fixture["output"],
        yolo_run_dir=fixture["yolo_run"],
        yolo_pred_dir=fixture["yolo_pred"],
        smoke_run_dir=fixture["smoke"],
        cpu10_run_dir=fixture["cpu10"],
        server30_run_dir=fixture["server30"],
        cpu10_data_dir=fixture["data"],
    )


def test_build_final_showcase_writes_reports_and_manifest(tmp_path):
    build_final_showcase = _load_script("build_final_showcase")
    fixture = _make_fixture(tmp_path)

    artifacts = _build(build_final_showcase, fixture)

    output = fixture["output"]
    assert artifacts
    assert (output / "README.md").exists()
    assert (output / "project_summary.md").read_text(encoding="utf-8").find("AeroTrack") >= 0
    assert (output / "reports" / "yolo_training_report.md").exists()
    assert (output / "figures" / "yolo_predictions" / "prediction_01_000001.jpg").exists()
    assert (output / "figures" / "yolo_predictions" / "prediction_03_000003.jpg").exists()
    manifest = (output / "tables" / "artifact_manifest.csv").read_text(encoding="utf-8")
    assert "missing" not in manifest
    assert "copied" in manifest
    assert "aerotrack_final_" not in manifest
    assert "final/figures/yolo_predictions/prediction_01_000001.jpg" in manifest


def test_build_final_showcase_fails_when_required_artifact_missing(tmp_path):
    build_final_showcase = _load_script("build_final_showcase")
    fixture = _make_fixture(tmp_path)
    (fixture["yolo_run"] / "BoxPR_curve.png").unlink()

    try:
        _build(build_final_showcase, fixture)
    except FileNotFoundError as exc:
        assert "BoxPR_curve.png" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for missing required artifact")


def test_build_final_showcase_fails_when_predictions_missing(tmp_path):
    build_final_showcase = _load_script("build_final_showcase")
    fixture = _make_fixture(tmp_path)
    for image in fixture["yolo_pred"].glob("*.jpg"):
        image.unlink()

    try:
        _build(build_final_showcase, fixture)
    except FileNotFoundError as exc:
        assert "yolo predictions" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for empty prediction directory")


def test_build_final_showcase_fails_when_diagnostic_summary_missing(tmp_path):
    build_final_showcase = _load_script("build_final_showcase")
    fixture = _make_fixture(tmp_path)
    (fixture["server30"] / "metrics" / "summary.csv").unlink()

    try:
        _build(build_final_showcase, fixture)
    except FileNotFoundError as exc:
        assert "server30 diagnostic summary" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for missing diagnostic summary")


def test_build_final_showcase_fails_when_sample_index_missing(tmp_path):
    build_final_showcase = _load_script("build_final_showcase")
    fixture = _make_fixture(tmp_path)
    (fixture["data"] / "sample_index.csv").unlink()

    try:
        _build(build_final_showcase, fixture)
    except FileNotFoundError as exc:
        assert "sample index" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for missing sample index")


def test_build_final_showcase_failure_cleans_final_dir(tmp_path):
    build_final_showcase = _load_script("build_final_showcase")
    fixture = _make_fixture(tmp_path)
    (fixture["yolo_run"] / "results.png").unlink()

    try:
        _build(build_final_showcase, fixture)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")

    assert not fixture["output"].exists()


def test_reproducible_commands_use_absolute_model_path(tmp_path):
    build_final_showcase = _load_script("build_final_showcase")
    fixture = _make_fixture(tmp_path)

    _build(build_final_showcase, fixture)

    commands = (fixture["output"] / "commands" / "reproducible_commands.md").read_text(encoding="utf-8")
    assert "model=runs/" not in commands
    assert f"model={fixture['yolo_run']}/weights/best.pt" in commands
    assert f"source={fixture['yolo_pred'].parent / 'carrada_ra_cpu10_showcase_sources.txt'}" in commands
