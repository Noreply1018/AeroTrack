from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_yolo_demo.py"
    spec = importlib.util.spec_from_file_location("summarize_yolo_demo", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_summary_reads_latest_metrics_and_counts_prediction_files(tmp_path: Path) -> None:
    module = _load_script()
    run_dir = tmp_path / "run"
    pred_dir = tmp_path / "pred"
    (run_dir / "weights").mkdir(parents=True)
    (pred_dir / "labels").mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "\n".join(
            [
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss",
                "1,0.1,0.2,0.3,0.4,1,2,3,4,5,6",
                "2,0.7,0.8,0.9,1.0,7,8,9,10,11,12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "weights" / "best.pt").write_text("weights", encoding="utf-8")
    (pred_dir / "a.jpg").write_text("image", encoding="utf-8")
    (pred_dir / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n", encoding="utf-8")

    output = tmp_path / "summary.csv"
    module.write_summary(run_dir, pred_dir, output)

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["name"]: row for row in rows}
    assert by_name["metrics/precision(B)"]["value"] == "0.7"
    assert by_name["weights/best.pt"]["value"] == "exists"
    assert by_name["showcase_prediction_images"]["value"] == "1"
    assert by_name["showcase_prediction_label_files"]["value"] == "1"
