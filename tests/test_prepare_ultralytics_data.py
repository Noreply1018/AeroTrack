from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_ultralytics_data.py"
    spec = importlib.util.spec_from_file_location("prepare_ultralytics_data", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_ultralytics_data_writes_container_image_lists(tmp_path: Path) -> None:
    module = _load_script()
    prepared_root = tmp_path / "processed"
    (prepared_root / "splits").mkdir(parents=True)
    (prepared_root / "sample_index.csv").write_text(
        "\n".join(
            [
                "sample_id,sequence_id,frame_id,split,representation,image_path,label_path",
                "seq_000001,seq,000001,train,range_angle,images/seq/000001.png,labels/seq/000001.txt",
                "seq_000002,seq,000002,val,range_angle,images/seq/000002.png,labels/seq/000002.txt",
                "seq_000003,seq,000003,test,range_angle,images/seq/000003.png,labels/seq/000003.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (prepared_root / "splits" / "train.txt").write_text("seq_000001\n", encoding="utf-8")
    (prepared_root / "splits" / "val.txt").write_text("seq_000002\n", encoding="utf-8")
    (prepared_root / "splits" / "test.txt").write_text("seq_000003\n", encoding="utf-8")

    outputs = module.prepare_ultralytics_data(prepared_root, "/workspace/data/processed/demo")

    assert outputs["train"].read_text(encoding="utf-8").splitlines() == [
        "/workspace/data/processed/demo/images/seq/000001.png"
    ]
    assert outputs["val"].read_text(encoding="utf-8").splitlines() == [
        "/workspace/data/processed/demo/images/seq/000002.png"
    ]
    data = yaml.safe_load(outputs["data"].read_text(encoding="utf-8"))
    assert data["path"] == "/workspace/data/processed/demo"
    assert data["train"] == "ultralytics/train.txt"
    assert data["names"][2] == "car"
