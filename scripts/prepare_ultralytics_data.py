from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


CLASS_NAMES = {
    0: "pedestrian",
    1: "cyclist",
    2: "car",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Ultralytics split lists for an AeroTrack processed dataset.")
    parser.add_argument(
        "--prepared-root",
        default="data/processed/carrada_ra_cpu10",
        help="AeroTrack prepared dataset root containing sample_index.csv, images/, labels/, and splits/.",
    )
    parser.add_argument(
        "--container-path",
        default="/workspace/data/processed/carrada_ra_cpu10",
        help="Absolute dataset path as seen inside the Docker container.",
    )
    args = parser.parse_args(argv)

    prepared_root = Path(args.prepared_root).resolve()
    container_path = str(args.container_path).rstrip("/")
    outputs = prepare_ultralytics_data(prepared_root, container_path)
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


def prepare_ultralytics_data(prepared_root: Path, container_path: str) -> dict[str, Path]:
    sample_index = _read_sample_index(prepared_root / "sample_index.csv")
    output_dir = prepared_root / "ultralytics"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_outputs: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_path = prepared_root / "splits" / f"{split}.txt"
        sample_ids = _read_split(split_path)
        image_paths = [_container_image_path(container_path, sample_index[sample_id]) for sample_id in sample_ids]
        output_path = output_dir / f"{split}.txt"
        output_path.write_text("\n".join(image_paths) + ("\n" if image_paths else ""), encoding="utf-8")
        split_outputs[split] = output_path

    data_yaml = output_dir / "yolo_data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": container_path,
                "train": "ultralytics/train.txt",
                "val": "ultralytics/val.txt",
                "test": "ultralytics/test.txt",
                "names": CLASS_NAMES,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {"data": data_yaml, **split_outputs}


def _read_sample_index(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sample_index.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {row["sample_id"]: row["image_path"] for row in rows}


def _read_split(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _container_image_path(container_path: str, relative_image_path: str) -> str:
    return f"{container_path}/{relative_image_path.lstrip('/')}"


if __name__ == "__main__":
    raise SystemExit(main())
