from pathlib import Path

import pytest

from aerotrack.config import load_experiment_config
from aerotrack.data_prep import prepare_carrada_ra_smoke


def test_prepare_data_generates_real_carrada_outputs(tmp_path: Path) -> None:
    if not Path("data/carrada/Carrada/annotations_instance_oriented.json").exists():
        pytest.skip("local CARRADA data is not available")
    config = load_experiment_config("configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml")
    config["dataset"]["prepared_root"] = str(tmp_path / "processed")
    config["dataset"]["splits"]["max_sequences"] = 1

    prepared = prepare_carrada_ra_smoke(config)

    assert prepared.sample_index_path.exists()
    assert prepared.annotations_path.exists()
    assert (prepared.root / "classes.yaml").exists()
    assert (prepared.root / "splits" / "train.txt").exists()
    assert any((prepared.root / "labels").glob("*/*.txt"))
    assert any((prepared.root / "visual_checks" / "gt").glob("*.png"))


def test_prepare_data_accepts_root_that_points_at_carrada_directory(tmp_path: Path) -> None:
    carrada_dir = Path("data/carrada/Carrada")
    if not (carrada_dir / "annotations_instance_oriented.json").exists():
        pytest.skip("local CARRADA data is not available")
    config = load_experiment_config("configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml")
    config["dataset"]["root"] = str(carrada_dir)
    config["dataset"]["prepared_root"] = str(tmp_path / "processed")
    config["dataset"]["splits"]["max_sequences"] = 1

    prepared = prepare_carrada_ra_smoke(config)

    assert prepared.sample_index_path.exists()
