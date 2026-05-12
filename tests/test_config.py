from pathlib import Path

from aerotrack.config import load_experiment_config


def test_load_experiment_config_resolves_references() -> None:
    config = load_experiment_config("configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml")

    assert config["experiment_name"] == "carrada_ra_gtbbox_sort_smoke"
    assert config["dataset"]["name"] == "carrada"
    assert config["dataset"]["representation"] == "range_angle"
    assert config["detector"]["source"] == "gt_bbox"
    assert config["tracker"]["name"] == "sort"
    assert Path(config["dataset"]["config"]).exists()
