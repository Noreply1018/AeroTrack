from pathlib import Path

from aerotrack.cli import main
from aerotrack.config import load_detector_train_config, load_experiment_config
from aerotrack.preflight import run_preflight


def test_preflight_stops_at_dataset_gate_when_carrada_missing() -> None:
    config = load_experiment_config("configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml")
    config["dataset"]["root"] = "__missing_carrada_for_test__"

    result = run_preflight(config)

    assert result.has_gate
    assert result.exit_code == 3


def test_preflight_stops_when_carrada_metadata_missing(tmp_path: Path) -> None:
    root = tmp_path / "carrada"
    root.mkdir()
    config = load_experiment_config("configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml")
    config["dataset"]["root"] = str(root)

    result = run_preflight(config)

    assert result.has_gate
    assert any(check.name == "dataset.files" for check in result.checks)


def test_preflight_stops_at_yolo_weight_gate_when_weights_missing() -> None:
    config = load_experiment_config("configs/experiment/carrada_ra_yolopretrained_sort_smoke.yaml")
    config["dataset"]["root"] = "__missing_carrada_for_test__"
    config["detector"]["weights"] = "weights/__missing_yolo_for_test__.pt"

    result = run_preflight(config)

    assert result.has_gate
    assert result.exit_code == 3
    assert any(check.name == "detector.weights" and check.status == "gate" for check in result.checks)


def test_cli_gate_does_not_create_run_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "gate_experiment.yaml"
    output_root = tmp_path / "runs"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: gate_experiment",
                "dataset:",
                f"  config: {Path('configs/dataset/carrada_ra_smoke.yaml').resolve()}",
                "  root: __missing_carrada_for_test__",
                "detector:",
                f"  config: {Path('configs/detector/gt_bbox.yaml').resolve()}",
                "tracker:",
                f"  config: {Path('configs/tracker/sort.yaml').resolve()}",
                "output:",
                f"  root: {output_root}",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["run-experiment", "--config", str(config_path)])

    assert exit_code == 3
    assert not output_root.exists()


def test_training_preflight_template_is_gated() -> None:
    config = load_detector_train_config("configs/detector/yolo_train.yaml")

    from aerotrack.preflight import run_training_preflight

    result = run_training_preflight(config)

    assert result.has_gate
    assert result.exit_code == 3
    assert any(check.name == "detector.training" and check.status == "gate" for check in result.checks)
