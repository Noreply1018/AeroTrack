from pathlib import Path

from aerotrack.cli import main


def _write_config(path: Path, prepared_root: Path, output_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "experiment_name: stage_cli_test",
                "dataset:",
                f"  config: {Path('configs/dataset/carrada_ra_smoke.yaml').resolve()}",
                f"  prepared_root: {prepared_root}",
                "detector:",
                f"  config: {Path('configs/detector/gt_bbox.yaml').resolve()}",
                "tracker:",
                f"  config: {Path('configs/tracker/sort.yaml').resolve()}",
                "evaluation:",
                "  split: test",
                "  iou_threshold: 0.5",
                "output:",
                f"  root: {output_root}",
            ]
        ),
        encoding="utf-8",
    )


def test_stage_scripts_do_not_run_full_pipeline_when_inputs_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    prepared_root = tmp_path / "processed"
    output_root = tmp_path / "runs"
    _write_config(config_path, prepared_root, output_root)

    exit_code = main(["run-tracking", "--config", str(config_path)])

    assert exit_code == 2
    assert not (output_root / "stage_cli_test" / "detections" / "detections.csv").exists()
