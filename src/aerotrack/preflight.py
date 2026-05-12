from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    message: str


@dataclass(frozen=True)
class PreflightResult:
    checks: tuple[PreflightCheck, ...]

    @property
    def has_error(self) -> bool:
        return any(check.status == "error" for check in self.checks)

    @property
    def has_gate(self) -> bool:
        return any(check.status == "gate" for check in self.checks)

    @property
    def exit_code(self) -> int:
        if self.has_error:
            return 2
        if self.has_gate:
            return 3
        return 0


def _repo_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config.get("repo_root", ".")).resolve() / path


def run_preflight(config: dict[str, Any]) -> PreflightResult:
    checks: list[PreflightCheck] = []

    checks.append(_check_python())
    checks.append(_check_experiment_name(config))
    checks.append(_check_output_root(config))
    checks.extend(_check_dataset(config))
    checks.extend(_check_detector(config))

    return PreflightResult(tuple(checks))


def run_training_preflight(config: dict[str, Any]) -> PreflightResult:
    checks: list[PreflightCheck] = []

    checks.append(_check_python())
    checks.append(_check_experiment_name(config))
    checks.append(_check_output_root(config))
    checks.extend(_check_dataset(config))
    checks.extend(_check_training_detector(config))

    return PreflightResult(tuple(checks))


def _check_python() -> PreflightCheck:
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        return PreflightCheck("python", "ok", f"Python {version.major}.{version.minor} matches project constraint")
    return PreflightCheck(
        "python",
        "error",
        f"Python {version.major}.{version.minor} is active; use Python 3.11 via uv for this project",
    )


def _check_experiment_name(config: dict[str, Any]) -> PreflightCheck:
    name = config.get("experiment_name")
    if isinstance(name, str) and name.strip():
        return PreflightCheck("experiment_name", "ok", f"Experiment name: {name}")
    return PreflightCheck("experiment_name", "error", "experiment_name is required")


def _check_output_root(config: dict[str, Any]) -> PreflightCheck:
    output_root = config.get("output", {}).get("root", "runs")
    path = _repo_path(config, output_root)
    parent = path if path.exists() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    free_gb = usage.free / (1024**3)
    if free_gb < 5:
        return PreflightCheck("disk", "warning", f"Only {free_gb:.1f} GiB free near {path}")
    return PreflightCheck("disk", "ok", f"{free_gb:.1f} GiB free near {path}")


def _check_dataset(config: dict[str, Any]) -> list[PreflightCheck]:
    dataset = config.get("dataset", {})
    checks: list[PreflightCheck] = []
    name = dataset.get("name")
    representation = dataset.get("representation")
    if name != "carrada":
        checks.append(PreflightCheck("dataset.name", "error", "Stage1 expects dataset.name=carrada"))
    else:
        checks.append(PreflightCheck("dataset.name", "ok", "Dataset is CARRADA"))

    if representation != "range_angle":
        checks.append(
            PreflightCheck(
                "dataset.representation",
                "error",
                "Stage1 expects dataset.representation=range_angle",
            )
        )
    else:
        checks.append(PreflightCheck("dataset.representation", "ok", "Representation is range_angle"))

    root = dataset.get("root", "data/carrada")
    root_path = _repo_path(config, root)
    if root_path.exists():
        if any(root_path.iterdir()):
            checks.append(PreflightCheck("dataset.root", "ok", f"CARRADA root exists: {root_path}"))
        else:
            checks.append(PreflightCheck("dataset.root", "gate", f"CARRADA root is empty: {root_path}"))
    else:
        checks.append(
            PreflightCheck(
                "dataset.root",
                "gate",
                f"CARRADA data not found at {root_path}; provide a local path or confirm download before continuing",
            )
        )

    return checks


def _check_detector(config: dict[str, Any]) -> list[PreflightCheck]:
    detector = config.get("detector", {})
    source = detector.get("source")
    if source == "gt_bbox":
        return [PreflightCheck("detector.source", "ok", "Detection source is gt_bbox for smoke diagnostics")]

    if source == "yolo_pretrained":
        weight = detector.get("weights")
        if not weight:
            return [PreflightCheck("detector.weights", "gate", "YOLO pretrained detector requires weights")]
        weight_path = _repo_path(config, weight)
        if not weight_path.exists():
            return [
                PreflightCheck(
                    "detector.weights",
                    "gate",
                    f"YOLO weights not found at {weight_path}; confirm download or provide weights",
                )
            ]
        return [PreflightCheck("detector.weights", "ok", f"YOLO weights found: {weight_path}")]

    return [PreflightCheck("detector.source", "error", "detector.source must be gt_bbox or yolo_pretrained")]


def _check_training_detector(config: dict[str, Any]) -> list[PreflightCheck]:
    detector = config.get("detector", {})
    checks: list[PreflightCheck] = []

    if detector.get("task") != "train":
        checks.append(PreflightCheck("detector.task", "error", "Detector training config must set task=train"))
    else:
        checks.append(PreflightCheck("detector.task", "ok", "Detector task is train"))

    if detector.get("implementation") != "ultralytics":
        checks.append(
            PreflightCheck(
                "detector.implementation",
                "error",
                "Stage1 detector training template expects implementation=ultralytics",
            )
        )
    else:
        checks.append(PreflightCheck("detector.implementation", "ok", "Detector implementation is Ultralytics"))

    weights = detector.get("initial_weights")
    allow_download = bool(detector.get("allow_weight_download", False))
    if weights:
        weight_path = _repo_path(config, weights)
        if weight_path.exists():
            checks.append(PreflightCheck("detector.initial_weights", "ok", f"Initial weights found: {weight_path}"))
        elif allow_download:
            checks.append(
                PreflightCheck(
                    "detector.initial_weights",
                    "gate",
                    f"Initial weights are not local at {weight_path}; confirm download before training",
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "detector.initial_weights",
                    "gate",
                    f"Initial weights not found at {weight_path}; provide weights or enable explicit download",
                )
            )
    else:
        checks.append(PreflightCheck("detector.initial_weights", "warning", "No initial weights configured"))

    if detector.get("enabled") is False:
        checks.append(
            PreflightCheck(
                "detector.training",
                "gate",
                "YOLO training is disabled for Stage1 smoke execution; enable only after data and weights are confirmed",
            )
        )
    else:
        checks.append(PreflightCheck("detector.training", "ok", "YOLO training is enabled by config"))

    return checks


def format_preflight(result: PreflightResult) -> str:
    lines: list[str] = []
    for check in result.checks:
        lines.append(f"[{check.status.upper()}] {check.name}: {check.message}")
    return "\n".join(lines)
