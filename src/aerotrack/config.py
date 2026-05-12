from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is invalid."""


def read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_path(path: str | Path, *, base_dir: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(base_dir) / candidate).resolve()


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    experiment = read_yaml(config_path)
    repo_root = find_repo_root(config_path.parent)

    resolved: dict[str, Any] = {
        "config_path": str(config_path),
        "repo_root": str(repo_root),
    }
    resolved = deep_merge(resolved, experiment)

    for section in ("dataset", "detector", "tracker"):
        section_data = resolved.get(section, {})
        if not isinstance(section_data, dict):
            raise ConfigError(f"Experiment section must be a mapping: {section}")
        reference = section_data.get("config")
        if reference is None:
            continue
        reference_path = resolve_path(reference, base_dir=repo_root)
        referenced_data = read_yaml(reference_path)
        inline_override = {k: v for k, v in section_data.items() if k != "config"}
        resolved[section] = deep_merge(referenced_data, inline_override)
        resolved[section]["config"] = str(reference_path)

    return resolved


def load_detector_train_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    detector = read_yaml(config_path)
    repo_root = find_repo_root(config_path.parent)

    resolved: dict[str, Any] = {
        "config_path": str(config_path),
        "repo_root": str(repo_root),
        "experiment_name": detector.get("name", "yolo_train_template"),
        "output": detector.get("output", {"root": "runs"}),
        "detector": detector,
    }

    dataset = detector.get("dataset", {})
    if isinstance(dataset, dict):
        reference = dataset.get("config")
        if reference is not None:
            reference_path = resolve_path(reference, base_dir=repo_root)
            dataset_config = read_yaml(reference_path)
            inline_override = {k: v for k, v in dataset.items() if k != "config"}
            resolved["dataset"] = deep_merge(dataset_config, inline_override)
            resolved["dataset"]["config"] = str(reference_path)
        else:
            resolved["dataset"] = dataset
    else:
        raise ConfigError("Detector train config section dataset must be a mapping")

    return resolved


def find_repo_root(start: str | Path) -> Path:
    current = Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()
