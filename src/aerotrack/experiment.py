from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aerotrack.config import write_yaml


EXPERIMENT_SUBDIRS = [
    "detections",
    "tracks",
    "metrics",
    "visualizations",
    "logs",
]


def experiment_dir(config: dict[str, Any]) -> Path:
    repo_root = Path(config.get("repo_root", "."))
    output_root = Path(config.get("output", {}).get("root", "runs"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    name = config.get("experiment_name")
    if not name:
        raise ValueError("Experiment config must define experiment_name")
    return output_root / str(name)


def prepare_experiment_dir(config: dict[str, Any]) -> Path:
    run_dir = experiment_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in EXPERIMENT_SUBDIRS:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    archived = dict(config)
    archived.setdefault("run_metadata", {})
    archived["run_metadata"]["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_yaml(run_dir / "config.yaml", archived)
    return run_dir
