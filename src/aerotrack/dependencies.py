from __future__ import annotations

from importlib.util import find_spec


SMOKE_DEPENDENCIES = {
    "numpy": "numpy",
    "PIL": "pillow",
    "scipy": "scipy",
    "motmetrics": "motmetrics",
}


def missing_smoke_dependencies() -> list[str]:
    return [package for module, package in SMOKE_DEPENDENCIES.items() if find_spec(module) is None]


def smoke_dependency_hint(missing: list[str] | None = None) -> str:
    packages = missing if missing is not None else missing_smoke_dependencies()
    if not packages:
        return ""
    joined = ", ".join(packages)
    return (
        f"Missing Stage1 smoke dependencies: {joined}. "
        "Run `uv sync` or `uv run --extra vision --extra tracking ...` and retry."
    )
