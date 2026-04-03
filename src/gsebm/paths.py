"""Path helpers for locating repository-owned data directories."""

from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    """Return the repository root directory."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate the repository root from pyproject.toml.")


def get_data_dir(*, create: bool = True) -> Path:
    """Return the repository data directory."""

    data_dir = get_repo_root() / "data"
    if create:
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
