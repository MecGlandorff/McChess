"""Shared helpers for evaluation CLIs and artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and require a top-level mapping."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return dict(raw)


def write_text_atomic(path: Path, text: str) -> None:
    """Write a text file via a same-directory temporary path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an indented JSON file atomically."""

    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    """Write a CSV file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def git_commit() -> str | None:
    """Return the current git commit, if available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def resolve_executable(
    *,
    explicit: str | None,
    config_value: str | None,
    env_value: str | None,
    path_name: str,
    display_name: str,
) -> str:
    """Resolve an executable from override, config, environment, or PATH."""

    candidate = explicit or config_value or env_value
    if candidate is None:
        candidate = shutil.which(path_name)
    if not candidate:
        raise FileNotFoundError(
            f"{display_name} binary not found. Set the environment variable, "
            f"pass an explicit path, or put {path_name} on PATH."
        )

    path = Path(candidate)
    if path.exists():
        if not path.is_file():
            raise FileNotFoundError(f"{display_name} path is not a file: {candidate}")
        return str(path)

    resolved = shutil.which(candidate)
    if resolved:
        return resolved

    raise FileNotFoundError(f"{display_name} binary not found: {candidate}")
