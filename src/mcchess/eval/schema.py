"""Versioned result-envelope helpers for evaluation artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 2
REQUIRED_TOP_LEVEL = ("schema_version", "run", "protocol", "participants", "summary", "config")
REQUIRED_RUN_FIELDS = (
    "id",
    "type",
    "status",
    "seed",
    "started_at",
    "completed_at",
    "elapsed_seconds",
    "git_commit",
    "config_path",
)


def result_envelope(
    *,
    run_id: str,
    run_type: str,
    status: str,
    seed: int,
    started_at: str,
    completed_at: str,
    elapsed_seconds: float,
    git_commit: str | None,
    config_path: str | None,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    participants: Mapping[str, Any],
    summary: Mapping[str, Any],
    games: Sequence[Mapping[str, Any]] | None = None,
    samples: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a schema-v2 evaluation result."""

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "id": run_id,
            "type": run_type,
            "status": status,
            "seed": seed,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": elapsed_seconds,
            "git_commit": git_commit,
            "config_path": config_path,
        },
        "protocol": dict(protocol),
        "participants": dict(participants),
        "summary": dict(summary),
        "config": dict(config),
    }
    if games is not None:
        result["games"] = [dict(game) for game in games]
    if samples is not None:
        result["samples"] = dict(samples)
    if metrics is not None:
        result["metrics"] = dict(metrics)
    validate_result_envelope(result)
    return result


def validate_result_envelope(result: Mapping[str, Any]) -> None:
    """Validate the common schema-v2 envelope shared by eval artifacts."""

    _require_keys("result", result, REQUIRED_TOP_LEVEL)
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported result schema_version: {result['schema_version']}")

    run = _require_mapping("run", result["run"])
    _require_keys("run", run, REQUIRED_RUN_FIELDS)
    if run["status"] not in {"completed", "failed", "aborted", "inconclusive"}:
        raise ValueError(f"run.status is not a valid terminal status: {run['status']}")

    _require_mapping("protocol", result["protocol"])
    _require_mapping("participants", result["participants"])
    _require_mapping("summary", result["summary"])
    _require_mapping("config", result["config"])

    if "games" in result and not isinstance(result["games"], list):
        raise ValueError("games must be a list when present")
    if "samples" in result:
        _require_mapping("samples", result["samples"])
    if "metrics" in result:
        _require_mapping("metrics", result["metrics"])


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_keys(name: str, mapping: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{name} is missing required keys: {missing}")
