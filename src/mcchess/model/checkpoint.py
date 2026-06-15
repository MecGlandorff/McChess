"""Checkpoint loading helpers for policy/value models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from mcchess.model.network import PolicyValueResNet, ResNetConfig


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata copied from a supervised training checkpoint."""

    path: Path
    epoch: int | None
    saved_at: str | None
    completed_at: str | None
    metrics: dict[str, Any]
    train_config: dict[str, Any]


@dataclass(frozen=True)
class LoadedPolicyValueCheckpoint:
    """Loaded model plus the metadata needed to identify it in evaluations."""

    model: PolicyValueResNet
    model_config: ResNetConfig
    metadata: CheckpointMetadata
    device: torch.device


@dataclass(frozen=True)
class _CheckpointSummary:
    path: Path
    metric_value: float | None
    completed_at: str
    epoch: int
    modified_at: float


def resolve_torch_device(name: str | torch.device = "auto") -> torch.device:
    """Resolve `auto`, `cpu`, `cuda`, `mps`, or an explicit torch device."""

    if isinstance(name, torch.device):
        return name
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def find_best_policy_value_checkpoint(
    runs_dir: str | Path,
    *,
    metric_name: str = "val_total_loss",
    checkpoint_names: tuple[str, ...] = ("checkpoint_latest.pt", "checkpoint.pt"),
) -> Path:
    """Return the playable checkpoint with the lowest recorded validation metric.

    The notebook uses this as a convenience selector. "Best" here means lowest
    saved validation loss, not playing strength. If no candidate records
    ``metric_name``, the newest completed or modified checkpoint is returned.
    """

    runs_path = Path(runs_dir)
    summaries = [
        _load_checkpoint_summary(path, metric_name)
        for path in _candidate_checkpoint_paths(runs_path, checkpoint_names)
    ]
    if not summaries:
        raise FileNotFoundError(f"No policy/value checkpoints found under {runs_path}")

    with_metric = [summary for summary in summaries if summary.metric_value is not None]
    if with_metric:
        return min(
            with_metric,
            key=lambda summary: (
                summary.metric_value,
                _checkpoint_name_rank(summary.path.name),
                -summary.epoch,
                -summary.modified_at,
                str(summary.path),
            ),
        ).path

    return max(
        summaries,
        key=lambda summary: (
            summary.completed_at,
            summary.modified_at,
            summary.epoch,
            -_checkpoint_name_rank(summary.path.name),
            str(summary.path),
        ),
    ).path


def load_policy_value_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> LoadedPolicyValueCheckpoint:
    """Load a `PolicyValueResNet` checkpoint written by supervised training."""

    checkpoint_path = Path(path)
    resolved_device = resolve_torch_device(device)
    raw = torch.load(checkpoint_path, map_location=resolved_device)
    if not isinstance(raw, dict):
        raise ValueError(f"{checkpoint_path} must contain a checkpoint dictionary")

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise ValueError(f"{checkpoint_path} missing model_config")
    model_config = ResNetConfig(**model_config_raw)

    state_dict = raw.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path} missing model_state_dict")

    model = PolicyValueResNet(model_config).to(resolved_device)
    model.load_state_dict(cast(dict[str, torch.Tensor], state_dict))
    model.eval()

    epoch = raw.get("epoch")
    metrics = raw.get("metrics")
    train_config = raw.get("train_config")
    metadata = CheckpointMetadata(
        path=checkpoint_path,
        epoch=epoch if isinstance(epoch, int) else None,
        saved_at=raw.get("saved_at") if isinstance(raw.get("saved_at"), str) else None,
        completed_at=raw.get("completed_at") if isinstance(raw.get("completed_at"), str) else None,
        metrics=metrics if isinstance(metrics, dict) else {},
        train_config=train_config if isinstance(train_config, dict) else {},
    )
    return LoadedPolicyValueCheckpoint(
        model=model,
        model_config=model_config,
        metadata=metadata,
        device=resolved_device,
    )


def _candidate_checkpoint_paths(root: Path, checkpoint_names: tuple[str, ...]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for name in checkpoint_names:
        for path in root.glob(f"**/{name}"):
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)
    return sorted(paths)


def _load_checkpoint_summary(path: Path, metric_name: str) -> _CheckpointSummary:
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a checkpoint dictionary")
    if not isinstance(raw.get("model_config"), dict) or not isinstance(
        raw.get("model_state_dict"),
        dict,
    ):
        raise ValueError(f"{path} is not a policy/value checkpoint")

    metrics = raw.get("metrics")
    metric_value = _finite_float(metrics.get(metric_name)) if isinstance(metrics, dict) else None
    completed_at = raw.get("completed_at")
    epoch = raw.get("epoch")
    return _CheckpointSummary(
        path=path,
        metric_value=metric_value,
        completed_at=completed_at if isinstance(completed_at, str) else "",
        epoch=epoch if isinstance(epoch, int) else -1,
        modified_at=path.stat().st_mtime,
    )


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _checkpoint_name_rank(name: str) -> int:
    if name == "checkpoint_latest.pt":
        return 0
    if name == "checkpoint.pt":
        return 1
    return 2
