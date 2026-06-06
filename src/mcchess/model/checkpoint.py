"""Checkpoint loading helpers for policy/value models."""

from __future__ import annotations

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
