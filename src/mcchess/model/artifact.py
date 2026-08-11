"""Export compact, auditable inference artifacts from training checkpoints."""

from __future__ import annotations

import datetime as dt
import hashlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import torch

from mcchess.model.network import PolicyValueResNet, ResNetConfig

ARTIFACT_SCHEMA_VERSION = 1
ABSOLUTE_PATH_FIELD_NAMES = frozenset({"resume_from_checkpoint"})
INFERENCE_CHECKPOINT_FIELDS = (
    "model_state_dict",
    "model_config",
    "epoch",
    "metrics",
    "train_config",
    "saved_at",
    "completed_at",
)


@dataclass(frozen=True)
class ExportedModelArtifact:
    """Paths and hashes produced by an inference-artifact export."""

    artifact_id: str
    source_path: Path
    output_path: Path
    source_sha256: str
    output_sha256: str
    output_size_bytes: int


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_inference_artifact(
    source_path: str | Path,
    output_path: str | Path,
    *,
    artifact_id: str,
    overwrite: bool = False,
    exported_at: str | None = None,
) -> ExportedModelArtifact:
    """Export model state and audit metadata without optimizer or resume state."""

    source = Path(source_path)
    output = Path(output_path)
    if not artifact_id.strip():
        raise ValueError("artifact_id must not be empty")
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("source and output checkpoint paths must differ")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")

    raw = torch.load(source, map_location="cpu", weights_only=True)
    checkpoint = _validated_checkpoint(raw, source)
    source_sha256 = sha256_file(source)
    timestamp = exported_at or dt.datetime.now(dt.timezone.utc).isoformat()
    artifact = {
        field: checkpoint.get(field)
        for field in INFERENCE_CHECKPOINT_FIELDS
    }
    artifact.update(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "provenance": {
                "source_checkpoint_sha256": source_sha256,
                "exported_at": timestamp,
            },
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            # Passing the handle keeps PyTorch's internal archive root stable.
            # Passing a random temporary pathname would embed that name and
            # produce a different checksum for otherwise identical exports.
            torch.save(artifact, temporary_file)
            temporary_file.flush()
        _validated_inference_artifact(temporary_path, artifact_id=artifact_id)
        temporary_path.replace(output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ExportedModelArtifact(
        artifact_id=artifact_id,
        source_path=source,
        output_path=output,
        source_sha256=source_sha256,
        output_sha256=sha256_file(output),
        output_size_bytes=output.stat().st_size,
    )


def _validated_checkpoint(raw: object, path: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a checkpoint dictionary")

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise ValueError(f"{path} missing model_config")
    model_config = ResNetConfig(**model_config_raw)

    state_dict_raw = raw.get("model_state_dict")
    if not isinstance(state_dict_raw, Mapping) or not state_dict_raw:
        raise ValueError(f"{path} missing model_state_dict")
    state_dict = cast(Mapping[str, object], state_dict_raw)
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{path} has invalid model tensor {name!r}")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"{path} model tensor {name!r} contains non-finite values")

    model = PolicyValueResNet(model_config)
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict_raw))
    model.eval()
    inputs = torch.zeros(1, model_config.input_planes, 8, 8)
    with torch.inference_mode():
        policy_logits, value = model(inputs)
    if policy_logits.shape != (1, model_config.policy_size) or value.shape != (1,):
        raise ValueError(f"{path} model outputs have unexpected shapes")
    if not torch.isfinite(policy_logits).all() or not torch.isfinite(value).all():
        raise ValueError(f"{path} model outputs contain non-finite values")

    epoch = raw.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"{path} missing a valid epoch")
    for field in ("metrics", "train_config"):
        if not isinstance(raw.get(field), dict):
            raise ValueError(f"{path} missing {field}")
    for field in ("saved_at", "completed_at"):
        value = raw.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{path} has invalid {field}")

    absolute_field = _first_absolute_path_field(cast(dict[str, Any], raw["train_config"]))
    if absolute_field is not None:
        raise ValueError(
            f"{path} train_config contains machine-absolute path at {absolute_field}"
        )
    return cast(dict[str, Any], raw)


def _validated_inference_artifact(path: Path, *, artifact_id: str) -> None:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint = _validated_checkpoint(raw, path)
    if checkpoint.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"{path} has unsupported artifact_schema_version")
    if checkpoint.get("artifact_id") != artifact_id:
        raise ValueError(f"{path} has unexpected artifact_id")
    if "optimizer_state_dict" in checkpoint or "global_step" in checkpoint:
        raise ValueError(f"{path} contains training-only resume state")


def _first_absolute_path_field(value: object, prefix: str = "train_config") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            field = f"{prefix}.{key}"
            is_path_field = isinstance(key, str) and (
                key.endswith(("_path", "_dir")) or key in ABSOLUTE_PATH_FIELD_NAMES
            )
            if is_path_field and isinstance(child, str):
                if PureWindowsPath(child).is_absolute() or PurePosixPath(child).is_absolute():
                    return field
            nested = _first_absolute_path_field(child, field)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            nested = _first_absolute_path_field(child, f"{prefix}[{index}]")
            if nested is not None:
                return nested
    return None
