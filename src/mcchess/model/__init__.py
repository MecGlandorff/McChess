"""Policy/value neural network models and losses."""

from mcchess.model.checkpoint import (
    CheckpointMetadata,
    LoadedPolicyValueCheckpoint,
    load_policy_value_checkpoint,
    resolve_torch_device,
)
from mcchess.model.loss import PolicyValueLoss, policy_value_loss
from mcchess.model.network import PolicyValueResNet, ResNetConfig, ResidualBlock
from mcchess.model.model_presets import (
    RESNET_B,
    RESNET_BASELINE,
    ModelPreset,
    available_model_presets,
    build_model_from_preset,
    get_model_preset,
)

__all__ = [
    "CheckpointMetadata",
    "LoadedPolicyValueCheckpoint",
    "ModelPreset",
    "PolicyValueLoss",
    "PolicyValueResNet",
    "RESNET_B",
    "RESNET_BASELINE",
    "ResNetConfig",
    "ResidualBlock",
    "available_model_presets",
    "build_model_from_preset",
    "get_model_preset",
    "load_policy_value_checkpoint",
    "policy_value_loss",
    "resolve_torch_device",
]
