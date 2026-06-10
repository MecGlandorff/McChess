"""ResNet-A: the single-board ResNet default."""

from __future__ import annotations

from mcchess.model.network import ResNetConfig
from mcchess.model.preset import ModelPreset

RESNET_A = ModelPreset(
    name="resnet_a",
    family="resnet",
    config=ResNetConfig(),
    description="Single-board ResNet default.",
)
