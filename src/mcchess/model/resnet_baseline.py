"""ResNet-A: the single-board ResNet default."""

from __future__ import annotations

from mcchess.model.network import ResNetConfig
from mcchess.model.preset import ModelPreset

RESNET_BASELINE = ModelPreset(
    name="resnet_baseline",
    family="resnet",
    config=ResNetConfig(),
    description="Existing single-board ResNet default.",
)
