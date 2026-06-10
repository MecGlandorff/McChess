"""ResNet-B: a deeper compact single-board ResNet."""

from __future__ import annotations

from mcchess.model.network import ResNetConfig
from mcchess.model.preset import ModelPreset

RESNET_B = ModelPreset(
    name="resnet_b",
    family="resnet",
    config=ResNetConfig(channels=64, num_blocks=6, value_hidden_dim=128),
    description="Deeper compact single-board ResNet for the next controlled baseline.",
)
