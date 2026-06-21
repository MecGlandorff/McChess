"""ResNet-C: a larger BatchNorm single-board ResNet."""

from __future__ import annotations

from mcchess.model.network import ResNetConfig
from mcchess.model.preset import ModelPreset

RESNET_C = ModelPreset(
    name="resnet_c",
    family="resnet",
    config=ResNetConfig(
        channels=128,
        num_blocks=10,
        value_hidden_dim=256,
        normalization="batchnorm",
    ),
    description="Larger single-board ResNet with BatchNorm for a measured ablation.",
)
