"""Compact policy/value ResNet baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn

from mcchess.board import BOARD_PLANE_COUNT, POLICY_SIZE

NormalizationName = Literal["none", "batchnorm"]


@dataclass(frozen=True)
class ResNetConfig:
    """Configuration for the compact single-board policy/value ResNet."""

    input_planes: int = BOARD_PLANE_COUNT
    channels: int = 64
    num_blocks: int = 4
    policy_size: int = POLICY_SIZE
    value_hidden_dim: int = 128
    normalization: NormalizationName = "none"

    def __post_init__(self) -> None:
        if self.input_planes <= 0:
            raise ValueError("input_planes must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.policy_size <= 0:
            raise ValueError("policy_size must be positive")
        if self.value_hidden_dim <= 0:
            raise ValueError("value_hidden_dim must be positive")
        if self.normalization not in ("none", "batchnorm"):
            raise ValueError("normalization must be 'none' or 'batchnorm'")


class ResidualBlock(nn.Module):
    """Small 8x8 residual block used by the baseline network."""

    def __init__(self, channels: int, normalization: NormalizationName = "none") -> None:
        super().__init__()
        if normalization == "none":
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.relu(x + self.net(x)))


class PolicyValueResNet(nn.Module):
    """Single-board ResNet with policy logits and side-to-move value output."""

    def __init__(self, config: ResNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResNetConfig()

        self.stem = _make_stem(
            self.config.input_planes,
            self.config.channels,
            self.config.normalization,
        )
        self.tower = nn.Sequential(
            *(
                ResidualBlock(self.config.channels, self.config.normalization)
                for _ in range(self.config.num_blocks)
            )
        )
        self.policy_head = _make_policy_head(
            self.config.channels,
            self.config.policy_size,
            self.config.normalization,
        )
        self.value_head = _make_value_head(
            self.config.channels,
            self.config.value_hidden_dim,
            self.config.normalization,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(policy_logits, value)` with shapes `[batch, 4672]` and `[batch]`."""

        self._validate_input(x)
        features = self.tower(self.stem(x))
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError("input must have shape [batch, planes, 8, 8]")
        if x.shape[1] != self.config.input_planes or x.shape[2] != 8 or x.shape[3] != 8:
            raise ValueError(
                f"input must have shape [batch, {self.config.input_planes}, 8, 8]"
            )


def _make_stem(
    input_planes: int,
    channels: int,
    normalization: NormalizationName,
) -> nn.Sequential:
    if normalization == "none":
        return nn.Sequential(
            nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
    return nn.Sequential(
        nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
    )


def _make_policy_head(
    channels: int,
    policy_size: int,
    normalization: NormalizationName,
) -> nn.Sequential:
    if normalization == "none":
        return nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, policy_size),
        )
    return nn.Sequential(
        nn.Conv2d(channels, 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2 * 8 * 8, policy_size),
    )


def _make_value_head(
    channels: int,
    value_hidden_dim: int,
    normalization: NormalizationName,
) -> nn.Sequential:
    if normalization == "none":
        return nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
            nn.Tanh(),
        )
    return nn.Sequential(
        nn.Conv2d(channels, 1, kernel_size=1, bias=False),
        nn.BatchNorm2d(1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8, value_hidden_dim),
        nn.ReLU(),
        nn.Linear(value_hidden_dim, 1),
        nn.Tanh(),
    )
