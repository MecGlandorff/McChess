"""Named model preset type."""

from __future__ import annotations

from dataclasses import dataclass

from mcchess.model.network import PolicyValueResNet, ResNetConfig


@dataclass(frozen=True)
class ModelPreset:
    """Named model configuration for reproducible architecture comparisons."""

    name: str
    family: str
    config: ResNetConfig
    description: str

    def build(self) -> PolicyValueResNet:
        """Build the model described by this preset."""

        if self.family != "resnet":
            raise ValueError(f"unsupported model family: {self.family}")
        return PolicyValueResNet(self.config)
