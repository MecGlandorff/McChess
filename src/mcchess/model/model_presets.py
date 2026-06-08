"""Named policy/value model presets."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

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


RESNET_BASELINE = ModelPreset(
    name="resnet_baseline",
    family="resnet",
    config=ResNetConfig(),
    description="Existing single-board ResNet default.",
)

RESNET_B = ModelPreset(
    name="resnet_b",
    family="resnet",
    config=ResNetConfig(channels=64, num_blocks=6, value_hidden_dim=128),
    description="Deeper compact single-board ResNet for the next controlled baseline.",
)

_CANONICAL_PRESETS = (RESNET_BASELINE, RESNET_B)
_MODEL_PRESETS_BY_NAME = MappingProxyType(
    {
        RESNET_BASELINE.name: RESNET_BASELINE,
        "baseline": RESNET_BASELINE,
        RESNET_B.name: RESNET_B,
        "resnet-b": RESNET_B,
    }
)


def available_model_presets() -> tuple[str, ...]:
    """Return canonical preset names in display order."""

    return tuple(preset.name for preset in _CANONICAL_PRESETS)


def get_model_preset(name: str) -> ModelPreset:
    """Return a named model preset, accepting stable hyphen/underscore aliases."""

    normalized = name.strip().lower()
    try:
        return _MODEL_PRESETS_BY_NAME[normalized]
    except KeyError as exc:
        expected = ", ".join(available_model_presets())
        raise ValueError(f"unknown model preset {name!r}; expected one of: {expected}") from exc


def build_model_from_preset(name: str) -> PolicyValueResNet:
    """Build a policy/value model from a named preset."""

    return get_model_preset(name).build()
