"""Model preset registry. Each preset is defined in its own module."""

from __future__ import annotations

from types import MappingProxyType

from mcchess.model.network import PolicyValueResNet
from mcchess.model.preset import ModelPreset
from mcchess.model.resnet_b import RESNET_B
from mcchess.model.resnet_baseline import RESNET_BASELINE

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
