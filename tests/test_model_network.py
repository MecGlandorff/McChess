import pytest
import torch
from torch import nn

from mcchess.board import BOARD_PLANE_COUNT, POLICY_SIZE
from mcchess.model import (
    RESNET_A,
    RESNET_B,
    RESNET_C,
    PolicyValueResNet,
    ResNetConfig,
    available_model_presets,
    build_model_from_preset,
    get_model_preset,
    policy_value_loss,
)


def small_config() -> ResNetConfig:
    return ResNetConfig(channels=16, num_blocks=2, value_hidden_dim=32)


def test_policy_value_resnet_output_shapes_and_bounds() -> None:
    model = PolicyValueResNet(small_config())
    x = torch.randn(3, BOARD_PLANE_COUNT, 8, 8)

    policy_logits, value = model(x)

    assert policy_logits.shape == (3, POLICY_SIZE)
    assert value.shape == (3,)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_policy_value_resnet_backward_smoke() -> None:
    model = PolicyValueResNet(small_config())
    x = torch.randn(2, BOARD_PLANE_COUNT, 8, 8)
    policy_targets = torch.tensor([0, POLICY_SIZE - 1])
    value_targets = torch.tensor([1.0, -1.0])

    policy_logits, value = model(x)
    loss = policy_value_loss(policy_logits, value, policy_targets, value_targets)
    loss.total.backward()

    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients
    assert all(grad is not None for grad in gradients)
    assert all(torch.isfinite(grad).all() for grad in gradients if grad is not None)


def test_policy_value_resnet_rejects_invalid_input_shape() -> None:
    model = PolicyValueResNet(small_config())

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(BOARD_PLANE_COUNT, 8, 8))

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, BOARD_PLANE_COUNT - 1, 8, 8))


def test_resnet_config_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        ResNetConfig(channels=0)
    with pytest.raises(ValueError):
        ResNetConfig(num_blocks=0)
    with pytest.raises(ValueError):
        ResNetConfig(normalization="layernorm")  # type: ignore[arg-type]


def test_default_resnet_keeps_no_normalization_baseline() -> None:
    model = PolicyValueResNet(small_config())

    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())


def test_resnet_b_preset_has_policy_value_contract() -> None:
    assert RESNET_B.config.channels == 64
    assert RESNET_B.config.num_blocks == 6
    assert RESNET_B.config.value_hidden_dim == 128
    assert RESNET_B.config.normalization == "none"

    model = build_model_from_preset("resnet_b")
    x = torch.randn(1, BOARD_PLANE_COUNT, 8, 8)

    with torch.no_grad():
        policy_logits, value = model(x)

    assert policy_logits.shape == (1, POLICY_SIZE)
    assert value.shape == (1,)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()


def test_resnet_c_preset_has_batchnorm_policy_value_contract() -> None:
    assert RESNET_C.config.channels == 128
    assert RESNET_C.config.num_blocks == 10
    assert RESNET_C.config.value_hidden_dim == 256
    assert RESNET_C.config.normalization == "batchnorm"

    model = build_model_from_preset("resnet_c")
    x = torch.randn(2, BOARD_PLANE_COUNT, 8, 8)

    with torch.no_grad():
        policy_logits, value = model(x)

    assert any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert policy_logits.shape == (2, POLICY_SIZE)
    assert value.shape == (2,)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()


def test_model_preset_lookup_uses_canonical_names_and_aliases() -> None:
    assert available_model_presets() == ("resnet_a", "resnet_b", "resnet_c")
    assert get_model_preset("resnet-a") is RESNET_A
    assert get_model_preset("resnet_b") is RESNET_B
    assert get_model_preset("resnet-b") is RESNET_B
    assert get_model_preset("resnet_c") is RESNET_C
    assert get_model_preset("resnet-c") is RESNET_C

    with pytest.raises(ValueError, match="unknown model preset"):
        get_model_preset("missing")
