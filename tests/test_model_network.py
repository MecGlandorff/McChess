import pytest
import torch

from mcchess.board import BOARD_PLANE_COUNT, POLICY_SIZE
from mcchess.model import PolicyValueResNet, ResNetConfig, policy_value_loss


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
