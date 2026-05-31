import pytest
import torch

from mcchess.board import POLICY_SIZE
from mcchess.model import policy_value_loss


def test_policy_value_loss_returns_weighted_components() -> None:
    policy_logits = torch.zeros(2, POLICY_SIZE, requires_grad=True)
    value = torch.tensor([0.25, -0.5], requires_grad=True)
    policy_targets = torch.tensor([0, POLICY_SIZE - 1])
    value_targets = torch.tensor([1.0, -1.0])

    loss = policy_value_loss(
        policy_logits,
        value,
        policy_targets,
        value_targets,
        value_weight=0.5,
    )

    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.policy)
    assert torch.isfinite(loss.value)
    assert torch.allclose(loss.total, loss.policy + 0.5 * loss.value)

    loss.total.backward()
    assert policy_logits.grad is not None
    assert value.grad is not None


def test_policy_value_loss_rejects_invalid_shapes() -> None:
    policy_logits = torch.zeros(2, POLICY_SIZE)
    value = torch.zeros(2)
    policy_targets = torch.zeros(2, dtype=torch.long)
    value_targets = torch.zeros(2)

    with pytest.raises(ValueError, match="policy_logits"):
        policy_value_loss(policy_logits.view(2, 1, POLICY_SIZE), value, policy_targets, value_targets)

    with pytest.raises(ValueError, match="value"):
        policy_value_loss(policy_logits, value.view(2, 1), policy_targets, value_targets)

    with pytest.raises(ValueError, match="policy_targets"):
        policy_value_loss(policy_logits, value, policy_targets.view(2, 1), value_targets)

    with pytest.raises(ValueError, match="value_targets"):
        policy_value_loss(policy_logits, value, policy_targets, value_targets.view(2, 1))


def test_policy_value_loss_rejects_negative_value_weight() -> None:
    with pytest.raises(ValueError, match="value_weight"):
        policy_value_loss(
            torch.zeros(1, POLICY_SIZE),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1),
            value_weight=-1.0,
        )
