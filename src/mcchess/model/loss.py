"""Loss functions for supervised policy/value training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mcchess.board import POLICY_SIZE


@dataclass(frozen=True)
class PolicyValueLoss:
    """Policy, value, and weighted total loss tensors."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


def policy_value_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    *,
    value_weight: float = 1.0,
) -> PolicyValueLoss:
    """Return supervised cross-entropy policy loss plus weighted value MSE."""

    if value_weight < 0:
        raise ValueError("value_weight must be non-negative")
    _validate_loss_shapes(policy_logits, value, policy_targets, value_targets)

    policy_targets = policy_targets.to(device=policy_logits.device, dtype=torch.long)
    value_targets = value_targets.to(device=value.device, dtype=value.dtype)

    policy_loss = F.cross_entropy(policy_logits, policy_targets)
    value_loss = F.mse_loss(value, value_targets)
    total = policy_loss + value_weight * value_loss
    return PolicyValueLoss(total=total, policy=policy_loss, value=value_loss)


def _validate_loss_shapes(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
) -> None:
    if policy_logits.ndim != 2:
        raise ValueError("policy_logits must have shape [batch, 4672]")
    if policy_logits.shape[1] != POLICY_SIZE:
        raise ValueError(f"policy_logits must have shape [batch, {POLICY_SIZE}]")
    if value.ndim != 1:
        raise ValueError("value must have shape [batch]")
    if policy_targets.ndim != 1:
        raise ValueError("policy_targets must have shape [batch]")
    if value_targets.ndim != 1:
        raise ValueError("value_targets must have shape [batch]")

    batch_size = policy_logits.shape[0]
    if value.shape[0] != batch_size:
        raise ValueError("value batch size must match policy_logits")
    if policy_targets.shape[0] != batch_size:
        raise ValueError("policy_targets batch size must match policy_logits")
    if value_targets.shape[0] != batch_size:
        raise ValueError("value_targets batch size must match policy_logits")
