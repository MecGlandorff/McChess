"""Neural policy-only bot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from mcchess.board import encode_board, index_to_move, legal_policy_mask
from mcchess.bots.base import legal_moves_or_raise
from mcchess.model import LoadedPolicyValueCheckpoint, load_policy_value_checkpoint


@dataclass(frozen=True)
class PolicyOnlyBot:
    """Choose the highest-logit legal move from a policy/value checkpoint."""

    checkpoint: LoadedPolicyValueCheckpoint
    name: str = "policy_only"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "auto",
        name: str = "policy_only",
    ) -> "PolicyOnlyBot":
        loaded = load_policy_value_checkpoint(checkpoint_path, device=device)
        return cls(checkpoint=loaded, name=name)

    @property
    def device(self) -> torch.device:
        return self.checkpoint.device

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves_or_raise(board)

        encoded = encode_board(board)
        board_tensor = torch.from_numpy(encoded).unsqueeze(0).to(
            device=self.device,
            dtype=torch.float32,
        )
        mask_tensor = torch.from_numpy(legal_policy_mask(board).astype(bool)).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.checkpoint.model(board_tensor)
            logits = policy_logits.squeeze(0)

        if not torch.isfinite(logits).all():
            raise ValueError("policy logits contain non-finite values")

        masked_logits = logits.masked_fill(~mask_tensor, -torch.inf)
        policy_index = int(torch.argmax(masked_logits).item())
        move = index_to_move(board, policy_index)
        if move is None:
            raise RuntimeError("masked policy selected an undecodable move")
        return move
