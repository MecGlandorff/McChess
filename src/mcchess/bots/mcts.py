"""Neural MCTS bot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from mcchess.bots.base import legal_moves_or_raise
from mcchess.model import LoadedPolicyValueCheckpoint, load_policy_value_checkpoint
from mcchess.search import MCTSConfig, MCTSSearch


@dataclass(frozen=True)
class MCTSBot:
    """Choose moves by running fixed-budget PUCT search from a checkpoint."""

    checkpoint: LoadedPolicyValueCheckpoint
    config: MCTSConfig = MCTSConfig()
    name: str = "mcts"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "auto",
        name: str = "mcts",
        simulations: int = 50,
        c_puct: float = 1.5,
        inference_batch_size: int = 1,
    ) -> "MCTSBot":
        loaded = load_policy_value_checkpoint(checkpoint_path, device=device)
        return cls(
            checkpoint=loaded,
            config=MCTSConfig(
                simulations=simulations,
                c_puct=c_puct,
                inference_batch_size=inference_batch_size,
            ),
            name=name,
        )

    @property
    def device(self) -> torch.device:
        return self.checkpoint.device

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves_or_raise(board)
        search = MCTSSearch(self.checkpoint.model, self.device, self.config)
        return search.search(board).move
