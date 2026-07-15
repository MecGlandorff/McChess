"""Legal policy masks for the fixed chess policy space."""

from __future__ import annotations

import chess
import numpy as np

from mcchess.board.move_index import POLICY_SIZE, legal_moves_with_policy_indices


def legal_policy_mask(board: chess.Board) -> np.ndarray:
    """Return a float32 mask with `1.0` at legal move indices."""

    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    for _, policy_index in legal_moves_with_policy_indices(board):
        mask[policy_index] = 1.0
    return mask
