"""Legal policy masks for the fixed chess policy space."""

from __future__ import annotations

import chess
import numpy as np

from mcchess.board.move_index import POLICY_SIZE, move_to_index


def legal_policy_mask(board: chess.Board) -> np.ndarray:
    """Return a float32 mask with `1.0` at legal move indices."""

    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        mask[move_to_index(board, move)] = 1.0
    return mask
