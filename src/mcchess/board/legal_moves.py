"""Legal-move helpers shared by bots and search."""

from __future__ import annotations

import chess


class NoLegalMoveError(ValueError):
    """Raised when move selection is requested from a terminal position."""


def legal_moves_or_raise(board: chess.Board) -> list[chess.Move]:
    """Return legal moves or raise when the position is terminal."""

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise NoLegalMoveError("board has no legal moves")
    return legal_moves
