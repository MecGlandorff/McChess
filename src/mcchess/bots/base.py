"""Shared bot interfaces."""

from __future__ import annotations

from typing import Protocol

import chess


class NoLegalMoveError(ValueError):
    """Raised when a bot is asked to move from a terminal position."""


class Bot(Protocol):
    """Chess bot interface used by play and evaluation utilities."""

    @property
    def name(self) -> str:
        """Human-readable bot identifier."""
        ...

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose a legal move for the current board."""


def legal_moves_or_raise(board: chess.Board) -> list[chess.Move]:
    """Return legal moves or raise when the position is terminal."""

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise NoLegalMoveError("board has no legal moves")
    return legal_moves
