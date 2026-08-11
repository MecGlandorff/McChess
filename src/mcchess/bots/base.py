"""Shared bot interfaces."""

from __future__ import annotations

from typing import Protocol

import chess

from mcchess.board.legal_moves import NoLegalMoveError as NoLegalMoveError
from mcchess.board.legal_moves import legal_moves_or_raise as legal_moves_or_raise


class Bot(Protocol):
    """Chess bot interface used by play and evaluation utilities."""

    @property
    def name(self) -> str:
        """Human-readable bot identifier."""
        ...

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Choose a legal move for the current board."""
