"""Simple non-neural chess bots."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Final

import chess

from mcchess.bots.base import legal_moves_or_raise

PIECE_VALUES: Final[dict[chess.PieceType, int]] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@dataclass
class RandomLegalBot:
    """Choose uniformly from legal moves using a local RNG."""

    seed: int = 0
    name: str = "random"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = legal_moves_or_raise(board)
        return self._rng.choice(legal_moves)


@dataclass(frozen=True)
class MaterialBot:
    """Choose the legal move with the best one-ply material balance."""

    name: str = "material"

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = legal_moves_or_raise(board)
        mover = board.turn
        best_move = legal_moves[0]
        best_score = _score_after_move(board, best_move, mover)

        for move in legal_moves[1:]:
            score = _score_after_move(board, move, mover)
            if score > best_score or (score == best_score and move.uci() < best_move.uci()):
                best_move = move
                best_score = score
        return best_move


def material_balance(board: chess.Board, color: chess.Color) -> int:
    """Return material balance from `color`'s perspective."""

    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color == color else -value
    return score


def _score_after_move(board: chess.Board, move: chess.Move, color: chess.Color) -> int:
    board.push(move)
    try:
        if board.is_checkmate():
            return 1_000_000
        return material_balance(board, color)
    finally:
        board.pop()
