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

MATE_SCORE: Final[int] = 1_000_000
_INFINITY: Final[int] = 2 * MATE_SCORE


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
            return MATE_SCORE
        return material_balance(board, color)
    finally:
        board.pop()


@dataclass(frozen=True)
class NegamaxBot:
    """Fixed-depth negamax with alpha-beta pruning over material balance.

    Mates are scored by distance so faster mates win. Known limitation: there
    is no quiescence search, so captures at the horizon are not resolved and
    the bot shows horizon effects at low depth. Draw rules beyond stalemate
    are ignored.
    """

    depth: int = 2
    name: str = "negamax"

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least 1")

    def choose_move(self, board: chess.Board) -> chess.Move:
        legal_moves = legal_moves_or_raise(board)
        best_move = legal_moves[0]
        alpha = -_INFINITY
        for move in sorted(legal_moves, key=chess.Move.uci):
            board.push(move)
            try:
                score = -_negamax(board, self.depth - 1, -_INFINITY, -alpha)
            finally:
                board.pop()
            if score > alpha:
                best_move = move
                alpha = score
        return best_move


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return -(MATE_SCORE - board.ply()) if board.is_check() else 0
    if depth == 0:
        return material_balance(board, board.turn)

    value = -_INFINITY
    for move in legal_moves:
        board.push(move)
        try:
            value = max(value, -_negamax(board, depth - 1, -beta, -alpha))
        finally:
            board.pop()
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return value
