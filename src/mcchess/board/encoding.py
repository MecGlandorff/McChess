"""Board tensor encoding.

The single-board encoder uses 17 planes:

- 12 piece planes in white-then-black, pawn/knight/bishop/rook/queen/king order
- 1 side-to-move metadata plane
- 4 castling-right metadata planes

Square orientation is board-diagram orientation: tensor row 0 is rank 8,
row 7 is rank 1, column 0 is file a, and column 7 is file h.
"""

from __future__ import annotations

from typing import Final

import chess
import numpy as np

PIECE_PLANE_NAMES: Final[tuple[str, ...]] = (
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
)

METADATA_PLANE_NAMES: Final[tuple[str, ...]] = (
    "white_to_move",
    "white_kingside_castling",
    "white_queenside_castling",
    "black_kingside_castling",
    "black_queenside_castling",
)

PLANE_NAMES: Final[tuple[str, ...]] = PIECE_PLANE_NAMES + METADATA_PLANE_NAMES
BOARD_PLANE_COUNT: Final[int] = len(PLANE_NAMES)
BOARD_TENSOR_SHAPE: Final[tuple[int, int, int]] = (BOARD_PLANE_COUNT, 8, 8)

_PIECE_TO_PLANE: Final[dict[tuple[chess.Color, chess.PieceType], int]] = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def square_to_tensor_coords(square: chess.Square) -> tuple[int, int]:
    """Return `(row, col)` for a python-chess square.

    The returned coordinates follow board-diagram orientation:
    `a8 -> (0, 0)`, `h8 -> (0, 7)`, `a1 -> (7, 0)`, `h1 -> (7, 7)`.
    """

    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)
    return row, col


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a `python-chess` board as a `float32` tensor.

    Piece planes contain one-hot occupancy. Metadata planes are filled with
    either `1.0` or `0.0` across all 64 squares.
    """

    tensor = np.zeros(BOARD_TENSOR_SHAPE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = _PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        row, col = square_to_tensor_coords(square)
        tensor[plane, row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    return tensor
