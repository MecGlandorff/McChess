"""Move indexing for the fixed 4672-action chess policy space."""

from __future__ import annotations

from typing import Final

import chess

BOARD_SQUARES: Final[int] = 64
MOVE_PLANES: Final[int] = 73
POLICY_SIZE: Final[int] = BOARD_SQUARES * MOVE_PLANES

# Planes 0-55: queen-like moves. Each direction has distances 1 through 7.
# Deltas are expressed as `(file_delta, rank_delta)` in python-chess square
# coordinates where rank increases from White's side toward Black's side.
QUEEN_DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),  # north
    (1, 1),  # north-east
    (1, 0),  # east
    (1, -1),  # south-east
    (0, -1),  # south
    (-1, -1),  # south-west
    (-1, 0),  # west
    (-1, 1),  # north-west
)

# Planes 56-63: knight moves.
KNIGHT_DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# Planes 64-72: underpromotions. Queen promotions are encoded as ordinary
# queen-like pawn moves and reconstructed as queen promotions when legal.
UNDERPROMOTION_PIECES: Final[tuple[chess.PieceType, ...]] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)
UNDERPROMOTION_DIRECTIONS: Final[tuple[str, ...]] = ("forward", "left", "right")

_QUEEN_DIRECTION_TO_INDEX: Final[dict[tuple[int, int], int]] = {
    direction: index for index, direction in enumerate(QUEEN_DIRECTIONS)
}
_KNIGHT_DIRECTION_TO_INDEX: Final[dict[tuple[int, int], int]] = {
    direction: index for index, direction in enumerate(KNIGHT_DIRECTIONS)
}
_UNDERPROMOTION_PIECE_TO_INDEX: Final[dict[chess.PieceType, int]] = {
    piece_type: index for index, piece_type in enumerate(UNDERPROMOTION_PIECES)
}


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Return the fixed policy index for a legal move.

    Raises:
        ValueError: If the move is not legal in `board` or cannot be represented
            in the 4672-action policy space.
    """

    if move not in board.legal_moves:
        raise ValueError(f"Move {move.uci()} is not legal in the given position")

    return _legal_move_to_index(board, move)


def legal_moves_with_policy_indices(board: chess.Board) -> list[tuple[chess.Move, int]]:
    """Enumerate legal moves and their policy indices in one legality pass."""

    return [(move, _legal_move_to_index(board, move)) for move in board.legal_moves]


def _legal_move_to_index(board: chess.Board, move: chess.Move) -> int:
    plane = _move_to_plane(board, move)
    return move.from_square * MOVE_PLANES + plane


def index_to_move(board: chess.Board, index: int) -> chess.Move | None:
    """Decode a policy index into a legal move for `board`, if possible."""

    if index < 0 or index >= POLICY_SIZE:
        return None

    from_square, plane = divmod(index, MOVE_PLANES)
    piece = board.piece_at(from_square)
    if piece is None or piece.color != board.turn:
        return None

    move = _plane_to_move(piece, from_square, plane)
    if move is None:
        return None

    if move in board.legal_moves:
        return move
    return None


def _move_to_plane(board: chess.Board, move: chess.Move) -> int:
    if move.promotion in _UNDERPROMOTION_PIECE_TO_INDEX:
        promotion_plane = _underpromotion_plane(board, move)
        if promotion_plane is None:
            raise ValueError(f"Underpromotion move {move.uci()} cannot be encoded")
        return promotion_plane

    return _queen_or_knight_plane(move)


def _queen_or_knight_plane(move: chess.Move) -> int:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank

    knight_direction_index = _KNIGHT_DIRECTION_TO_INDEX.get((file_delta, rank_delta))
    if knight_direction_index is not None:
        return 56 + knight_direction_index

    if file_delta == 0 or rank_delta == 0 or abs(file_delta) == abs(rank_delta):
        distance = max(abs(file_delta), abs(rank_delta))
        if 1 <= distance <= 7:
            direction = (_sign(file_delta), _sign(rank_delta))
            direction_index = _QUEEN_DIRECTION_TO_INDEX[direction]
            return direction_index * 7 + distance - 1

    raise ValueError(f"Move {move.uci()} cannot be encoded")


def _underpromotion_plane(board: chess.Board, move: chess.Move) -> int | None:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return None

    direction_index = _promotion_direction_index(piece.color, move)
    if direction_index is None or move.promotion is None:
        return None

    piece_index = _UNDERPROMOTION_PIECE_TO_INDEX[move.promotion]
    return 64 + piece_index * len(UNDERPROMOTION_DIRECTIONS) + direction_index


def _plane_to_move(piece: chess.Piece, from_square: chess.Square, plane: int) -> chess.Move | None:
    if 0 <= plane < 56:
        direction = QUEEN_DIRECTIONS[plane // 7]
        distance = plane % 7 + 1
        to_square = _offset_square(from_square, direction[0] * distance, direction[1] * distance)
        if to_square is None:
            return None

        promotion = None
        if piece.piece_type == chess.PAWN and _is_promotion_rank(piece.color, to_square):
            promotion = chess.QUEEN
        return chess.Move(from_square, to_square, promotion=promotion)

    if 56 <= plane < 64:
        direction = KNIGHT_DIRECTIONS[plane - 56]
        to_square = _offset_square(from_square, direction[0], direction[1])
        if to_square is None:
            return None
        return chess.Move(from_square, to_square)

    if 64 <= plane < MOVE_PLANES:
        if piece.piece_type != chess.PAWN:
            return None

        offset = plane - 64
        piece_index, direction_index = divmod(offset, len(UNDERPROMOTION_DIRECTIONS))
        promotion = UNDERPROMOTION_PIECES[piece_index]
        direction = _underpromotion_direction(piece.color, direction_index)
        to_square = _offset_square(from_square, direction[0], direction[1])
        if to_square is None:
            return None
        return chess.Move(from_square, to_square, promotion=promotion)

    return None


def _promotion_direction_index(color: chess.Color, move: chess.Move) -> int | None:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank

    expected_rank_delta = 1 if color == chess.WHITE else -1
    if rank_delta != expected_rank_delta:
        return None

    if file_delta == 0:
        return UNDERPROMOTION_DIRECTIONS.index("forward")
    if file_delta == (-1 if color == chess.WHITE else 1):
        return UNDERPROMOTION_DIRECTIONS.index("left")
    if file_delta == (1 if color == chess.WHITE else -1):
        return UNDERPROMOTION_DIRECTIONS.index("right")
    return None


def _underpromotion_direction(color: chess.Color, direction_index: int) -> tuple[int, int]:
    direction = UNDERPROMOTION_DIRECTIONS[direction_index]
    rank_delta = 1 if color == chess.WHITE else -1

    if direction == "forward":
        file_delta = 0
    elif direction == "left":
        file_delta = -1 if color == chess.WHITE else 1
    else:
        file_delta = 1 if color == chess.WHITE else -1

    return file_delta, rank_delta


def _offset_square(from_square: chess.Square, file_delta: int, rank_delta: int) -> chess.Square | None:
    to_file = chess.square_file(from_square) + file_delta
    to_rank = chess.square_rank(from_square) + rank_delta
    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return None
    return chess.square(to_file, to_rank)


def _is_promotion_rank(color: chess.Color, square: chess.Square) -> bool:
    rank = chess.square_rank(square)
    return rank == (7 if color == chess.WHITE else 0)


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
