"""Board encoding, move indexing, and legal masking utilities."""

from mcchess.board.encoding import (
    BOARD_PLANE_COUNT,
    BOARD_TENSOR_SHAPE,
    METADATA_PLANE_NAMES,
    PIECE_PLANE_NAMES,
    PLANE_NAMES,
    encode_board,
    square_to_tensor_coords,
)
from mcchess.board.legal_mask import legal_policy_mask
from mcchess.board.move_index import (
    MOVE_PLANES,
    POLICY_SIZE,
    UNDERPROMOTION_DIRECTIONS,
    UNDERPROMOTION_PIECES,
    index_to_move,
    legal_moves_with_policy_indices,
    move_to_index,
)

__all__ = [
    "BOARD_PLANE_COUNT",
    "BOARD_TENSOR_SHAPE",
    "METADATA_PLANE_NAMES",
    "MOVE_PLANES",
    "PIECE_PLANE_NAMES",
    "PLANE_NAMES",
    "POLICY_SIZE",
    "UNDERPROMOTION_DIRECTIONS",
    "UNDERPROMOTION_PIECES",
    "encode_board",
    "index_to_move",
    "legal_moves_with_policy_indices",
    "legal_policy_mask",
    "move_to_index",
    "square_to_tensor_coords",
]
