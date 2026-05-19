import chess
import numpy as np

from mcchess.board import BOARD_TENSOR_SHAPE, PLANE_NAMES, encode_board, square_to_tensor_coords


def test_initial_position_shape_and_dtype() -> None:
    tensor = encode_board(chess.Board())

    assert tensor.shape == BOARD_TENSOR_SHAPE
    assert tensor.shape == (18, 8, 8)
    assert tensor.dtype == np.float32


def test_initial_position_piece_counts() -> None:
    tensor = encode_board(chess.Board())

    expected_counts = {
        "white_pawn": 8,
        "white_knight": 2,
        "white_bishop": 2,
        "white_rook": 2,
        "white_queen": 1,
        "white_king": 1,
        "black_pawn": 8,
        "black_knight": 2,
        "black_bishop": 2,
        "black_rook": 2,
        "black_queen": 1,
        "black_king": 1,
    }

    for plane_name, expected_count in expected_counts.items():
        plane_index = PLANE_NAMES.index(plane_name)
        assert int(tensor[plane_index].sum()) == expected_count


def test_square_orientation_matches_board_diagram() -> None:
    tensor = encode_board(chess.Board())

    assert square_to_tensor_coords(chess.A8) == (0, 0)
    assert square_to_tensor_coords(chess.H8) == (0, 7)
    assert square_to_tensor_coords(chess.A1) == (7, 0)
    assert square_to_tensor_coords(chess.H1) == (7, 7)

    assert tensor[PLANE_NAMES.index("white_rook"), 7, 0] == 1.0
    assert tensor[PLANE_NAMES.index("white_king"), 7, 4] == 1.0
    assert tensor[PLANE_NAMES.index("black_rook"), 0, 7] == 1.0
    assert tensor[PLANE_NAMES.index("black_king"), 0, 4] == 1.0


def test_side_to_move_plane_changes_after_move() -> None:
    board = chess.Board()

    white_to_move = encode_board(board)[PLANE_NAMES.index("white_to_move")]
    assert np.all(white_to_move == 1.0)

    board.push(chess.Move.from_uci("e2e4"))
    black_to_move = encode_board(board)[PLANE_NAMES.index("white_to_move")]
    assert np.all(black_to_move == 0.0)


def test_castling_right_planes_update_after_king_move() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    before = encode_board(board)
    assert np.all(before[PLANE_NAMES.index("white_kingside_castling")] == 1.0)
    assert np.all(before[PLANE_NAMES.index("white_queenside_castling")] == 1.0)
    assert np.all(before[PLANE_NAMES.index("black_kingside_castling")] == 1.0)
    assert np.all(before[PLANE_NAMES.index("black_queenside_castling")] == 1.0)

    board.push(chess.Move.from_uci("e1f1"))
    after = encode_board(board)

    assert np.all(after[PLANE_NAMES.index("white_kingside_castling")] == 0.0)
    assert np.all(after[PLANE_NAMES.index("white_queenside_castling")] == 0.0)
    assert np.all(after[PLANE_NAMES.index("black_kingside_castling")] == 1.0)
    assert np.all(after[PLANE_NAMES.index("black_queenside_castling")] == 1.0)


def test_initial_position_has_empty_en_passant_plane() -> None:
    tensor = encode_board(chess.Board())

    assert np.all(tensor[PLANE_NAMES.index("en_passant_target")] == 0.0)


def test_legal_en_passant_target_square_is_encoded() -> None:
    board = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push(chess.Move.from_uci(uci))

    tensor = encode_board(board)
    plane = tensor[PLANE_NAMES.index("en_passant_target")]
    row, col = square_to_tensor_coords(chess.D6)

    assert board.has_legal_en_passant()
    assert board.ep_square == chess.D6
    assert plane[row, col] == 1.0
    assert int(plane.sum()) == 1


def test_en_passant_plane_disambiguates_identical_piece_placement() -> None:
    board = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push(chess.Move.from_uci(uci))

    fen_parts = board.fen().split()
    fen_parts[3] = "-"
    without_ep_right = chess.Board(" ".join(fen_parts))

    with_ep_tensor = encode_board(board)
    without_ep_tensor = encode_board(without_ep_right)
    ep_plane_index = PLANE_NAMES.index("en_passant_target")

    assert board.board_fen() == without_ep_right.board_fen()
    assert board.turn == without_ep_right.turn
    assert board.castling_rights == without_ep_right.castling_rights
    assert np.array_equal(with_ep_tensor[:ep_plane_index], without_ep_tensor[:ep_plane_index])
    assert not np.array_equal(with_ep_tensor[ep_plane_index], without_ep_tensor[ep_plane_index])


def test_stale_en_passant_square_is_not_encoded() -> None:
    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )

    tensor = encode_board(board)
    plane = tensor[PLANE_NAMES.index("en_passant_target")]

    assert board.ep_square == chess.E3
    assert not board.has_legal_en_passant()
    assert np.all(plane == 0.0)
