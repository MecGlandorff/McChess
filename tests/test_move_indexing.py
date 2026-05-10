import random

import chess
import numpy as np
import pytest

from mcchess.board import POLICY_SIZE, index_to_move, legal_policy_mask, move_to_index


def assert_legal_moves_round_trip(board: chess.Board) -> None:
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        index = move_to_index(board, move)
        assert 0 <= index < POLICY_SIZE
        assert index_to_move(board, index) == move


def assert_mask_matches_legal_moves(board: chess.Board) -> None:
    mask = legal_policy_mask(board)
    legal_moves = list(board.legal_moves)
    legal_indices = {move_to_index(board, move) for move in legal_moves}

    assert mask.shape == (POLICY_SIZE,)
    assert mask.dtype == np.float32
    assert int(mask.sum()) == len(legal_moves)

    mask_indices = set(np.flatnonzero(mask).tolist())
    assert mask_indices == legal_indices

    for index in mask_indices:
        assert index_to_move(board, index) in legal_moves


def test_initial_position_round_trip_and_mask() -> None:
    board = chess.Board()

    assert board.legal_moves.count() == 20
    assert_legal_moves_round_trip(board)
    assert_mask_matches_legal_moves(board)


def test_generated_positions_round_trip_and_mask() -> None:
    rng = random.Random(20260510)
    board = chess.Board()

    for _ in range(40):
        assert_legal_moves_round_trip(board)
        assert_mask_matches_legal_moves(board)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        board.push(rng.choice(legal_moves))


def test_castling_round_trip() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    for uci in ("e1g1", "e1c1"):
        move = chess.Move.from_uci(uci)
        index = move_to_index(board, move)
        assert index_to_move(board, index) == move

    assert_mask_matches_legal_moves(board)


def test_queen_promotion_round_trip() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move.from_uci("a7a8q")

    index = move_to_index(board, move)

    assert index_to_move(board, index) == move
    assert legal_policy_mask(board)[index] == 1.0


def test_underpromotion_round_trip() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")

    for uci in ("a7a8n", "a7a8b", "a7a8r"):
        move = chess.Move.from_uci(uci)
        index = move_to_index(board, move)
        assert index_to_move(board, index) == move


def test_capture_underpromotion_round_trip() -> None:
    board = chess.Board("r7/1P6/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move.from_uci("b7a8n")

    index = move_to_index(board, move)

    assert index_to_move(board, index) == move
    assert legal_policy_mask(board)[index] == 1.0


def test_black_underpromotion_round_trip() -> None:
    board = chess.Board("k6K/8/8/8/8/8/p7/8 b - - 0 1")
    move = chess.Move.from_uci("a2a1n")

    index = move_to_index(board, move)

    assert index_to_move(board, index) == move
    assert legal_policy_mask(board)[index] == 1.0


def test_en_passant_round_trip() -> None:
    board = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push(chess.Move.from_uci(uci))

    move = chess.Move.from_uci("e5d6")

    assert board.is_en_passant(move)
    index = move_to_index(board, move)
    assert index_to_move(board, index) == move
    assert legal_policy_mask(board)[index] == 1.0


def test_terminal_positions_have_empty_legal_mask() -> None:
    checkmate = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        checkmate.push(chess.Move.from_uci(uci))

    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

    assert checkmate.is_checkmate()
    assert stalemate.is_stalemate()
    assert int(legal_policy_mask(checkmate).sum()) == 0
    assert int(legal_policy_mask(stalemate).sum()) == 0


def test_invalid_index_returns_none() -> None:
    board = chess.Board()

    assert index_to_move(board, -1) is None
    assert index_to_move(board, POLICY_SIZE) is None
    assert index_to_move(board, chess.A3 * 73) is None


def test_illegal_move_to_index_raises() -> None:
    board = chess.Board()

    with pytest.raises(ValueError):
        move_to_index(board, chess.Move.from_uci("a3a4"))
