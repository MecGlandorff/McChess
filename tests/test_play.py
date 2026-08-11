from __future__ import annotations

from collections.abc import Iterator

import chess
import pytest

from mcchess.play import (
    DEFAULT_MCTS_SIMULATIONS,
    format_board,
    parse_args,
    parse_human_move,
    play_terminal_game,
)


class ScriptedBot:
    name = "scripted"

    def __init__(self, moves: list[str]) -> None:
        self._moves = [chess.Move.from_uci(move) for move in moves]

    def choose_move(self, board: chess.Board) -> chess.Move:
        move = self._moves.pop(0)
        assert move in board.legal_moves
        return move


def _input_from(moves: Iterator[str]):
    def read_move(_prompt: str) -> str:
        return next(moves)

    return read_move


def test_parse_human_move_accepts_san_and_uci() -> None:
    board = chess.Board()

    assert parse_human_move(board, "e4") == chess.Move.from_uci("e2e4")
    assert parse_human_move(board, "E2E4") == chess.Move.from_uci("e2e4")


def test_parse_human_move_rejects_illegal_or_empty_input() -> None:
    board = chess.Board()

    with pytest.raises(ValueError, match="must not be empty"):
        parse_human_move(board, "  ")
    with pytest.raises(ValueError, match="not a legal SAN or UCI move"):
        parse_human_move(board, "e5")


def test_format_board_uses_human_orientation() -> None:
    board = chess.Board()

    white_view = format_board(board, orientation=chess.WHITE).splitlines()
    black_view = format_board(board, orientation=chess.BLACK).splitlines()

    assert white_view[0] == "8  r n b q k b n r"
    assert white_view[-1] == "   a b c d e f g h"
    assert black_view[0] == "1  R N B K Q B N R"
    assert black_view[-1] == "   h g f e d c b a"


def test_terminal_game_accepts_moves_until_checkmate() -> None:
    output: list[str] = []
    bot = ScriptedBot(["e7e5", "d8h4"])

    outcome = play_terminal_game(
        bot,
        input_fn=_input_from(iter(["f3", "g2g4"])),
        output_fn=output.append,
    )

    assert outcome is not None
    assert outcome.winner == chess.BLACK
    assert output[-1] == "Game over: 0-1 (checkmate)."


def test_terminal_game_stops_cleanly_after_bot_opens_for_black() -> None:
    output: list[str] = []

    outcome = play_terminal_game(
        ScriptedBot(["e2e4"]),
        human_color=chess.BLACK,
        input_fn=_input_from(iter(["quit"])),
        output_fn=output.append,
    )

    assert outcome is None
    assert "scripted played e4 (e2e4)." in output
    assert output[-1] == "Game stopped."


def test_terminal_defaults_to_mcts_800() -> None:
    args = parse_args([])

    assert args.mode == "mcts"
    assert args.simulations == DEFAULT_MCTS_SIMULATIONS == 800
