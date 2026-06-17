from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chess
import pytest
import torch

from mcchess.board import BOARD_PLANE_COUNT, POLICY_SIZE, move_to_index
from mcchess.bots import MaterialBot, NegamaxBot, NoLegalMoveError, PolicyOnlyBot, RandomLegalBot
from mcchess.bots.notebook import (
    BOARD_SQUARE_SIZE,
    LAST_MOVE_DARK_COLOR,
    ClickableChessBoard,
    NotebookChessGame,
)
from mcchess.model import (
    CheckpointMetadata,
    LoadedPolicyValueCheckpoint,
    PolicyValueResNet,
    ResNetConfig,
    find_best_policy_value_checkpoint,
    load_policy_value_checkpoint,
)


def write_checkpoint(
    path: Path,
    config: ResNetConfig,
    *,
    metrics: dict[str, float] | None = None,
    completed_at: str | None = "2026-06-06T00:01:00+00:00",
) -> None:
    model = PolicyValueResNet(config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.__dict__,
            "train_config": {"seed": 11},
            "epoch": 3,
            "metrics": {"val_total_loss": 1.25} if metrics is None else metrics,
            "saved_at": "2026-06-06T00:00:00+00:00",
            "completed_at": completed_at,
        },
        path,
    )


def test_load_policy_value_checkpoint_restores_model(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    config = ResNetConfig(channels=4, num_blocks=1, value_hidden_dim=8)
    write_checkpoint(checkpoint_path, config)

    loaded = load_policy_value_checkpoint(checkpoint_path, device="cpu")

    assert loaded.metadata.epoch == 3
    assert loaded.metadata.metrics["val_total_loss"] == 1.25
    assert loaded.model_config.channels == 4
    policy_logits, value = loaded.model(torch.zeros(2, BOARD_PLANE_COUNT, 8, 8))
    assert policy_logits.shape == (2, POLICY_SIZE)
    assert value.shape == (2,)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()


def test_find_best_policy_value_checkpoint_uses_lowest_validation_loss(tmp_path: Path) -> None:
    config = ResNetConfig(channels=4, num_blocks=1, value_hidden_dim=8)
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    write_checkpoint(run_a / "checkpoint_latest.pt", config, metrics={"val_total_loss": 1.5})
    write_checkpoint(run_b / "checkpoint_latest.pt", config, metrics={"val_total_loss": 0.75})

    selected = find_best_policy_value_checkpoint(tmp_path)

    assert selected == run_b / "checkpoint_latest.pt"


def test_find_best_policy_value_checkpoint_falls_back_to_newest_completed(
    tmp_path: Path,
) -> None:
    config = ResNetConfig(channels=4, num_blocks=1, value_hidden_dim=8)
    older_run = tmp_path / "older"
    newer_run = tmp_path / "newer"
    older_run.mkdir()
    newer_run.mkdir()
    write_checkpoint(
        older_run / "checkpoint.pt",
        config,
        metrics={},
        completed_at="2026-06-01T00:00:00+00:00",
    )
    write_checkpoint(
        newer_run / "checkpoint.pt",
        config,
        metrics={},
        completed_at="2026-06-02T00:00:00+00:00",
    )

    selected = find_best_policy_value_checkpoint(tmp_path)

    assert selected == newer_run / "checkpoint.pt"


def test_random_legal_bot_is_seeded_and_legal() -> None:
    board = chess.Board()

    move_a = RandomLegalBot(seed=123).choose_move(board)
    move_b = RandomLegalBot(seed=123).choose_move(board)

    assert move_a == move_b
    assert move_a in board.legal_moves


def test_material_bot_prefers_available_queen_capture() -> None:
    board = chess.Board("4k3/8/8/8/8/8/q7/R3K3 w Q - 0 1")

    move = MaterialBot().choose_move(board)

    captured = board.piece_at(move.to_square)
    assert captured is not None
    assert captured.piece_type == chess.QUEEN
    assert move in board.legal_moves


def test_negamax_bot_finds_mate_in_one() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")

    move = NegamaxBot(depth=2).choose_move(board)

    board.push(move)
    assert board.is_checkmate()


def test_negamax_bot_avoids_capture_that_material_bot_takes() -> None:
    board = chess.Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")
    losing_capture = chess.Move.from_uci("d1d5")

    assert MaterialBot().choose_move(board) == losing_capture
    assert NegamaxBot(depth=2).choose_move(board) != losing_capture


def test_negamax_bot_is_deterministic() -> None:
    board = chess.Board()

    move_a = NegamaxBot(depth=2).choose_move(board)
    move_b = NegamaxBot(depth=2).choose_move(board)

    assert move_a == move_b
    assert move_a in board.legal_moves


def test_negamax_bot_plays_legal_moves_through_a_game() -> None:
    board = chess.Board()
    bot = NegamaxBot(depth=2)

    for _ in range(12):
        if board.is_game_over():
            break
        move = bot.choose_move(board)
        assert move in board.legal_moves
        board.push(move)


def test_negamax_bot_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError):
        NegamaxBot(depth=0)


class FakePolicyModel:
    def __init__(self, legal_index: int) -> None:
        self.legal_index = legal_index

    def __call__(self, board_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros((board_tensor.shape[0], POLICY_SIZE), device=board_tensor.device)
        logits[:, 0] = 100.0
        logits[:, self.legal_index] = 10.0
        value = torch.zeros((board_tensor.shape[0],), device=board_tensor.device)
        return logits, value


def fake_loaded_checkpoint(model: object) -> LoadedPolicyValueCheckpoint:
    return LoadedPolicyValueCheckpoint(
        model=cast(Any, model),
        model_config=ResNetConfig(channels=4, num_blocks=1, value_hidden_dim=8),
        metadata=CheckpointMetadata(
            path=Path("fake.pt"),
            epoch=1,
            saved_at=None,
            completed_at=None,
            metrics={},
            train_config={},
        ),
        device=torch.device("cpu"),
    )


def test_policy_only_bot_masks_illegal_logits() -> None:
    board = chess.Board()
    expected_move = chess.Move.from_uci("e2e4")
    expected_index = move_to_index(board, expected_move)
    bot = PolicyOnlyBot(fake_loaded_checkpoint(FakePolicyModel(expected_index)))

    move = bot.choose_move(board)

    assert move == expected_move


def test_policy_only_bot_rejects_terminal_position() -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))

    bot = PolicyOnlyBot(fake_loaded_checkpoint(FakePolicyModel(0)))

    with pytest.raises(NoLegalMoveError):
        bot.choose_move(board)


class ScriptedBot:
    name = "scripted"

    def __init__(self, moves: list[str]) -> None:
        self.moves = [chess.Move.from_uci(move) for move in moves]

    def choose_move(self, board: chess.Board) -> chess.Move:
        move = self.moves.pop(0)
        assert move in board.legal_moves
        return move


class RecordingClickableChessBoard(ClickableChessBoard):
    def __init__(self, game: NotebookChessGame) -> None:
        self.refresh_moves: list[list[str]] = []
        super().__init__(game)

    def _refresh(self) -> None:
        self.refresh_moves.append([move.uci() for move in self.game.board.move_stack])
        super()._refresh()


def test_notebook_game_applies_human_move_and_bot_reply() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)

    game.play("e4")

    assert [move.uci() for move in game.board.move_stack] == ["e2e4", "e7e5"]
    assert game.board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert game.board.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.BLACK)


def test_notebook_game_accepts_san_and_uci() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5", "b8c6"]), human_color=chess.WHITE)

    game.play("e2e4")
    game.play("Nf3")

    assert [move.uci() for move in game.board.move_stack] == ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_notebook_game_illegal_move_does_not_mutate_board() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)
    initial_fen = game.board.fen()

    game.play("Ke2")
    game.play("not a move")

    assert game.board.fen() == initial_fen


def test_notebook_game_bot_moves_first_when_human_plays_black() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e2e4"]), human_color=chess.BLACK)

    assert [move.uci() for move in game.board.move_stack] == ["e2e4"]


def test_notebook_game_renders_board_as_svg() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)

    game.play("e4")

    assert game._repr_svg_().startswith("<svg")


def test_clickable_board_applies_clicks_and_bot_reply() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)
    ui = ClickableChessBoard(game)
    grid_children = tuple(ui.widget.children[1].children)
    e2_button = ui._buttons[chess.E2]
    assert e2_button.layout.width == BOARD_SQUARE_SIZE
    assert e2_button.layout.height == BOARD_SQUARE_SIZE

    ui.click_square("e2")
    ui.click_square("e4")

    assert [move.uci() for move in game.board.move_stack] == ["e2e4", "e7e5"]
    assert tuple(ui.widget.children[1].children) == grid_children
    assert ui._buttons[chess.E2] is e2_button
    # Vacated squares must not be ""; the widget frontend skips repainting
    # buttons with an empty description, leaving the old piece glyph visible.
    assert ui._buttons[chess.E2].description == " "
    assert ui._buttons[chess.E7].description == " "
    assert ui._buttons[chess.E2].icon == ""
    assert ui._buttons[chess.E7].icon == ""
    assert ui._buttons[chess.E7].style.button_color == LAST_MOVE_DARK_COLOR
    assert ui._buttons[chess.E5].style.button_color == LAST_MOVE_DARK_COLOR
    assert "mcchess-white-piece" in ui._buttons[chess.E4]._dom_classes
    assert "mcchess-black-piece" in ui._buttons[chess.E5]._dom_classes
    assert ui._buttons[chess.E4].description == chess.Piece(chess.PAWN, chess.WHITE).unicode_symbol()
    assert ui._buttons[chess.E5].description == chess.Piece(chess.PAWN, chess.BLACK).unicode_symbol()


def test_clickable_board_refreshes_human_move_before_bot_reply() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)
    ui = RecordingClickableChessBoard(game)

    ui.click_square("e2")
    ui.click_square("e4")

    human_move_index = ui.refresh_moves.index(["e2e4"])
    bot_reply_index = ui.refresh_moves.index(["e2e4", "e7e5"])
    assert human_move_index < bot_reply_index


def test_clickable_board_illegal_target_does_not_mutate_board() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)
    ui = ClickableChessBoard(game)
    initial_fen = game.board.fen()

    ui.click_square("e3")
    assert game.board.fen() == initial_fen

    ui.click_square("e2")
    ui.click_square("e5")
    assert game.board.fen() == initial_fen


def test_clickable_board_reselects_when_clicking_own_piece() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)
    ui = ClickableChessBoard(game)

    ui.click_square("e2")
    ui.click_square("d2")
    ui.click_square("d4")

    assert [move.uci() for move in game.board.move_stack] == ["d2d4", "e7e5"]
