from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chess
import pytest
import torch

from mcchess.board import BOARD_PLANE_COUNT, POLICY_SIZE, move_to_index
from mcchess.bots import MaterialBot, NoLegalMoveError, PolicyOnlyBot, RandomLegalBot
from mcchess.bots.notebook import NotebookChessGame, create_notebook_game
from mcchess.model import (
    CheckpointMetadata,
    LoadedPolicyValueCheckpoint,
    PolicyValueResNet,
    ResNetConfig,
    load_policy_value_checkpoint,
)


def write_checkpoint(path: Path, config: ResNetConfig) -> None:
    model = PolicyValueResNet(config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.__dict__,
            "train_config": {"seed": 11},
            "epoch": 3,
            "metrics": {"val_total_loss": 1.25},
            "saved_at": "2026-06-06T00:00:00+00:00",
            "completed_at": "2026-06-06T00:01:00+00:00",
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


def test_notebook_game_accepts_click_move_and_bot_reply() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)

    game.click_square("e2")
    game.click_square("e4")

    assert [move.uci() for move in game.board.move_stack] == ["e2e4", "e7e5"]


def test_notebook_game_illegal_click_does_not_mutate_board() -> None:
    game = NotebookChessGame(bot=ScriptedBot(["e7e5"]), human_color=chess.WHITE)
    initial_fen = game.board.fen()

    game.click_square("e3")
    assert game.board.fen() == initial_fen

    game.click_square("e2")
    game.click_square("e5")
    assert game.board.fen() == initial_fen


def test_create_notebook_game_returns_widget() -> None:
    widget = create_notebook_game(ScriptedBot(["e7e5"]), human_color=chess.WHITE)

    assert hasattr(widget, "children")
