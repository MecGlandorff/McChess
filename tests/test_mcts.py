from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

from mcchess.board import POLICY_SIZE, move_to_index
from mcchess.bots import MCTSBot, NoLegalMoveError
from mcchess.model import CheckpointMetadata, LoadedPolicyValueCheckpoint, ResNetConfig
from mcchess.search import MCTSConfig, MCTSSearch, SearchEdge, backup_value


class FixedPolicyValueModel:
    def __init__(self, logits: torch.Tensor | None = None, value: float = 0.0) -> None:
        self.logits = logits if logits is not None else torch.zeros(POLICY_SIZE)
        self.value = value
        self.calls = 0

    def __call__(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        batch_size = board.shape[0]
        logits = self.logits.to(board.device).repeat(batch_size, 1)
        value = torch.full((batch_size,), self.value, dtype=torch.float32, device=board.device)
        return logits, value


def preferred_logits(board: chess.Board, move_uci: str, value: float = 6.0) -> torch.Tensor:
    logits = torch.zeros(POLICY_SIZE)
    move = chess.Move.from_uci(move_uci)
    logits[move_to_index(board, move)] = value
    return logits


def test_backup_flips_value_sign_each_ply() -> None:
    first = SearchEdge(move=chess.Move.from_uci("e2e4"), prior=0.5)
    second = SearchEdge(move=chess.Move.from_uci("e7e5"), prior=0.5)

    backup_value([first, second], leaf_value=0.75)

    assert first.visit_count == 1
    assert second.visit_count == 1
    assert first.mean_value == pytest.approx(0.75)
    assert second.mean_value == pytest.approx(-0.75)


def test_mcts_expands_only_legal_root_moves_and_masks_priors() -> None:
    board = chess.Board()
    model = FixedPolicyValueModel(preferred_logits(board, "e2e4"))
    search = MCTSSearch(model, torch.device("cpu"), MCTSConfig(simulations=1))

    result = search.search(board)
    legal_uci = {move.uci() for move in board.legal_moves}
    stats_by_move = {stat.move_uci: stat for stat in result.root_stats}

    assert set(stats_by_move) == legal_uci
    assert result.move == chess.Move.from_uci("e2e4")
    assert stats_by_move["e2e4"].prior == max(stat.prior for stat in result.root_stats)


def test_mcts_returns_legal_move_deterministically() -> None:
    board = chess.Board()
    model = FixedPolicyValueModel(preferred_logits(board, "d2d4"))
    config = MCTSConfig(simulations=5, c_puct=1.5)

    first = MCTSSearch(model, torch.device("cpu"), config).search(board)
    second = MCTSSearch(model, torch.device("cpu"), config).search(board)

    assert first.move in board.legal_moves
    assert first.move == second.move


def test_mcts_bot_returns_legal_move_from_checkpoint() -> None:
    board = chess.Board()
    model = FixedPolicyValueModel(preferred_logits(board, "g1f3"))
    checkpoint = LoadedPolicyValueCheckpoint(
        model=model,
        model_config=ResNetConfig(),
        metadata=CheckpointMetadata(
            path=Path("checkpoint.pt"),
            epoch=None,
            saved_at=None,
            completed_at=None,
            metrics={},
            train_config={},
        ),
        device=torch.device("cpu"),
    )
    bot = MCTSBot(checkpoint=checkpoint, config=MCTSConfig(simulations=1), name="mcts")

    move = bot.choose_move(board)

    assert move in board.legal_moves


def test_terminal_child_is_not_evaluated_as_ordinary_leaf() -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4"):
        board.push(chess.Move.from_uci(uci))
    mate = chess.Move.from_uci("d8h4")
    model = FixedPolicyValueModel(preferred_logits(board, mate.uci()))

    result = MCTSSearch(model, torch.device("cpu"), MCTSConfig(simulations=1)).search(board)
    mate_stats = next(stat for stat in result.root_stats if stat.move == mate)

    assert result.move == mate
    assert mate_stats.visit_count == 1
    assert mate_stats.mean_value == pytest.approx(1.0)
    assert model.calls == 1


def test_mcts_rejects_terminal_root_without_model_eval() -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))
    model = FixedPolicyValueModel()

    with pytest.raises(NoLegalMoveError):
        MCTSSearch(model, torch.device("cpu"), MCTSConfig(simulations=1)).search(board)

    assert model.calls == 0


def test_mcts_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="simulations"):
        MCTSConfig(simulations=0)
    with pytest.raises(ValueError, match="c_puct"):
        MCTSConfig(c_puct=0.0)
