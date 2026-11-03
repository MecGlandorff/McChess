from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

from mcchess.board import POLICY_SIZE, move_to_index
from mcchess.bots import MCTSBot, NoLegalMoveError
from mcchess.model import CheckpointMetadata, LoadedPolicyValueCheckpoint, ResNetConfig
from mcchess.search import MCTSConfig, MCTSSearch, SearchEdge, SearchNode, backup_value


class FixedPolicyValueModel:
    def __init__(self, logits: torch.Tensor | None = None, value: float = 0.0) -> None:
        self.logits = logits if logits is not None else torch.zeros(POLICY_SIZE)
        self.value = value
        self.calls = 0
        self.inference_mode_calls: list[bool] = []

    def __call__(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        self.inference_mode_calls.append(torch.is_inference_mode_enabled())
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
    logits = preferred_logits(board, "e2e4")
    logits[chess.A1 * 73] = 100.0  # An illegal action must not enter the legal softmax.
    model = FixedPolicyValueModel(logits)
    search = MCTSSearch(model, torch.device("cpu"), MCTSConfig(simulations=1))

    result = search.search(board)
    legal_uci = {move.uci() for move in board.legal_moves}
    stats_by_move = {stat.move_uci: stat for stat in result.root_stats}

    assert set(stats_by_move) == legal_uci
    assert result.move == chess.Move.from_uci("e2e4")
    assert stats_by_move["e2e4"].prior == max(stat.prior for stat in result.root_stats)
    assert sum(stat.prior for stat in result.root_stats) == pytest.approx(1.0)


def test_mcts_model_calls_use_inference_mode() -> None:
    board = chess.Board()
    model = FixedPolicyValueModel(preferred_logits(board, "e2e4"))

    MCTSSearch(model, torch.device("cpu"), MCTSConfig(simulations=2)).search(board)

    assert model.inference_mode_calls
    assert all(model.inference_mode_calls)


@pytest.mark.parametrize("fpu_reduction", [None, 0.0, 0.2])
def test_mcts_returns_legal_move_deterministically(fpu_reduction: float | None) -> None:
    board = chess.Board()
    model = FixedPolicyValueModel(preferred_logits(board, "d2d4"))
    config = MCTSConfig(simulations=5, c_puct=1.5, fpu_reduction=fpu_reduction)

    first = MCTSSearch(model, torch.device("cpu"), config).search(board)
    second = MCTSSearch(model, torch.device("cpu"), config).search(board)

    assert first.move in board.legal_moves
    assert first == second
    assert sum(stat.visit_count for stat in first.root_stats) == config.simulations


@pytest.mark.parametrize(
    ("reduction", "expected_move"),
    [(None, "e2e4"), (0.0, "d2d4"), (0.2, "d2d4"), (0.6, "e2e4")],
)
def test_fpu_changes_selection_for_unvisited_edges(
    reduction: float | None, expected_move: str,
) -> None:
    node = SearchNode(
        network_value=0.6,
        visit_count=1,
        edges=[
            SearchEdge(chess.Move.from_uci("e2e4"), prior=0.5, visit_count=1, value_sum=0.4),
            SearchEdge(chess.Move.from_uci("d2d4"), prior=0.5),
        ],
    )
    config = MCTSConfig() if reduction is None else MCTSConfig(fpu_reduction=reduction)
    search = MCTSSearch(FixedPolicyValueModel(), torch.device("cpu"), config)

    # With default Q=0, the visited move narrowly wins: 0.775 versus 0.75.
    assert search._select_edge(node).move.uci() == expected_move


def test_fpu_preserves_visited_edge_mean_values() -> None:
    node = SearchNode(
        network_value=0.8,
        visit_count=5,
        edges=[
            SearchEdge(chess.Move.from_uci("e2e4"), prior=0.5, visit_count=1, value_sum=0.1),
            SearchEdge(chess.Move.from_uci("d2d4"), prior=0.5, visit_count=4, value_sum=0.8),
        ],
    )
    search = MCTSSearch(
        FixedPolicyValueModel(), torch.device("cpu"),
        MCTSConfig(c_puct=0.05, fpu_reduction=0.2),
    )

    assert search._select_edge(node) is node.edges[1]


def test_fpu_clamps_unvisited_value_at_minus_one() -> None:
    node = SearchNode(
        network_value=-0.9,
        visit_count=1,
        edges=[
            SearchEdge(chess.Move.from_uci("e2e4"), prior=0.5, visit_count=1, value_sum=-0.7),
            SearchEdge(chess.Move.from_uci("d2d4"), prior=0.5),
        ],
    )
    search = MCTSSearch(
        FixedPolicyValueModel(), torch.device("cpu"), MCTSConfig(fpu_reduction=0.5),
    )

    # Clamping gives the unvisited move -0.25; without it, -0.65 loses to -0.325.
    assert search._select_edge(node) is node.edges[1]


def test_fpu_stores_root_and_leaf_values_without_flipping_perspective() -> None:
    board = chess.Board()
    root = SearchNode()
    search = MCTSSearch(
        FixedPolicyValueModel(value=-0.4), torch.device("cpu"),
        MCTSConfig(simulations=1, fpu_reduction=0.2),
    )

    search._evaluate_node(root, board)
    pending = search._collect_pending_leaves(root, board, batch_size=1)
    search._evaluate_pending_leaves(pending)

    assert root.network_value == pytest.approx(-0.4)
    assert pending[0].node.network_value == pytest.approx(-0.4)
    assert pending[0].path[0].mean_value == pytest.approx(0.4)


def test_batched_mcts_uses_fewer_model_calls_and_preserves_visit_budget() -> None:
    board = chess.Board()
    model = FixedPolicyValueModel(preferred_logits(board, "e2e4"))
    config = MCTSConfig(simulations=8, c_puct=1.5, inference_batch_size=4)

    result = MCTSSearch(model, torch.device("cpu"), config).search(board)

    assert result.move in board.legal_moves
    assert sum(stat.visit_count for stat in result.root_stats) == config.simulations
    assert model.calls < config.simulations + 1


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


@pytest.mark.parametrize("fpu_reduction", [None, 0.2])
def test_terminal_child_is_not_evaluated_as_ordinary_leaf(fpu_reduction: float | None) -> None:
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4"):
        board.push(chess.Move.from_uci(uci))
    mate = chess.Move.from_uci("d8h4")
    model = FixedPolicyValueModel(preferred_logits(board, mate.uci()))

    config = MCTSConfig(simulations=1, fpu_reduction=fpu_reduction)
    result = MCTSSearch(model, torch.device("cpu"), config).search(board)
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
    with pytest.raises(ValueError, match="inference_batch_size"):
        MCTSConfig(inference_batch_size=0)


@pytest.mark.parametrize("reduction", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_mcts_config_rejects_invalid_fpu_reduction(reduction: float) -> None:
    with pytest.raises(ValueError, match="fpu_reduction"):
        MCTSConfig(fpu_reduction=reduction)


@pytest.mark.parametrize("reduction", [0.0, 0.2])
def test_fpu_requires_single_leaf_inference(reduction: float) -> None:
    with pytest.raises(ValueError, match="inference_batch_size=1"):
        MCTSConfig(fpu_reduction=reduction, inference_batch_size=4)
