"""PUCT Monte Carlo tree search for policy/value checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import chess
import torch

from mcchess.board import POLICY_SIZE, encode_board, legal_policy_mask, move_to_index
from mcchess.bots.base import legal_moves_or_raise


class PolicyValueModel(Protocol):
    """Callable policy/value model used by MCTS."""

    def __call__(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and side-to-move value for a board batch."""
        ...


@dataclass(frozen=True)
class MCTSConfig:
    """Fixed-budget PUCT search configuration."""

    simulations: int = 50
    c_puct: float = 1.5

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if not math.isfinite(self.c_puct) or self.c_puct <= 0.0:
            raise ValueError("c_puct must be a positive finite value")


@dataclass
class SearchEdge:
    """One action edge in the MCTS tree."""

    move: chess.Move
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: SearchNode | None = None

    @property
    def mean_value(self) -> float:
        """Mean value from the parent node side-to-move perspective."""

        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class SearchNode:
    """MCTS node containing legal outgoing edges for one board position."""

    expanded: bool = False
    terminal: bool = False
    edges: list[SearchEdge] = field(default_factory=list)

    @property
    def visit_count(self) -> int:
        """Return the number of simulations that selected an outgoing edge."""

        return sum(edge.visit_count for edge in self.edges)


@dataclass(frozen=True)
class RootMoveStats:
    """Inspectable root statistics for a searched move."""

    move: chess.Move
    move_uci: str
    prior: float
    visit_count: int
    mean_value: float


@dataclass(frozen=True)
class MCTSResult:
    """Result of a fixed-budget search from one root position."""

    move: chess.Move
    simulations: int
    root_stats: tuple[RootMoveStats, ...]


class MCTSSearch:
    """Run deterministic PUCT search using a policy/value neural network."""

    def __init__(
        self,
        model: PolicyValueModel,
        device: torch.device,
        config: MCTSConfig | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config or MCTSConfig()

    def search(self, board: chess.Board) -> MCTSResult:
        """Run MCTS from `board` and return the selected legal move."""

        legal_moves_or_raise(board)
        root = SearchNode()
        self._evaluate_node(root, board)

        for _ in range(self.config.simulations):
            search_board = board.copy(stack=True)
            node = root
            path: list[SearchEdge] = []

            while node.expanded and not node.terminal:
                edge = self._select_edge(node)
                path.append(edge)
                search_board.push(edge.move)
                if edge.child is None:
                    edge.child = SearchNode()
                node = edge.child
                if not node.expanded:
                    break

            leaf_value = self._evaluate_node(node, search_board)
            backup_value(path, leaf_value)

        best_edge = self._best_root_edge(root)
        return MCTSResult(
            move=best_edge.move,
            simulations=self.config.simulations,
            root_stats=_root_stats(root),
        )

    def _evaluate_node(self, node: SearchNode, board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            node.expanded = True
            node.terminal = True
            node.edges.clear()
            return _terminal_value(board, outcome)

        if node.expanded:
            raise RuntimeError("attempted to evaluate an already expanded nonterminal node")

        policy_logits, value = self._model_forward(board)
        node.edges = self._policy_edges(board, policy_logits)
        node.expanded = True
        return _value_scalar(value)

    def _policy_edges(self, board: chess.Board, policy_logits: torch.Tensor) -> list[SearchEdge]:
        legal_moves = sorted(legal_moves_or_raise(board), key=chess.Move.uci)
        mask = torch.from_numpy(legal_policy_mask(board).astype(bool)).to(self.device)
        masked_logits = policy_logits.masked_fill(~mask, -torch.inf)
        priors = torch.softmax(masked_logits, dim=0)

        if not torch.isfinite(priors).all():
            raise ValueError("MCTS policy priors contain non-finite values")

        return [
            SearchEdge(
                move=move,
                prior=float(priors[move_to_index(board, move)].item()),
            )
            for move in legal_moves
        ]

    def _model_forward(self, board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = encode_board(board)
        board_tensor = torch.from_numpy(encoded).unsqueeze(0).to(
            device=self.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)

        if policy_logits.shape != (1, POLICY_SIZE):
            raise ValueError("MCTS model policy logits must have shape [1, 4672]")
        if value.shape != (1,):
            raise ValueError("MCTS model value must have shape [1]")
        if not torch.isfinite(policy_logits).all():
            raise ValueError("MCTS policy logits contain non-finite values")
        if not torch.isfinite(value).all():
            raise ValueError("MCTS value prediction contains non-finite values")
        return policy_logits.squeeze(0), value

    def _select_edge(self, node: SearchNode) -> SearchEdge:
        if not node.edges:
            raise RuntimeError("cannot select from a node with no outgoing edges")

        parent_visits = max(1, node.visit_count)
        exploration_base = math.sqrt(parent_visits)
        best_edge = node.edges[0]
        best_score = -math.inf

        for edge in node.edges:
            score = edge.mean_value + (
                self.config.c_puct * edge.prior * exploration_base / (1 + edge.visit_count)
            )
            if score > best_score:
                best_edge = edge
                best_score = score
        return best_edge

    @staticmethod
    def _best_root_edge(root: SearchNode) -> SearchEdge:
        if not root.edges:
            raise RuntimeError("MCTS root has no legal moves")
        return max(root.edges, key=lambda edge: (edge.visit_count, edge.mean_value, edge.prior))


def backup_value(path: Sequence[SearchEdge], leaf_value: float) -> None:
    """Back up a leaf value, flipping perspective once per ply."""

    value = leaf_value
    for edge in reversed(path):
        value = -value
        edge.visit_count += 1
        edge.value_sum += value


def _terminal_value(board: chess.Board, outcome: chess.Outcome) -> float:
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _value_scalar(value: torch.Tensor) -> float:
    scalar = float(value.item())
    if scalar < -1.0 or scalar > 1.0:
        raise ValueError("MCTS value prediction must be in [-1, 1]")
    return scalar


def _root_stats(root: SearchNode) -> tuple[RootMoveStats, ...]:
    return tuple(
        RootMoveStats(
            move=edge.move,
            move_uci=edge.move.uci(),
            prior=edge.prior,
            visit_count=edge.visit_count,
            mean_value=edge.mean_value,
        )
        for edge in sorted(root.edges, key=lambda edge: edge.move.uci())
    )
