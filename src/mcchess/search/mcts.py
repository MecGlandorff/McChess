"""PUCT Monte Carlo tree search for policy/value checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import chess
import numpy as np
import torch

from mcchess.board import POLICY_SIZE, encode_board, legal_moves_with_policy_indices
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
    inference_batch_size: int = 1

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if not math.isfinite(self.c_puct) or self.c_puct <= 0.0:
            raise ValueError("c_puct must be a positive finite value")
        if self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")


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
    visit_count: int = 0


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


@dataclass(frozen=True)
class _PendingLeaf:
    node: SearchNode
    board: chess.Board
    path: tuple[SearchEdge, ...]


@dataclass
class _PendingLeafGroup:
    node: SearchNode
    board: chess.Board
    paths: list[tuple[SearchEdge, ...]] = field(default_factory=list)


@dataclass(frozen=True)
class _LegalExpansion:
    moves: tuple[chess.Move, ...]
    policy_indices: tuple[int, ...]


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

        remaining = self.config.simulations
        while remaining > 0:
            batch_size = min(self.config.inference_batch_size, remaining)
            pending = self._collect_pending_leaves(root, board, batch_size)
            self._evaluate_pending_leaves(pending)
            remaining -= len(pending)

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

        edge_batches, values = self._evaluate_nonterminal_boards([board])
        node.edges = edge_batches[0]
        node.expanded = True
        return float(values[0])

    def _evaluate_nonterminal_boards(
        self,
        boards: Sequence[chess.Board],
    ) -> tuple[list[list[SearchEdge]], np.ndarray]:
        expansions = [_legal_expansion(board) for board in boards]
        policy_logits, values = self._model_forward_batch(boards)
        prior_batches, host_values = self._legal_priors_and_values(
            policy_logits,
            values,
            expansions,
        )
        edge_batches = [
            _policy_edges_from_priors(expansion, priors)
            for expansion, priors in zip(expansions, prior_batches, strict=True)
        ]
        return edge_batches, host_values

    def _model_forward_batch(
        self,
        boards: Sequence[chess.Board],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not boards:
            raise ValueError("MCTS model batch must contain at least one board")

        encoded = np.stack([encode_board(board) for board in boards])
        board_tensor = torch.from_numpy(encoded).to(
            device=self.device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            policy_logits, value = self.model(board_tensor)

        expected_policy_shape = (len(boards), POLICY_SIZE)
        expected_value_shape = (len(boards),)
        if policy_logits.shape != expected_policy_shape:
            raise ValueError("MCTS model policy logits must have shape [batch, 4672]")
        if value.shape != expected_value_shape:
            raise ValueError("MCTS model value must have shape [batch]")
        if not torch.isfinite(policy_logits).all():
            raise ValueError("MCTS policy logits contain non-finite values")
        if not torch.isfinite(value).all():
            raise ValueError("MCTS value prediction contains non-finite values")
        return policy_logits, value

    def _legal_priors_and_values(
        self,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        expansions: Sequence[_LegalExpansion],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        max_legal_moves = max(len(expansion.moves) for expansion in expansions)
        policy_indices = np.zeros((len(expansions), max_legal_moves), dtype=np.int64)
        legal_slots = np.zeros((len(expansions), max_legal_moves), dtype=np.bool_)
        for batch_index, expansion in enumerate(expansions):
            count = len(expansion.moves)
            policy_indices[batch_index, :count] = expansion.policy_indices
            legal_slots[batch_index, :count] = True

        index_tensor = torch.from_numpy(policy_indices).to(self.device)
        legal_slot_tensor = torch.from_numpy(legal_slots).to(self.device)
        legal_logits = policy_logits.gather(dim=1, index=index_tensor)
        legal_logits = legal_logits.masked_fill(~legal_slot_tensor, -torch.inf)
        legal_priors = torch.softmax(legal_logits, dim=1)
        legal_priors = legal_priors.masked_fill(~legal_slot_tensor, 0.0)
        if not torch.isfinite(legal_priors).all():
            raise ValueError("MCTS policy priors contain non-finite values")

        # One device-to-host transfer replaces a synchronized `.item()` call per edge.
        host_output = (
            torch.cat((values.unsqueeze(1), legal_priors), dim=1)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        host_values = host_output[:, 0]
        if np.any((host_values < -1.0) | (host_values > 1.0)):
            raise ValueError("MCTS value prediction must be in [-1, 1]")
        prior_batches = [
            host_output[index, 1 : 1 + len(expansion.moves)]
            for index, expansion in enumerate(expansions)
        ]
        return prior_batches, host_values

    def _collect_pending_leaves(
        self,
        root: SearchNode,
        board: chess.Board,
        batch_size: int,
    ) -> list[_PendingLeaf]:
        pending: list[_PendingLeaf] = []
        for _ in range(batch_size):
            search_board = board.copy(stack=True)
            node = root
            path: list[SearchEdge] = []
            parent_nodes: list[SearchNode] = []

            while node.expanded and not node.terminal:
                parent_nodes.append(node)
                edge = self._select_edge(node)
                path.append(edge)
                search_board.push(edge.move)
                if edge.child is None:
                    edge.child = SearchNode()
                node = edge.child
                if not node.expanded:
                    break

            reserved_path = tuple(path)
            _reserve_path_visit(reserved_path, parent_nodes)
            pending.append(
                _PendingLeaf(
                    node=node,
                    board=search_board,
                    path=reserved_path,
                )
            )
        return pending

    def _evaluate_pending_leaves(self, pending: Sequence[_PendingLeaf]) -> None:
        groups_by_node: dict[int, _PendingLeafGroup] = {}
        for leaf in pending:
            outcome = leaf.board.outcome(claim_draw=True)
            if outcome is not None:
                leaf.node.expanded = True
                leaf.node.terminal = True
                leaf.node.edges.clear()
                backup_reserved_value(leaf.path, _terminal_value(leaf.board, outcome))
                continue

            if leaf.node.expanded:
                raise RuntimeError("attempted to evaluate an already expanded nonterminal node")

            key = id(leaf.node)
            group = groups_by_node.get(key)
            if group is None:
                group = _PendingLeafGroup(node=leaf.node, board=leaf.board)
                groups_by_node[key] = group
            group.paths.append(leaf.path)

        if not groups_by_node:
            return

        groups = list(groups_by_node.values())
        boards = [group.board for group in groups]
        edge_batches, values = self._evaluate_nonterminal_boards(boards)

        for index, group in enumerate(groups):
            group.node.edges = edge_batches[index]
            group.node.expanded = True
            leaf_value = float(values[index])
            for path in group.paths:
                backup_reserved_value(path, leaf_value)

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


def backup_reserved_value(path: Sequence[SearchEdge], leaf_value: float) -> None:
    """Back up a leaf value onto edges whose visits were already reserved."""

    value = leaf_value
    for edge in reversed(path):
        value = -value
        edge.value_sum += value


def _reserve_path_visit(
    path: Sequence[SearchEdge],
    parent_nodes: Sequence[SearchNode],
) -> None:
    if len(path) != len(parent_nodes):
        raise ValueError("MCTS reserved path nodes and edges must have matching lengths")
    for node, edge in zip(parent_nodes, path, strict=True):
        node.visit_count += 1
        edge.visit_count += 1


def _terminal_value(board: chess.Board, outcome: chess.Outcome) -> float:
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


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


def _legal_expansion(board: chess.Board) -> _LegalExpansion:
    indexed_moves = sorted(
        legal_moves_with_policy_indices(board),
        key=lambda indexed_move: indexed_move[0].uci(),
    )
    if not indexed_moves:
        raise RuntimeError("cannot expand a node with no legal moves")
    return _LegalExpansion(
        moves=tuple(move for move, _ in indexed_moves),
        policy_indices=tuple(policy_index for _, policy_index in indexed_moves),
    )


def _policy_edges_from_priors(
    expansion: _LegalExpansion,
    priors: np.ndarray,
) -> list[SearchEdge]:
    if priors.shape != (len(expansion.moves),):
        raise ValueError("MCTS legal policy priors have the wrong shape")

    return [
        SearchEdge(
            move=move,
            prior=float(prior),
        )
        for move, prior in zip(expansion.moves, priors, strict=True)
    ]
