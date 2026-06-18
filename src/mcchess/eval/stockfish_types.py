"""Typed protocol objects for external Stockfish benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

import chess

from mcchess.eval.arena import BotConfig
from mcchess.eval.openings import normalize_opening_fens
from mcchess.eval.stockfish_utils import EngineOptionValue
from mcchess.eval.stockfish_utils import engine_limit as stockfish_engine_limit

DRAW_RULE: Final[str] = "python_chess_outcome_or_max_ply_draw"
COLOR_POLICY: Final[str] = "alternating_mcchess_white_first_global_schedule"
SCOPE_NOTE: Final[str] = (
    "External Stockfish benchmark only. Do not use Stockfish moves, evaluations, "
    "or game outcomes for McChess training data, labels, distillation targets, or "
    "checkpoint selection targets."
)
MoveCallback = Callable[[dict[str, Any]], None]
GameCallback = Callable[["StockfishGameRecord"], None]


@dataclass(frozen=True)
class StockfishLevelConfig:
    """One Stockfish strength level in an external benchmark schedule."""

    name: str
    games: int
    options: dict[str, EngineOptionValue] = field(default_factory=dict)
    limit: dict[str, float | int] = field(default_factory=dict)
    include_in_elo: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Stockfish level name must not be empty")
        if self.games <= 0:
            raise ValueError("Stockfish level games must be positive")
        stockfish_engine_limit(self.limit)
        if self.include_in_elo and self.stockfish_elo is None:
            raise ValueError(
                "Stockfish levels included in Elo estimation require "
                "UCI_LimitStrength=true and UCI_Elo"
            )

    @property
    def stockfish_elo(self) -> int | None:
        """Return the configured UCI Elo when this level is an Elo-handicap level."""

        if self.options.get("UCI_LimitStrength") is not True:
            return None
        raw_elo = self.options.get("UCI_Elo")
        if isinstance(raw_elo, bool) or raw_elo is None:
            return None
        if isinstance(raw_elo, int):
            elo = raw_elo
        elif isinstance(raw_elo, str):
            try:
                elo = int(raw_elo)
            except ValueError:
                return None
        else:
            return None
        return elo if elo > 0 else None


@dataclass(frozen=True)
class StockfishEvalConfig:
    """Configuration for a fixed external Stockfish benchmark."""

    run_id: str
    output_dir: str
    agent: BotConfig
    stockfish_levels: list[StockfishLevelConfig]
    stockfish_path: str | None = None
    seed: int = 0
    max_ply: int = 180
    move_delay_seconds: float = 0.0
    print_moves: bool = False
    opening_fens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        object.__setattr__(self, "opening_fens", normalize_opening_fens(self.opening_fens))
        if self.agent.kind != "mcts":
            raise ValueError("Stockfish benchmark agent must be an MCTS bot")
        if self.agent.simulations != 200:
            raise ValueError("Stockfish benchmark is intentionally fixed to MCTS-200")
        if not self.stockfish_levels:
            raise ValueError("stockfish_levels must contain at least one level")
        if self.max_ply <= 0:
            raise ValueError("max_ply must be positive")
        if self.move_delay_seconds < 0.0:
            raise ValueError("move_delay_seconds must be non-negative")

    @property
    def num_games(self) -> int:
        """Total scheduled games."""

        return sum(level.games for level in self.stockfish_levels)


@dataclass(frozen=True)
class ScheduledStockfishGame:
    """Expanded schedule entry for one game."""

    game_index: int
    level_game_index: int
    level: StockfishLevelConfig
    mcchess_color: chess.Color
    opening_index: int | None = None
    starting_fen: str = chess.STARTING_FEN


@dataclass(frozen=True)
class StockfishGameRecord:
    """Single external benchmark game record from McChess's perspective."""

    game_index: int
    level: str
    level_game_index: int
    stockfish_elo: int | None
    include_in_elo: bool
    status: str
    mcchess_color: str
    white: str
    black: str
    result: str
    winner: str | None
    winner_name: str | None
    mcchess_score: float | None
    termination: str
    ply_count: int
    final_fen: str
    moves: list[str]
    stockfish_options: dict[str, object]
    stockfish_limit: dict[str, float | int]
    opening_index: int | None = None
    starting_fen: str = chess.STARTING_FEN
    illegal_move: dict[str, str] | None = None
    error: str | None = None
