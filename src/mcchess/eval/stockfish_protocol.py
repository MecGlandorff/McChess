"""Compatibility imports for Stockfish benchmark protocol helpers."""

from __future__ import annotations

from mcchess.eval.stockfish_config import (
    iter_scheduled_games,
    level_config_dict,
    load_config,
    protocol_summary,
)
from mcchess.eval.stockfish_game import play_stockfish_game
from mcchess.eval.stockfish_types import (
    COLOR_POLICY,
    DRAW_RULE,
    SCOPE_NOTE,
    GameCallback,
    MoveCallback,
    ScheduledStockfishGame,
    StockfishEvalConfig,
    StockfishGameRecord,
    StockfishLevelConfig,
)

__all__ = [
    "COLOR_POLICY",
    "DRAW_RULE",
    "SCOPE_NOTE",
    "GameCallback",
    "MoveCallback",
    "ScheduledStockfishGame",
    "StockfishEvalConfig",
    "StockfishGameRecord",
    "StockfishLevelConfig",
    "iter_scheduled_games",
    "level_config_dict",
    "load_config",
    "play_stockfish_game",
    "protocol_summary",
]
