"""Config loading and schedule expansion for Stockfish benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chess

from mcchess.eval.arena import BotConfig
from mcchess.eval.common import load_yaml_mapping
from mcchess.eval.openings import opening_protocol, starting_fen_for_game
from mcchess.eval.stockfish_types import (
    COLOR_POLICY,
    DRAW_RULE,
    SCOPE_NOTE,
    ScheduledStockfishGame,
    StockfishEvalConfig,
    StockfishLevelConfig,
)


def load_config(path: str | Path) -> StockfishEvalConfig:
    """Load a Stockfish benchmark YAML config."""

    data = load_yaml_mapping(path)
    agent_raw = data.pop("agent", None)
    if not isinstance(agent_raw, dict):
        raise ValueError("agent must be a YAML mapping")

    levels_raw = data.pop("stockfish_levels", None)
    if not isinstance(levels_raw, list):
        raise ValueError("stockfish_levels must be a YAML list")
    levels = [_load_level(level_raw) for level_raw in levels_raw]

    return StockfishEvalConfig(
        **data,
        agent=BotConfig(**agent_raw),
        stockfish_levels=levels,
    )


def iter_scheduled_games(config: StockfishEvalConfig) -> list[ScheduledStockfishGame]:
    """Expand a level schedule into individual games with alternating colors."""

    scheduled: list[ScheduledStockfishGame] = []
    game_index = 0
    for level in config.stockfish_levels:
        for level_game_index in range(level.games):
            opening_index, starting_fen = starting_fen_for_game(config.opening_fens, level_game_index)
            scheduled.append(
                ScheduledStockfishGame(
                    game_index=game_index,
                    level_game_index=level_game_index,
                    level=level,
                    mcchess_color=chess.WHITE if game_index % 2 == 0 else chess.BLACK,
                    opening_index=opening_index,
                    starting_fen=starting_fen,
                )
            )
            game_index += 1
    return scheduled


def protocol_summary(config: StockfishEvalConfig) -> dict[str, Any]:
    """Return protocol metadata shared by completed and failed results."""

    return {
        "scope_note": SCOPE_NOTE,
        "max_ply": config.max_ply,
        "move_delay_seconds": config.move_delay_seconds,
        "print_moves": config.print_moves,
        "draw_rule": DRAW_RULE,
        "color_policy": COLOR_POLICY,
        "opening_protocol": opening_protocol(config.opening_fens),
        "opening_count": len(config.opening_fens) if config.opening_fens else 1,
        "mcts_budget": {"simulations": 200, "c_puct": config.agent.c_puct or 1.5},
        "stockfish_levels": [level_config_dict(level) for level in config.stockfish_levels],
    }


def level_config_dict(level: StockfishLevelConfig) -> dict[str, object]:
    return {
        "name": level.name,
        "games": level.games,
        "options": dict(level.options),
        "limit": dict(level.limit),
        "include_in_elo": level.include_in_elo,
    }


def _load_level(raw: object) -> StockfishLevelConfig:
    if not isinstance(raw, dict):
        raise ValueError("each stockfish_levels item must be a YAML mapping")
    data: dict[str, Any] = dict(raw)
    options = data.get("options") or {}
    limit = data.get("limit") or {}
    if not isinstance(options, dict):
        raise ValueError("Stockfish level options must be a YAML mapping")
    if not isinstance(limit, dict):
        raise ValueError("Stockfish level limit must be a YAML mapping")
    data["options"] = dict(options)
    data["limit"] = dict(limit)
    return StockfishLevelConfig(**data)
