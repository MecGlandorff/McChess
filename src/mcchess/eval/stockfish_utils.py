"""Shared Stockfish UCI helpers for external evaluation."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Protocol, TypeAlias

import chess
import chess.engine

from mcchess.eval.common import resolve_executable

EngineOptionValue: TypeAlias = str | int | bool | None


class UciEngine(Protocol):
    """Small subset of the python-chess UCI engine API used by eval code."""

    @property
    def id(self) -> Mapping[str, str]:
        """Engine identity fields."""
        ...

    @property
    def options(self) -> Mapping[str, object]:
        """Available UCI options."""
        ...

    def configure(self, options: Mapping[str, EngineOptionValue]) -> None:
        """Configure engine options before a game."""

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> object:
        """Return an object with a ``move`` attribute."""


def resolve_stockfish_path(config_value: str | None, override: str | None = None) -> str:
    """Resolve Stockfish from CLI override, config, environment, or PATH."""

    return resolve_executable(
        explicit=override,
        config_value=config_value,
        env_value=os.environ.get("STOCKFISH_PATH"),
        path_name="stockfish",
        display_name="Stockfish",
    )


def engine_limit(raw_limit: Mapping[str, float | int]) -> chess.engine.Limit:
    """Convert a YAML-friendly Stockfish search limit into python-chess form."""

    allowed = {"time", "depth", "nodes"}
    unknown = set(raw_limit) - allowed
    if unknown:
        raise ValueError(f"unsupported Stockfish limit fields: {sorted(unknown)}")
    if not raw_limit:
        raise ValueError("Stockfish limit must set at least one of time, depth, or nodes")

    time_limit: float | None = None
    depth_limit: int | None = None
    nodes_limit: int | None = None
    if "time" in raw_limit:
        time_value = float(raw_limit["time"])
        if not math.isfinite(time_value) or time_value <= 0.0:
            raise ValueError("Stockfish limit time must be a positive finite value")
        time_limit = time_value
    if "depth" in raw_limit:
        depth = int(raw_limit["depth"])
        if depth <= 0:
            raise ValueError("Stockfish limit depth must be positive")
        depth_limit = depth
    if "nodes" in raw_limit:
        nodes = int(raw_limit["nodes"])
        if nodes <= 0:
            raise ValueError("Stockfish limit nodes must be positive")
        nodes_limit = nodes
    return chess.engine.Limit(time=time_limit, depth=depth_limit, nodes=nodes_limit)


def uci_elo_options(elo: int, skill_level: int = 20) -> dict[str, str | int | bool]:
    """Return Stockfish UCI options for one limited-strength Elo level."""

    return {
        "Skill Level": skill_level,
        "UCI_LimitStrength": True,
        "UCI_Elo": elo,
    }


def start_uci_engine_pair(path: str) -> tuple[chess.engine.SimpleEngine, chess.engine.SimpleEngine]:
    """Start two UCI engines and clean up the first if the second fails."""

    first = chess.engine.SimpleEngine.popen_uci(path)
    try:
        second = chess.engine.SimpleEngine.popen_uci(path)
    except BaseException:
        first.quit()
        raise
    return first, second
