"""Match orchestration for external Stockfish benchmarks."""

from __future__ import annotations

import datetime as dt
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from mcchess.bots import Bot
from mcchess.eval.arena import BotConfig, build_bot
from mcchess.eval.schema import result_envelope
from mcchess.eval.stockfish_protocol import (
    COLOR_POLICY,
    DRAW_RULE,
    GameCallback,
    MoveCallback,
    SCOPE_NOTE,
    StockfishEvalConfig,
    StockfishGameRecord,
    iter_scheduled_games,
    play_stockfish_game,
    protocol_summary,
)
from mcchess.eval.stockfish_rating import estimate_mcchess_elo, insufficient_elo_estimate
from mcchess.eval.stockfish_utils import UciEngine


class BotBuilder(Protocol):
    def __call__(self, config: BotConfig, *, default_seed: int) -> Bot:
        """Build a configured bot."""


def run_stockfish_match(
    config: StockfishEvalConfig,
    engine: UciEngine,
    *,
    stockfish_id: Mapping[str, str] | None = None,
    stockfish_available_options: Sequence[str] | None = None,
    git_commit: str | None = None,
    config_path: str | None = None,
    move_callback: MoveCallback | None = None,
    game_callback: GameCallback | None = None,
    bot_builder: BotBuilder = build_bot,
) -> dict[str, Any]:
    """Run the configured external Stockfish benchmark."""

    random.seed(config.seed)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    start_time = time.perf_counter()
    agent = bot_builder(config.agent, default_seed=config.seed)
    games: list[StockfishGameRecord] = []
    wins = draws = losses = 0
    illegal_moves = 0
    status = "completed"
    failure: dict[str, Any] | None = None

    for scheduled in iter_scheduled_games(config):
        game = play_stockfish_game(
            agent,
            engine,
            scheduled,
            max_ply=config.max_ply,
            move_delay_seconds=config.move_delay_seconds,
            move_callback=move_callback,
        )
        games.append(game)
        if game_callback is not None:
            game_callback(game)

        if game.status != "completed":
            status = "failed"
            illegal_moves += 1 if game.illegal_move is not None else 0
            failure = {
                "game_index": game.game_index,
                "level": game.level,
                "termination": game.termination,
                "illegal_move": game.illegal_move,
                "error": game.error,
            }
            break

        if game.mcchess_score == 1.0:
            wins += 1
        elif game.mcchess_score == 0.5:
            draws += 1
        elif game.mcchess_score == 0.0:
            losses += 1
        else:
            raise RuntimeError("completed game has no McChess score")

    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    completed_games = wins + draws + losses
    score = (wins + 0.5 * draws) / completed_games if completed_games else 0.0
    options_data = list(
        stockfish_available_options
        if stockfish_available_options is not None
        else sorted(str(name) for name in engine.options)
    )
    return result_envelope(
        run_id=config.run_id,
        run_type="stockfish_benchmark",
        status=status,
        seed=config.seed,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=time.perf_counter() - start_time,
        git_commit=git_commit,
        config_path=config_path,
        config=asdict(config),
        protocol=protocol_summary(config),
        participants={
            "mcchess": {
                "name": agent.name,
                "checkpoint_path": config.agent.checkpoint_path,
            },
            "stockfish": {
                "path": config.stockfish_path,
                "id": dict(stockfish_id or engine.id),
                "available_options": options_data,
            },
        },
        summary={
            "games_scheduled": config.num_games,
            "games_completed": completed_games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": score,
            "illegal_moves": illegal_moves,
            "failure": failure,
        },
        metrics={"elo_estimate": asdict(estimate_mcchess_elo(games))},
        games=[asdict(game) for game in games],
    )


def setup_failure_result(
    config: StockfishEvalConfig,
    *,
    config_path: str,
    git_commit: str | None,
    started_at: str,
    elapsed_seconds: float,
    error: Exception,
) -> dict[str, Any]:
    """Build a failed result for setup failures before any game starts."""

    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    failure = {
        "stage": "setup",
        "error": f"{type(error).__name__}: {error}",
    }
    return result_envelope(
        run_id=config.run_id,
        run_type="stockfish_benchmark",
        status="failed",
        seed=config.seed,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
        git_commit=git_commit,
        config_path=config_path,
        config=asdict(config),
        protocol=protocol_summary(config),
        participants={
            "mcchess": {
                "name": config.agent.name,
                "checkpoint_path": config.agent.checkpoint_path,
            },
            "stockfish": {
                "path": config.stockfish_path,
                "id": None,
                "available_options": [],
            },
        },
        summary={
            "games_scheduled": config.num_games,
            "games_completed": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "score": 0.0,
            "illegal_moves": 0,
            "failure": failure,
        },
        metrics={"elo_estimate": asdict(insufficient_elo_estimate())},
        games=[],
    )


__all__ = [
    "COLOR_POLICY",
    "DRAW_RULE",
    "SCOPE_NOTE",
    "run_stockfish_match",
    "setup_failure_result",
]
