"""Reproducible bot-vs-bot arena evaluation."""

from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import chess

from mcchess.bots import Bot, MaterialBot, NegamaxBot, PolicyOnlyBot, RandomLegalBot

DRAW_RULE: Final[str] = "python_chess_outcome_or_max_ply_draw"
COLOR_POLICY: Final[str] = "alternating_agent_white_first"
OPENING_PROTOCOL: Final[str] = "standard_initial_position"


@dataclass(frozen=True)
class BotConfig:
    """YAML-friendly bot configuration for arena runs."""

    kind: str
    name: str | None = None
    seed: int | None = None
    depth: int | None = None
    checkpoint_path: str | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.kind not in {"random", "material", "negamax", "policy_only"}:
            raise ValueError(f"unsupported bot kind: {self.kind}")
        if self.kind == "negamax" and self.depth is not None and self.depth < 1:
            raise ValueError("negamax depth must be at least 1")
        if self.kind == "policy_only" and not self.checkpoint_path:
            raise ValueError("policy_only bot requires checkpoint_path")


@dataclass(frozen=True)
class ArenaConfig:
    """Configuration for a fixed bot-vs-bot arena run."""

    output_path: str
    agent: BotConfig
    opponent: BotConfig
    run_id: str = "arena"
    seed: int = 0
    num_games: int = 20
    max_ply: int = 160

    def __post_init__(self) -> None:
        if self.num_games <= 0:
            raise ValueError("num_games must be positive")
        if self.max_ply <= 0:
            raise ValueError("max_ply must be positive")
        if not self.run_id:
            raise ValueError("run_id must not be empty")


@dataclass(frozen=True)
class GameRecord:
    """Single arena game record from the named agent's perspective."""

    game_index: int
    status: str
    agent_color: str
    white: str
    black: str
    result: str
    winner: str | None
    agent_score: float | None
    termination: str
    ply_count: int
    final_fen: str
    moves: list[str]
    illegal_move: dict[str, str] | None = None
    error: str | None = None


def build_bot(config: BotConfig, *, default_seed: int) -> Bot:
    """Build a configured bot instance."""

    name = config.name or _default_bot_name(config)
    if config.kind == "random":
        return RandomLegalBot(seed=config.seed if config.seed is not None else default_seed, name=name)
    if config.kind == "material":
        return MaterialBot(name=name)
    if config.kind == "negamax":
        return NegamaxBot(depth=config.depth or 2, name=name)
    if config.kind == "policy_only":
        assert config.checkpoint_path is not None
        return PolicyOnlyBot.from_checkpoint(config.checkpoint_path, device=config.device, name=name)
    raise ValueError(f"unsupported bot kind: {config.kind}")


def play_game(
    agent: Bot,
    opponent: Bot,
    *,
    game_index: int,
    agent_color: chess.Color,
    max_ply: int,
) -> GameRecord:
    """Play one game and return a serializable record.

    The board handed to each bot is a copy so an arena participant cannot mutate
    the official game state directly.
    """

    board = chess.Board()
    moves: list[str] = []
    white_name = agent.name if agent_color == chess.WHITE else opponent.name
    black_name = agent.name if agent_color == chess.BLACK else opponent.name

    while len(moves) < max_ply:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return _completed_record(
                game_index=game_index,
                agent_color=agent_color,
                white_name=white_name,
                black_name=black_name,
                board=board,
                moves=moves,
                outcome=outcome,
                termination=outcome.termination.name.lower(),
            )

        bot = agent if board.turn == agent_color else opponent
        color_name = _color_name(board.turn)
        try:
            move = bot.choose_move(board.copy(stack=True))
        except Exception as exc:  # noqa: BLE001 - record bot failures in result JSON.
            return _failed_record(
                game_index=game_index,
                agent_color=agent_color,
                white_name=white_name,
                black_name=black_name,
                board=board,
                moves=moves,
                termination="bot_error",
                error=f"{type(exc).__name__}: {exc}",
            )

        if move not in board.legal_moves:
            return _failed_record(
                game_index=game_index,
                agent_color=agent_color,
                white_name=white_name,
                black_name=black_name,
                board=board,
                moves=moves,
                termination="illegal_move",
                illegal_move={
                    "bot": bot.name,
                    "color": color_name,
                    "move": move.uci(),
                },
            )

        board.push(move)
        moves.append(move.uci())

    return _completed_record(
        game_index=game_index,
        agent_color=agent_color,
        white_name=white_name,
        black_name=black_name,
        board=board,
        moves=moves,
        outcome=None,
        termination="max_ply",
    )


def run_match(config: ArenaConfig) -> dict[str, Any]:
    """Run an arena match and return a JSON-serializable result."""

    random.seed(config.seed)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    start_time = time.perf_counter()
    agent = build_bot(config.agent, default_seed=config.seed)
    opponent = build_bot(config.opponent, default_seed=config.seed + 1)
    games: list[GameRecord] = []
    wins = draws = losses = 0
    illegal_moves = 0
    status = "completed"
    failure: dict[str, Any] | None = None

    for game_index in range(config.num_games):
        agent_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
        game = play_game(
            agent,
            opponent,
            game_index=game_index,
            agent_color=agent_color,
            max_ply=config.max_ply,
        )
        games.append(game)

        if game.status != "completed":
            status = "failed"
            illegal_moves += 1 if game.illegal_move is not None else 0
            failure = {
                "game_index": game.game_index,
                "termination": game.termination,
                "illegal_move": game.illegal_move,
                "error": game.error,
            }
            break

        if game.agent_score == 1.0:
            wins += 1
        elif game.agent_score == 0.5:
            draws += 1
        elif game.agent_score == 0.0:
            losses += 1
        else:
            raise RuntimeError("completed game has no agent score")

    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    completed_games = wins + draws + losses
    score = (wins + 0.5 * draws) / completed_games if completed_games else 0.0
    result = {
        "run_id": config.run_id,
        "status": status,
        "seed": config.seed,
        "num_games": config.num_games,
        "games_completed": completed_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "illegal_moves": illegal_moves,
        "max_ply": config.max_ply,
        "draw_rule": DRAW_RULE,
        "color_policy": COLOR_POLICY,
        "opening_protocol": OPENING_PROTOCOL,
        "agent": agent.name,
        "opponent": opponent.name,
        "agent_checkpoint": config.agent.checkpoint_path,
        "opponent_checkpoint": config.opponent.checkpoint_path,
        "mcts_budget": None,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": time.perf_counter() - start_time,
        "failure": failure,
        "games": [asdict(game) for game in games],
    }
    return result


def _default_bot_name(config: BotConfig) -> str:
    if config.kind == "negamax":
        return f"negamax_depth_{config.depth or 2}"
    if config.kind == "policy_only":
        checkpoint_name = Path(config.checkpoint_path or "checkpoint").stem
        return f"policy_only:{checkpoint_name}"
    return config.kind


def _completed_record(
    *,
    game_index: int,
    agent_color: chess.Color,
    white_name: str,
    black_name: str,
    board: chess.Board,
    moves: list[str],
    outcome: chess.Outcome | None,
    termination: str,
) -> GameRecord:
    winner = outcome.winner if outcome is not None else None
    result = outcome.result() if outcome is not None else "1/2-1/2"
    agent_score = _agent_score(winner, agent_color)
    return GameRecord(
        game_index=game_index,
        status="completed",
        agent_color=_color_name(agent_color),
        white=white_name,
        black=black_name,
        result=result,
        winner=_color_name(winner) if winner is not None else None,
        agent_score=agent_score,
        termination=termination,
        ply_count=len(moves),
        final_fen=board.fen(),
        moves=list(moves),
    )


def _failed_record(
    *,
    game_index: int,
    agent_color: chess.Color,
    white_name: str,
    black_name: str,
    board: chess.Board,
    moves: list[str],
    termination: str,
    illegal_move: dict[str, str] | None = None,
    error: str | None = None,
) -> GameRecord:
    return GameRecord(
        game_index=game_index,
        status="failed",
        agent_color=_color_name(agent_color),
        white=white_name,
        black=black_name,
        result="*",
        winner=None,
        agent_score=None,
        termination=termination,
        ply_count=len(moves),
        final_fen=board.fen(),
        moves=list(moves),
        illegal_move=illegal_move,
        error=error,
    )


def _agent_score(winner: chess.Color | None, agent_color: chess.Color) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == agent_color else 0.0


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"
