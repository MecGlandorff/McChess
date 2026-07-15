"""Reproducible bot-vs-bot arena evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import chess
import yaml  # type: ignore[import-untyped]

from mcchess.bots import Bot, MCTSBot, MaterialBot, NegamaxBot, PolicyOnlyBot, RandomLegalBot
from mcchess.eval.common import git_commit as current_git_commit
from mcchess.eval.common import load_yaml_mapping, write_json_atomic, write_text_atomic
from mcchess.eval.openings import opening_protocol, normalize_opening_fens, starting_fen_for_game
from mcchess.eval.schema import result_envelope

DRAW_RULE: Final[str] = "python_chess_outcome_or_max_ply_draw"
COLOR_POLICY: Final[str] = "alternating_agent_white_first"
MoveCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class BotConfig:
    """YAML-friendly bot configuration for arena runs."""

    kind: str
    name: str | None = None
    seed: int | None = None
    depth: int | None = None
    checkpoint_path: str | None = None
    device: str = "auto"
    simulations: int | None = None
    c_puct: float | None = None
    inference_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"random", "material", "negamax", "policy_only", "mcts"}:
            raise ValueError(f"unsupported bot kind: {self.kind}")
        if self.kind == "negamax" and self.depth is not None and self.depth < 1:
            raise ValueError("negamax depth must be at least 1")
        if self.kind in {"policy_only", "mcts"} and not self.checkpoint_path:
            raise ValueError(f"{self.kind} bot requires checkpoint_path")
        if self.simulations is not None and self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.c_puct is not None and (not math.isfinite(self.c_puct) or self.c_puct <= 0.0):
            raise ValueError("c_puct must be a positive finite value")
        if self.inference_batch_size is not None and self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")


@dataclass(frozen=True)
class ArenaConfig:
    """Configuration for a fixed bot-vs-bot arena run."""

    output_dir: str
    agent: BotConfig
    opponent: BotConfig
    run_id: str = "arena"
    seed: int = 0
    num_games: int = 20
    max_ply: int = 160
    move_delay_seconds: float = 0.0
    print_moves: bool = False
    opening_fens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        object.__setattr__(self, "opening_fens", normalize_opening_fens(self.opening_fens))
        if self.num_games <= 0:
            raise ValueError("num_games must be positive")
        if self.max_ply <= 0:
            raise ValueError("max_ply must be positive")
        if self.move_delay_seconds < 0.0:
            raise ValueError("move_delay_seconds must be non-negative")
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
    opening_index: int | None
    starting_fen: str
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
    if config.kind == "mcts":
        assert config.checkpoint_path is not None
        return MCTSBot.from_checkpoint(
            config.checkpoint_path,
            device=config.device,
            name=name,
            simulations=_mcts_simulations(config),
            c_puct=_mcts_c_puct(config),
            inference_batch_size=_mcts_inference_batch_size(config),
        )
    raise ValueError(f"unsupported bot kind: {config.kind}")


def play_game(
    agent: Bot,
    opponent: Bot,
    *,
    game_index: int,
    agent_color: chess.Color,
    max_ply: int,
    starting_fen: str = chess.STARTING_FEN,
    opening_index: int | None = None,
    move_delay_seconds: float = 0.0,
    move_callback: MoveCallback | None = None,
) -> GameRecord:
    """Play one game and return a serializable record.

    The board handed to each bot is a copy so an arena participant cannot mutate
    the official game state directly.
    """

    board = chess.Board(starting_fen)
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
                opening_index=opening_index,
                starting_fen=starting_fen,
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
                opening_index=opening_index,
                starting_fen=starting_fen,
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
                opening_index=opening_index,
                starting_fen=starting_fen,
                termination="illegal_move",
                illegal_move={
                    "bot": bot.name,
                    "color": color_name,
                    "move": move.uci(),
                },
            )

        san = board.san(move)
        board.push(move)
        moves.append(move.uci())
        if move_callback is not None:
            move_callback(
                {
                    "game_index": game_index,
                    "ply": len(moves),
                    "color": color_name,
                    "bot": bot.name,
                    "uci": move.uci(),
                    "san": san,
                    "fen": board.fen(),
                    "opening_index": opening_index,
                }
            )
        if move_delay_seconds > 0.0:
            time.sleep(move_delay_seconds)

    return _completed_record(
        game_index=game_index,
        agent_color=agent_color,
        white_name=white_name,
        black_name=black_name,
        board=board,
        moves=moves,
        opening_index=opening_index,
        starting_fen=starting_fen,
        outcome=None,
        termination="max_ply",
    )


def run_match(
    config: ArenaConfig,
    *,
    move_callback: MoveCallback | None = None,
    git_commit: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
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
        opening_index, starting_fen = starting_fen_for_game(config.opening_fens, game_index)
        game = play_game(
            agent,
            opponent,
            game_index=game_index,
            agent_color=agent_color,
            max_ply=config.max_ply,
            opening_index=opening_index,
            starting_fen=starting_fen,
            move_delay_seconds=config.move_delay_seconds,
            move_callback=move_callback,
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
    return result_envelope(
        run_id=config.run_id,
        run_type="arena",
        status=status,
        seed=config.seed,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=time.perf_counter() - start_time,
        git_commit=git_commit,
        config_path=config_path,
        config=asdict(config),
        protocol={
            "max_ply": config.max_ply,
            "move_delay_seconds": config.move_delay_seconds,
            "print_moves": config.print_moves,
            "draw_rule": DRAW_RULE,
            "color_policy": COLOR_POLICY,
            "opening_protocol": opening_protocol(config.opening_fens),
            "opening_count": len(config.opening_fens) if config.opening_fens else 1,
            "mcts_budget": _match_mcts_budget(config),
        },
        participants={
            "agent": {"name": agent.name, "checkpoint_path": config.agent.checkpoint_path},
            "opponent": {"name": opponent.name, "checkpoint_path": config.opponent.checkpoint_path},
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
        games=[asdict(game) for game in games],
    )


def load_config(path: str | Path) -> ArenaConfig:
    """Load an arena YAML config."""

    raw = load_yaml_mapping(path)
    agent_raw = raw.pop("agent", None)
    opponent_raw = raw.pop("opponent", None)
    if not isinstance(agent_raw, dict):
        raise ValueError("agent must be a YAML mapping")
    if not isinstance(opponent_raw, dict):
        raise ValueError("opponent must be a YAML mapping")
    return ArenaConfig(
        **raw,
        agent=BotConfig(**agent_raw),
        opponent=BotConfig(**opponent_raw),
    )


def write_artifacts(config: ArenaConfig, result: dict[str, Any]) -> Path:
    """Write the arena result artifact and return its path."""

    output_dir = Path(config.output_dir)
    write_text_atomic(
        output_dir / "config.yaml",
        yaml.safe_dump(asdict(config), sort_keys=False),
    )
    result_path = output_dir / "result.json"
    write_json_atomic(result_path, result)
    config_path = result.get("run", {}).get("config_path")
    if isinstance(config_path, str):
        write_text_atomic(output_dir / "source_config_path.txt", config_path + "\n")
    return result_path


def print_move_event(event: dict[str, Any]) -> None:
    """Print one live arena move event."""

    print(
        f"game {event['game_index'] + 1:03d} "
        f"ply {event['ply']:03d} "
        f"{event['color']} {event['bot']}: {event['san']} ({event['uci']})",
        flush=True,
    )


def run_arena(config_path: str | Path) -> Path:
    """Run an arena config and write schema-v2 artifacts."""

    config_path = Path(config_path)
    config = load_config(config_path)
    result = run_match(
        config,
        move_callback=print_move_event if config.print_moves else None,
        git_commit=current_git_commit(),
        config_path=str(config_path),
    )
    result_path = write_artifacts(config, result)
    print(f"saved arena result to {result_path}")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a McChess bot-vs-bot arena.")
    parser.add_argument("config", type=Path, help="YAML arena config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_arena(args.config)
    return 0


def _default_bot_name(config: BotConfig) -> str:
    if config.kind == "negamax":
        return f"negamax_depth_{config.depth or 2}"
    if config.kind == "policy_only":
        checkpoint_name = Path(config.checkpoint_path or "checkpoint").stem
        return f"policy_only:{checkpoint_name}"
    if config.kind == "mcts":
        checkpoint_name = Path(config.checkpoint_path or "checkpoint").stem
        return f"mcts_{_mcts_simulations(config)}:{checkpoint_name}"
    return config.kind


def _mcts_simulations(config: BotConfig) -> int:
    return config.simulations if config.simulations is not None else 50


def _mcts_c_puct(config: BotConfig) -> float:
    return config.c_puct if config.c_puct is not None else 1.5


def _mcts_inference_batch_size(config: BotConfig) -> int:
    return config.inference_batch_size if config.inference_batch_size is not None else 1


def _bot_mcts_budget(config: BotConfig) -> dict[str, float | int] | None:
    if config.kind != "mcts":
        return None
    return {
        "simulations": _mcts_simulations(config),
        "c_puct": _mcts_c_puct(config),
        "inference_batch_size": _mcts_inference_batch_size(config),
    }


def _match_mcts_budget(config: ArenaConfig) -> dict[str, dict[str, float | int] | None] | None:
    agent_budget = _bot_mcts_budget(config.agent)
    opponent_budget = _bot_mcts_budget(config.opponent)
    if agent_budget is None and opponent_budget is None:
        return None
    return {
        "agent": agent_budget,
        "opponent": opponent_budget,
    }


def _completed_record(
    *,
    game_index: int,
    agent_color: chess.Color,
    white_name: str,
    black_name: str,
    board: chess.Board,
    moves: list[str],
    outcome: chess.Outcome | None,
    opening_index: int | None,
    starting_fen: str,
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
        opening_index=opening_index,
        starting_fen=starting_fen,
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
    opening_index: int | None,
    starting_fen: str,
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
        opening_index=opening_index,
        starting_fen=starting_fen,
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


if __name__ == "__main__":
    raise SystemExit(main())
