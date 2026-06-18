"""Run a Stockfish UCI_Elo self-consistency ladder check."""

from __future__ import annotations

import argparse
import datetime as dt
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import chess
import chess.engine
from tqdm.auto import tqdm  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from mcchess.eval.common import git_commit as current_git_commit
from mcchess.eval.common import load_yaml_mapping, write_csv, write_json_atomic, write_text_atomic
from mcchess.eval.schema import result_envelope
from mcchess.eval.stockfish_utils import UciEngine
from mcchess.eval.stockfish_utils import engine_limit as stockfish_engine_limit
from mcchess.eval.stockfish_utils import resolve_stockfish_path as resolve_stockfish_binary_path
from mcchess.eval.stockfish_utils import start_uci_engine_pair, uci_elo_options

SCOPE_NOTE = (
    "Stockfish UCI_Elo self-consistency diagnostic only. It checks ordering of "
    "configured Stockfish handicap levels under this local limit. It is not "
    "calibration to Lichess, FIDE, CCRL, or McChess strength."
)
DRAW_RULE = "python_chess_outcome_or_max_ply_draw"
COLOR_POLICY = "paired_colors_per_adjacent_uci_elo_pair"
OPENING_PROTOCOL = "standard_initial_position"
GAME_CSV_FIELDS = [
    "Game",
    "Pair",
    "White Elo",
    "Black Elo",
    "Result",
    "Winner Elo",
    "Higher Elo Score",
    "Termination",
]


@dataclass(frozen=True)
class StockfishLadderConfig:
    """Configuration for a Stockfish UCI_Elo self-consistency check."""

    run_id: str
    output_dir: str
    stockfish_elos: list[int]
    stockfish_path: str | None = None
    seed: int = 0
    max_ply: int = 180
    skill_level: int = 20
    limit: dict[str, float | int] | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if len(self.stockfish_elos) < 2:
            raise ValueError("stockfish_elos must contain at least two levels")
        if sorted(set(self.stockfish_elos)) != self.stockfish_elos:
            raise ValueError("stockfish_elos must be strictly increasing")
        if any(elo <= 0 for elo in self.stockfish_elos):
            raise ValueError("stockfish_elos must be positive")
        if self.max_ply <= 0:
            raise ValueError("max_ply must be positive")
        if self.skill_level < 0:
            raise ValueError("skill_level must be non-negative")
        stockfish_engine_limit(self.limit or {})

    @property
    def num_games(self) -> int:
        """Two paired games for every adjacent Elo pair."""

        return 2 * (len(self.stockfish_elos) - 1)


@dataclass(frozen=True)
class ScheduledLadderGame:
    """One scheduled Stockfish-vs-Stockfish game."""

    game_index: int
    pair_index: int
    lower_elo: int
    higher_elo: int
    white_elo: int
    black_elo: int


@dataclass(frozen=True)
class LadderGameRecord:
    """Serializable result for one ladder self-check game."""

    game_index: int
    pair_index: int
    lower_elo: int
    higher_elo: int
    white_elo: int
    black_elo: int
    status: str
    result: str
    winner: str | None
    winner_elo: int | None
    higher_elo_score: float | None
    termination: str
    ply_count: int
    final_fen: str
    moves: list[str]
    white_options: dict[str, str | int | bool]
    black_options: dict[str, str | int | bool]
    stockfish_limit: dict[str, float | int]
    illegal_move: dict[str, str] | None = None
    error: str | None = None


def load_config(path: str | Path) -> StockfishLadderConfig:
    """Load a Stockfish ladder diagnostic YAML config."""

    data = load_yaml_mapping(path)
    return StockfishLadderConfig(**data)


def resolve_stockfish_path(config: StockfishLadderConfig, override: str | None = None) -> str:
    """Resolve Stockfish from CLI override, config, environment, or PATH."""

    return resolve_stockfish_binary_path(config.stockfish_path, override)


def iter_ladder_games(config: StockfishLadderConfig) -> list[ScheduledLadderGame]:
    """Expand adjacent Elo pairs into paired-color games."""

    scheduled: list[ScheduledLadderGame] = []
    game_index = 0
    for pair_index, (lower_elo, higher_elo) in enumerate(
        zip(config.stockfish_elos, config.stockfish_elos[1:])
    ):
        scheduled.append(
            ScheduledLadderGame(
                game_index=game_index,
                pair_index=pair_index,
                lower_elo=lower_elo,
                higher_elo=higher_elo,
                white_elo=lower_elo,
                black_elo=higher_elo,
            )
        )
        game_index += 1
        scheduled.append(
            ScheduledLadderGame(
                game_index=game_index,
                pair_index=pair_index,
                lower_elo=lower_elo,
                higher_elo=higher_elo,
                white_elo=higher_elo,
                black_elo=lower_elo,
            )
        )
        game_index += 1
    return scheduled


def play_ladder_game(
    white_engine: UciEngine,
    black_engine: UciEngine,
    scheduled: ScheduledLadderGame,
    *,
    max_ply: int,
    skill_level: int,
    raw_limit: Mapping[str, float | int],
) -> LadderGameRecord:
    """Play one Stockfish handicap level against another."""

    board = chess.Board()
    moves: list[str] = []
    white_options = uci_elo_options(scheduled.white_elo, skill_level)
    black_options = uci_elo_options(scheduled.black_elo, skill_level)
    try:
        white_engine.configure(white_options)
        black_engine.configure(black_options)
    except Exception as exc:  # noqa: BLE001 - record external engine failures.
        return _failed_record(
            scheduled=scheduled,
            board=board,
            moves=moves,
            white_options=white_options,
            black_options=black_options,
            raw_limit=raw_limit,
            termination="engine_config_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    limit = stockfish_engine_limit(raw_limit)
    while len(moves) < max_ply:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return _completed_record(
                scheduled=scheduled,
                board=board,
                moves=moves,
                outcome=outcome,
                white_options=white_options,
                black_options=black_options,
                raw_limit=raw_limit,
                termination=outcome.termination.name.lower(),
            )

        engine = white_engine if board.turn == chess.WHITE else black_engine
        actor_elo = scheduled.white_elo if board.turn == chess.WHITE else scheduled.black_elo
        color_name = "white" if board.turn == chess.WHITE else "black"
        try:
            play_result = engine.play(board.copy(stack=True), limit)
            raw_move = getattr(play_result, "move", None)
            if not isinstance(raw_move, chess.Move):
                raise ValueError("Stockfish returned no move")
            move = raw_move
        except Exception as exc:  # noqa: BLE001 - record external engine failures.
            return _failed_record(
                scheduled=scheduled,
                board=board,
                moves=moves,
                white_options=white_options,
                black_options=black_options,
                raw_limit=raw_limit,
                termination="engine_play_error",
                error=f"{type(exc).__name__}: {exc}",
            )

        if move not in board.legal_moves:
            return _failed_record(
                scheduled=scheduled,
                board=board,
                moves=moves,
                white_options=white_options,
                black_options=black_options,
                raw_limit=raw_limit,
                termination="illegal_move",
                illegal_move={
                    "elo": str(actor_elo),
                    "color": color_name,
                    "move": move.uci(),
                },
            )

        board.push(move)
        moves.append(move.uci())

    return _completed_record(
        scheduled=scheduled,
        board=board,
        moves=moves,
        outcome=None,
        white_options=white_options,
        black_options=black_options,
        raw_limit=raw_limit,
        termination="max_ply",
    )


def run_ladder_match(
    config: StockfishLadderConfig,
    white_engine: UciEngine,
    black_engine: UciEngine,
    *,
    stockfish_id: Mapping[str, str] | None = None,
    stockfish_available_options: Sequence[str] | None = None,
    git_commit: str | None = None,
    config_path: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run the configured Stockfish UCI_Elo self-check."""

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    start_time = time.perf_counter()
    games: list[LadderGameRecord] = []
    status = "completed"
    failure: dict[str, Any] | None = None
    raw_limit = config.limit or {}

    scheduled_games = iter_ladder_games(config)
    with tqdm(
        scheduled_games,
        desc="stockfish ladder",
        unit="game",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        for scheduled in progress:
            game = play_ladder_game(
                white_engine,
                black_engine,
                scheduled,
                max_ply=config.max_ply,
                skill_level=config.skill_level,
                raw_limit=raw_limit,
            )
            games.append(game)
            progress.set_postfix(
                pair=f"{scheduled.lower_elo}-{scheduled.higher_elo}",
                result=game.result,
                score="" if game.higher_elo_score is None else f"{game.higher_elo_score:.1f}",
            )
            if game.status != "completed":
                status = "failed"
                failure = {
                    "game_index": game.game_index,
                    "termination": game.termination,
                    "illegal_move": game.illegal_move,
                    "error": game.error,
                }
                break

    completed_games = [game for game in games if game.status == "completed"]
    scores: list[float] = []
    for game in completed_games:
        score = game.higher_elo_score
        if score is not None:
            scores.append(score)
    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    higher_elo_score = (sum(scores) / len(scores)) if scores else None
    return result_envelope(
        run_id=config.run_id,
        run_type="stockfish_ladder",
        status=status,
        seed=config.seed,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=time.perf_counter() - start_time,
        git_commit=git_commit,
        config_path=config_path,
        config=asdict(config),
        protocol={
            "scope_note": SCOPE_NOTE,
            "stockfish_elos": list(config.stockfish_elos),
            "max_ply": config.max_ply,
            "skill_level": config.skill_level,
            "limit": raw_limit,
            "draw_rule": DRAW_RULE,
            "color_policy": COLOR_POLICY,
            "opening_protocol": OPENING_PROTOCOL,
        },
        participants={
            "stockfish": {
                "path": config.stockfish_path,
                "id": dict(stockfish_id or white_engine.id),
                "available_options": list(
                    stockfish_available_options
                    if stockfish_available_options is not None
                    else sorted(str(name) for name in white_engine.options)
                ),
            }
        },
        summary={
            "games_scheduled": config.num_games,
            "games_completed": len(completed_games),
            "higher_elo_score": higher_elo_score,
            "failure": failure,
        },
        metrics={"pair_summary": pair_summary(games)},
        games=[asdict(game) for game in games],
    )


def pair_summary(games: Sequence[LadderGameRecord]) -> list[dict[str, Any]]:
    """Aggregate completed ladder games by adjacent pair."""

    summaries: list[dict[str, Any]] = []
    pair_keys = sorted({(game.pair_index, game.lower_elo, game.higher_elo) for game in games})
    for pair_index, lower_elo, higher_elo in pair_keys:
        pair_games = [
            game
            for game in games
            if game.pair_index == pair_index
            and game.status == "completed"
            and game.higher_elo_score is not None
        ]
        score = sum(_required_score(game) for game in pair_games)
        summaries.append(
            {
                "pair_index": pair_index,
                "lower_elo": lower_elo,
                "higher_elo": higher_elo,
                "games_completed": len(pair_games),
                "higher_elo_score": score,
                "higher_elo_score_rate": score / len(pair_games) if pair_games else None,
            }
        )
    return summaries


def _required_score(game: LadderGameRecord) -> float:
    if game.higher_elo_score is None:
        raise ValueError("completed ladder game has no higher Elo score")
    return game.higher_elo_score


def write_artifacts(
    *,
    output_dir: Path,
    config_path: Path,
    config: StockfishLadderConfig,
    result: dict[str, Any],
) -> Path:
    """Write config copy, JSON result, CSV table, and Markdown report."""

    write_text_atomic(
        output_dir / "config.yaml",
        yaml.safe_dump(asdict(config), sort_keys=False),
    )
    result_path = output_dir / "result.json"
    write_json_atomic(result_path, result)
    write_csv(output_dir / "games.csv", GAME_CSV_FIELDS, game_rows(result["games"]))
    write_text_atomic(output_dir / "report.md", format_report(result))
    write_text_atomic(output_dir / "source_config_path.txt", str(config_path) + "\n")
    return result_path


def game_rows(games: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return display-friendly ladder rows."""

    rows: list[dict[str, str]] = []
    for game in games:
        score = game.get("higher_elo_score")
        rows.append(
            {
                "Game": str(int(game["game_index"]) + 1),
                "Pair": f"{game['lower_elo']}-{game['higher_elo']}",
                "White Elo": str(game["white_elo"]),
                "Black Elo": str(game["black_elo"]),
                "Result": str(game["result"]),
                "Winner Elo": "" if game.get("winner_elo") is None else str(game["winner_elo"]),
                "Higher Elo Score": "" if score is None else f"{float(score):.1f}",
                "Termination": str(game["termination"]),
            }
        )
    return rows


def format_report(result: Mapping[str, Any]) -> str:
    """Format a compact Markdown report for the ladder diagnostic."""

    run = _mapping(result.get("run"))
    protocol = _mapping(result.get("protocol"))
    summary = _mapping(result.get("summary"))
    rows = game_rows(result.get("games", []))
    lines = [
        "# Stockfish UCI_Elo Ladder Self-Check",
        "",
        f"Run ID: `{run.get('id', result.get('run_id'))}`",
        "",
        f"Scope: {SCOPE_NOTE}",
        "",
        "## Summary",
        "",
        f"- Status: `{run.get('status', result.get('status'))}`",
        "- Games completed: "
        f"{summary.get('games_completed', result.get('games_completed'))} / "
        f"{summary.get('games_scheduled', result.get('num_games'))}",
        f"- Higher-level score rate: `{summary.get('higher_elo_score', result.get('higher_elo_score'))}`",
        f"- Max ply: {protocol.get('max_ply', result.get('max_ply'))}",
        f"- Draw rule: `{protocol.get('draw_rule', result.get('draw_rule'))}`",
        f"- Color policy: `{protocol.get('color_policy', result.get('color_policy'))}`",
        f"- Opening protocol: `{protocol.get('opening_protocol', result.get('opening_protocol'))}`",
        "",
        "## Game Table",
        "",
        "| Game | Pair | White Elo | Black Elo | Result | Winner Elo | Higher Elo Score | Termination |",
        "|---:|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {Game} | {Pair} | {White Elo} | {Black Elo} | {Result} | "
            "{Winner Elo} | {Higher Elo Score} | {Termination} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- This checks ordering of Stockfish UCI_Elo handicap levels under one local limit.",
            "- This is not a calibration to Lichess Elo, FIDE Elo, CCRL Elo, or McChess strength.",
            "- Do not use Stockfish moves, outcomes, or level settings from this diagnostic as training targets.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def check_stockfish_uci_ladder(
    config_path: str | Path,
    *,
    stockfish_path: str | None = None,
    output_dir: str | Path | None = None,
    show_progress: bool = True,
) -> Path:
    """Run the ladder diagnostic config and return the result JSON path."""

    config_path = Path(config_path)
    config = load_config(config_path)
    resolved_stockfish_path = resolve_stockfish_path(config, stockfish_path)
    config = replace(config, stockfish_path=resolved_stockfish_path)
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))

    white_engine, black_engine = start_uci_engine_pair(resolved_stockfish_path)
    try:
        result = run_ladder_match(
            config,
            white_engine,
            black_engine,
            stockfish_id=dict(white_engine.id),
            stockfish_available_options=sorted(str(name) for name in white_engine.options),
            git_commit=current_git_commit(),
            config_path=str(config_path),
            show_progress=show_progress,
        )
    finally:
        white_engine.quit()
        black_engine.quit()

    result_path = write_artifacts(
        output_dir=Path(config.output_dir),
        config_path=config_path,
        config=config,
        result=result,
    )
    print(f"saved Stockfish ladder diagnostic result to {result_path}")
    print(f"saved game table to {Path(config.output_dir) / 'games.csv'}")
    print(f"saved report to {Path(config.output_dir) / 'report.md'}")
    return result_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Stockfish UCI_Elo self-consistency ladder check."
    )
    parser.add_argument("config", type=Path, help="YAML ladder diagnostic config.")
    parser.add_argument("--stockfish-path", help="Path to the Stockfish executable.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_stockfish_uci_ladder(
        args.config,
        stockfish_path=args.stockfish_path,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )
    return 0


def _completed_record(
    *,
    scheduled: ScheduledLadderGame,
    board: chess.Board,
    moves: list[str],
    outcome: chess.Outcome | None,
    white_options: dict[str, str | int | bool],
    black_options: dict[str, str | int | bool],
    raw_limit: Mapping[str, float | int],
    termination: str,
) -> LadderGameRecord:
    winner = outcome.winner if outcome is not None else None
    result = outcome.result() if outcome is not None else "1/2-1/2"
    winner_elo = _winner_elo(winner, scheduled.white_elo, scheduled.black_elo)
    return LadderGameRecord(
        game_index=scheduled.game_index,
        pair_index=scheduled.pair_index,
        lower_elo=scheduled.lower_elo,
        higher_elo=scheduled.higher_elo,
        white_elo=scheduled.white_elo,
        black_elo=scheduled.black_elo,
        status="completed",
        result=result,
        winner=_color_name(winner),
        winner_elo=winner_elo,
        higher_elo_score=_higher_elo_score(winner_elo, scheduled.higher_elo),
        termination=termination,
        ply_count=len(moves),
        final_fen=board.fen(),
        moves=list(moves),
        white_options=dict(white_options),
        black_options=dict(black_options),
        stockfish_limit=dict(raw_limit),
    )


def _failed_record(
    *,
    scheduled: ScheduledLadderGame,
    board: chess.Board,
    moves: list[str],
    white_options: dict[str, str | int | bool],
    black_options: dict[str, str | int | bool],
    raw_limit: Mapping[str, float | int],
    termination: str,
    illegal_move: dict[str, str] | None = None,
    error: str | None = None,
) -> LadderGameRecord:
    return LadderGameRecord(
        game_index=scheduled.game_index,
        pair_index=scheduled.pair_index,
        lower_elo=scheduled.lower_elo,
        higher_elo=scheduled.higher_elo,
        white_elo=scheduled.white_elo,
        black_elo=scheduled.black_elo,
        status="failed",
        result="*",
        winner=None,
        winner_elo=None,
        higher_elo_score=None,
        termination=termination,
        ply_count=len(moves),
        final_fen=board.fen(),
        moves=list(moves),
        white_options=dict(white_options),
        black_options=dict(black_options),
        stockfish_limit=dict(raw_limit),
        illegal_move=illegal_move,
        error=error,
    )


def _winner_elo(winner: chess.Color | None, white_elo: int, black_elo: int) -> int | None:
    if winner is None:
        return None
    return white_elo if winner == chess.WHITE else black_elo


def _higher_elo_score(winner_elo: int | None, higher_elo: int) -> float:
    if winner_elo is None:
        return 0.5
    return 1.0 if winner_elo == higher_elo else 0.0


def _color_name(color: chess.Color | None) -> str | None:
    if color is None:
        return None
    return "white" if color == chess.WHITE else "black"


if __name__ == "__main__":
    raise SystemExit(main())
