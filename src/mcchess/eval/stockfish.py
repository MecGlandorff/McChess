"""CLI facade for the external Stockfish benchmark.

Stockfish is allowed here only as an external evaluation opponent. Moves,
evaluations, and outcomes from this benchmark must not be used as training
labels, distillation targets, or checkpoint-selection targets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess.engine
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from mcchess.eval.arena import build_bot
from mcchess.eval.common import git_commit as current_git_commit
from mcchess.eval.openings import opening_protocol
from mcchess.eval.stockfish_artifacts import (
    GAME_CSV_FIELDS,
    format_markdown_report,
    game_summary_rows,
    write_artifacts,
)
from mcchess.eval.stockfish_protocol import (
    COLOR_POLICY,
    DRAW_RULE,
    SCOPE_NOTE,
    GameCallback,
    MoveCallback,
    ScheduledStockfishGame,
    StockfishEvalConfig,
    StockfishGameRecord,
    StockfishLevelConfig,
    iter_scheduled_games,
    load_config,
    play_stockfish_game,
)
from mcchess.eval.stockfish_rating import (
    EloEstimate,
    estimate_mcchess_elo,
    insufficient_elo_estimate,
)
from mcchess.eval.stockfish_runner import run_stockfish_match as _run_stockfish_match
from mcchess.eval.stockfish_runner import setup_failure_result
from mcchess.eval.stockfish_utils import UciEngine
from mcchess.eval.stockfish_utils import resolve_stockfish_path as resolve_stockfish_binary_path

StockfishEvalViewer: Any = None


def resolve_stockfish_path(config: StockfishEvalConfig, override: str | None = None) -> str:
    """Resolve Stockfish from CLI override, config, environment, or PATH."""

    return resolve_stockfish_binary_path(config.stockfish_path, override)


def run_stockfish_match(
    config: StockfishEvalConfig,
    engine: UciEngine,
    *,
    stockfish_id: dict[str, str] | None = None,
    stockfish_available_options: list[str] | None = None,
    git_commit: str | None = None,
    config_path: str | None = None,
    move_callback: MoveCallback | None = None,
    game_callback: GameCallback | None = None,
) -> dict[str, Any]:
    """Run the benchmark while preserving this module's patchable bot builder."""

    return _run_stockfish_match(
        config,
        engine,
        stockfish_id=stockfish_id,
        stockfish_available_options=stockfish_available_options,
        git_commit=git_commit,
        config_path=config_path,
        move_callback=move_callback,
        game_callback=game_callback,
        bot_builder=build_bot,
    )


def print_move_event(event: dict[str, Any]) -> None:
    """Print one live move event."""

    print(
        f"game {event['game_index'] + 1:03d} "
        f"ply {event['ply']:03d} "
        f"{event['level']} "
        f"{event['color']} {event['bot']}: {event['san']} ({event['uci']})",
        flush=True,
    )


def combine_move_callbacks(*callbacks: MoveCallback | None) -> MoveCallback | None:
    """Return one move callback from optional callback objects."""

    active = [callback for callback in callbacks if callback is not None]
    if not active:
        return None

    def combined(event: dict[str, Any]) -> None:
        for callback in active:
            callback(event)

    return combined


def combine_game_callbacks(*callbacks: Callable[[Any], None] | None) -> Callable[[Any], None] | None:
    """Return one game callback from optional callback objects."""

    active = [callback for callback in callbacks if callback is not None]
    if not active:
        return None

    def combined(game: Any) -> None:
        for callback in active:
            callback(game)

    return combined


def run_stockfish_eval(
    config_path: str | Path,
    *,
    stockfish_path: str | None = None,
    output_dir: str | Path | None = None,
    print_moves: bool | None = None,
    move_delay_seconds: float | None = None,
    show: bool = False,
    show_progress: bool = True,
) -> Path:
    """Run a Stockfish benchmark config and return the result JSON path."""

    config_path = Path(config_path)
    config = load_config(str(config_path))
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    if print_moves is not None:
        config = replace(config, print_moves=print_moves)
    if move_delay_seconds is not None:
        config = replace(config, move_delay_seconds=move_delay_seconds)

    run_git_commit = current_git_commit()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    start_time = time.perf_counter()
    viewer: Any | None = None
    progress: Any | None = None
    engine: chess.engine.SimpleEngine | None = None
    try:
        resolved_stockfish_path = resolve_stockfish_path(config, stockfish_path)
        config = replace(config, stockfish_path=resolved_stockfish_path)
        viewer = _viewer_class()() if show else None
        move_callback = combine_move_callbacks(
            print_move_event if config.print_moves else None,
            viewer.on_move if viewer is not None else None,
        )
        progress = tqdm(
            total=config.num_games,
            desc="stockfish eval",
            unit="game",
            dynamic_ncols=True,
            disable=not show_progress,
        )

        def update_progress(game: StockfishGameRecord) -> None:
            progress.update(1)
            progress.set_postfix(
                level=str(game.level),
                result=str(game.result),
                score="" if game.mcchess_score is None else f"{float(game.mcchess_score):.1f}",
            )

        game_callback = combine_game_callbacks(
            update_progress,
            viewer.on_game if viewer is not None else None,
        )
        engine = chess.engine.SimpleEngine.popen_uci(resolved_stockfish_path)
        result = run_stockfish_match(
            config,
            engine,
            stockfish_id=dict(engine.id),
            stockfish_available_options=sorted(str(name) for name in engine.options),
            git_commit=run_git_commit,
            config_path=str(config_path),
            move_callback=move_callback,
            game_callback=game_callback,
        )
    except Exception as exc:
        result = setup_failure_result(
            config,
            config_path=str(config_path),
            git_commit=run_git_commit,
            started_at=started_at,
            elapsed_seconds=time.perf_counter() - start_time,
            error=exc,
        )
        result_path = write_artifacts(
            output_dir=Path(config.output_dir),
            config_path=config_path,
            config=config,
            result=result,
        )
        print(f"saved failed Stockfish benchmark result to {result_path}")
        if viewer is not None:
            viewer.close()
        raise
    finally:
        if progress is not None:
            progress.close()
        if engine is not None:
            engine.quit()

    result_path = write_artifacts(
        output_dir=Path(config.output_dir),
        config_path=config_path,
        config=config,
        result=result,
    )
    print(f"saved Stockfish benchmark result to {result_path}")
    print(f"saved game table to {Path(config.output_dir) / 'games.csv'}")
    print(f"saved report to {Path(config.output_dir) / 'report.md'}")
    if viewer is not None:
        viewer.mark_complete(str(result_path))
        viewer.wait_until_closed()
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a McChess MCTS-200 external benchmark against Stockfish."
    )
    parser.add_argument("config", type=Path, help="YAML Stockfish benchmark config.")
    parser.add_argument("--stockfish-path", help="Path to the Stockfish executable.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--print-moves", action="store_true", help="Print live moves.")
    parser.add_argument("--show", action="store_true", help="Open live board and results windows.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress.")
    parser.add_argument(
        "--move-delay-seconds",
        type=float,
        help="Delay between live moves for watchable output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_stockfish_eval(
        args.config,
        stockfish_path=args.stockfish_path,
        output_dir=args.output_dir,
        print_moves=True if args.print_moves else None,
        move_delay_seconds=args.move_delay_seconds,
        show=args.show,
        show_progress=not args.no_progress,
    )
    return 0


def _viewer_class() -> Any:
    global StockfishEvalViewer
    if StockfishEvalViewer is None:
        from mcchess.eval.stockfish_gui import StockfishEvalViewer as viewer_class

        StockfishEvalViewer = viewer_class
    return StockfishEvalViewer


__all__ = [
    "COLOR_POLICY",
    "DRAW_RULE",
    "GAME_CSV_FIELDS",
    "SCOPE_NOTE",
    "EloEstimate",
    "ScheduledStockfishGame",
    "StockfishEvalConfig",
    "StockfishGameRecord",
    "StockfishLevelConfig",
    "combine_game_callbacks",
    "combine_move_callbacks",
    "estimate_mcchess_elo",
    "format_markdown_report",
    "game_summary_rows",
    "insufficient_elo_estimate",
    "iter_scheduled_games",
    "load_config",
    "main",
    "opening_protocol",
    "parse_args",
    "play_stockfish_game",
    "print_move_event",
    "resolve_stockfish_path",
    "run_stockfish_eval",
    "run_stockfish_match",
    "write_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
