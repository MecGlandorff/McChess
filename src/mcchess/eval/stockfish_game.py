"""Single-game execution for external Stockfish benchmarks."""

from __future__ import annotations

import time

import chess

from mcchess.bots import Bot
from mcchess.eval.stockfish_types import (
    MoveCallback,
    ScheduledStockfishGame,
    StockfishGameRecord,
)
from mcchess.eval.stockfish_utils import UciEngine
from mcchess.eval.stockfish_utils import engine_limit as stockfish_engine_limit


def play_stockfish_game(
    agent: Bot,
    engine: UciEngine,
    scheduled: ScheduledStockfishGame,
    *,
    max_ply: int,
    move_delay_seconds: float = 0.0,
    move_callback: MoveCallback | None = None,
) -> StockfishGameRecord:
    """Play one McChess-vs-Stockfish game and return a serializable record."""

    board = chess.Board(scheduled.starting_fen)
    stockfish_name = scheduled.level.name
    white_name = agent.name if scheduled.mcchess_color == chess.WHITE else stockfish_name
    black_name = agent.name if scheduled.mcchess_color == chess.BLACK else stockfish_name
    moves: list[str] = []

    try:
        engine.configure(scheduled.level.options)
    except Exception as exc:  # noqa: BLE001 - record external engine failures in artifacts.
        return _failed_record(
            scheduled=scheduled,
            mcchess_color=scheduled.mcchess_color,
            white_name=white_name,
            black_name=black_name,
            board=board,
            moves=moves,
            termination="engine_config_error",
            error=f"{type(exc).__name__}: {exc}",
        )

    limit = stockfish_engine_limit(scheduled.level.limit)

    while len(moves) < max_ply:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return _completed_record(
                scheduled=scheduled,
                mcchess_color=scheduled.mcchess_color,
                white_name=white_name,
                black_name=black_name,
                board=board,
                moves=moves,
                outcome=outcome,
                termination=outcome.termination.name.lower(),
            )

        color_name = _required_color_name(board.turn)
        bot_name = agent.name if board.turn == scheduled.mcchess_color else stockfish_name
        try:
            if board.turn == scheduled.mcchess_color:
                move = agent.choose_move(board.copy(stack=True))
            else:
                play_result = engine.play(board.copy(stack=True), limit)
                raw_move = getattr(play_result, "move", None)
                if not isinstance(raw_move, chess.Move):
                    raise ValueError("Stockfish returned no move")
                move = raw_move
        except Exception as exc:  # noqa: BLE001 - record participant failures in artifacts.
            return _failed_record(
                scheduled=scheduled,
                mcchess_color=scheduled.mcchess_color,
                white_name=white_name,
                black_name=black_name,
                board=board,
                moves=moves,
                termination="bot_error",
                error=f"{type(exc).__name__}: {exc}",
            )

        if move not in board.legal_moves:
            return _failed_record(
                scheduled=scheduled,
                mcchess_color=scheduled.mcchess_color,
                white_name=white_name,
                black_name=black_name,
                board=board,
                moves=moves,
                termination="illegal_move",
                illegal_move={
                    "bot": bot_name,
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
                    "game_index": scheduled.game_index,
                    "ply": len(moves),
                    "color": color_name,
                    "bot": bot_name,
                    "uci": move.uci(),
                    "san": san,
                    "fen": board.fen(),
                    "level": scheduled.level.name,
                    "opening_index": scheduled.opening_index,
                }
            )
        if move_delay_seconds > 0.0:
            time.sleep(move_delay_seconds)

    return _completed_record(
        scheduled=scheduled,
        mcchess_color=scheduled.mcchess_color,
        white_name=white_name,
        black_name=black_name,
        board=board,
        moves=moves,
        outcome=None,
        termination="max_ply",
    )


def _completed_record(
    *,
    scheduled: ScheduledStockfishGame,
    mcchess_color: chess.Color,
    white_name: str,
    black_name: str,
    board: chess.Board,
    moves: list[str],
    outcome: chess.Outcome | None,
    termination: str,
) -> StockfishGameRecord:
    winner = outcome.winner if outcome is not None else None
    result = outcome.result() if outcome is not None else "1/2-1/2"
    winner_name = _winner_name(winner, white_name=white_name, black_name=black_name)
    return StockfishGameRecord(
        game_index=scheduled.game_index,
        level=scheduled.level.name,
        level_game_index=scheduled.level_game_index,
        stockfish_elo=scheduled.level.stockfish_elo,
        include_in_elo=scheduled.level.include_in_elo,
        status="completed",
        mcchess_color=_required_color_name(mcchess_color),
        white=white_name,
        black=black_name,
        result=result,
        winner=_color_name(winner),
        winner_name=winner_name,
        mcchess_score=_mcchess_score(winner, mcchess_color),
        termination=termination,
        ply_count=len(moves),
        opening_index=scheduled.opening_index,
        starting_fen=scheduled.starting_fen,
        final_fen=board.fen(),
        moves=list(moves),
        stockfish_options=dict(scheduled.level.options),
        stockfish_limit=dict(scheduled.level.limit),
    )


def _failed_record(
    *,
    scheduled: ScheduledStockfishGame,
    mcchess_color: chess.Color,
    white_name: str,
    black_name: str,
    board: chess.Board,
    moves: list[str],
    termination: str,
    illegal_move: dict[str, str] | None = None,
    error: str | None = None,
) -> StockfishGameRecord:
    return StockfishGameRecord(
        game_index=scheduled.game_index,
        level=scheduled.level.name,
        level_game_index=scheduled.level_game_index,
        stockfish_elo=scheduled.level.stockfish_elo,
        include_in_elo=scheduled.level.include_in_elo,
        status="failed",
        mcchess_color=_required_color_name(mcchess_color),
        white=white_name,
        black=black_name,
        result="*",
        winner=None,
        winner_name=None,
        mcchess_score=None,
        termination=termination,
        ply_count=len(moves),
        opening_index=scheduled.opening_index,
        starting_fen=scheduled.starting_fen,
        final_fen=board.fen(),
        moves=list(moves),
        stockfish_options=dict(scheduled.level.options),
        stockfish_limit=dict(scheduled.level.limit),
        illegal_move=illegal_move,
        error=error,
    )


def _mcchess_score(winner: chess.Color | None, mcchess_color: chess.Color) -> float:
    if winner is None:
        return 0.5
    return 1.0 if winner == mcchess_color else 0.0


def _winner_name(winner: chess.Color | None, *, white_name: str, black_name: str) -> str | None:
    if winner is None:
        return None
    return white_name if winner == chess.WHITE else black_name


def _color_name(color: chess.Color | None) -> str | None:
    if color is None:
        return None
    return "white" if color == chess.WHITE else "black"


def _required_color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"
