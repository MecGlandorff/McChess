"""Streaming PGN reader that emits supervised `(board, policy, value)` samples.

This module is the input side of the dataset builder. It is intentionally
stream-oriented so that callers can build datasets without holding every
position in memory.

Output schema matches `docs/DATASET_PROTOCOL.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Iterable, Iterator, TextIO

import chess
import chess.pgn

from mcchess.board import move_to_index

_RESULT_TO_WHITE_VALUE: Final[dict[str, float]] = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0,
}


@dataclass
class PgnSample:
    """One supervised sample: a position with its played move and result."""

    game_id: str
    ply: int
    fen: str
    move_uci: str
    policy_index: int
    value: float
    result: str
    split: str = ""


@dataclass
class ReaderCounters:
    games_read: int = 0
    games_used: int = 0
    games_skipped_corrupt: int = 0
    games_skipped_unknown_result: int = 0
    positions_emitted: int = 0


@dataclass
class _GameAttempt:
    samples: list[PgnSample] = field(default_factory=list)
    failed: bool = False


def iter_pgn_games(stream: TextIO) -> Iterator[chess.pgn.Game]:
    """Yield games from a PGN text stream until EOF.

    Header-level parse failures advance to the next game rather than crashing
    the whole iteration; move-level legality is enforced later in
    `iter_samples`.
    """

    while True:
        try:
            game = chess.pgn.read_game(stream)
        except (ValueError, RuntimeError):
            continue
        if game is None:
            return
        yield game


def iter_samples(
    games: Iterable[chess.pgn.Game],
    *,
    game_id_prefix: str = "g",
    counters: ReaderCounters | None = None,
) -> Iterator[PgnSample]:
    """Yield `PgnSample`s for every legal move in every usable game.

    Games with `Result "*"` or any parse/legality error are skipped without
    emitting partial samples; the optional `counters` is updated in place so
    callers can build a dataset manifest.
    """

    if counters is None:
        counters = ReaderCounters()

    for game_index, game in enumerate(games):
        counters.games_read += 1
        game_id = f"{game_id_prefix}{game_index:06d}"

        result = game.headers.get("Result", "*")
        if result not in _RESULT_TO_WHITE_VALUE:
            counters.games_skipped_unknown_result += 1
            continue

        attempt = _try_collect_samples(game, game_id=game_id, result=result)
        if attempt.failed:
            counters.games_skipped_corrupt += 1
            continue

        counters.games_used += 1
        counters.positions_emitted += len(attempt.samples)
        yield from attempt.samples


def _try_collect_samples(
    game: chess.pgn.Game, *, game_id: str, result: str
) -> _GameAttempt:
    attempt = _GameAttempt()

    # python-chess records illegal-SAN and parse errors here without raising;
    # a non-empty list means the mainline cannot be trusted.
    if game.errors:
        attempt.failed = True
        return attempt

    white_value = _RESULT_TO_WHITE_VALUE[result]
    board = game.board()

    try:
        for ply, move in enumerate(game.mainline_moves()):
            policy_index = move_to_index(board, move)
            value = white_value if board.turn == chess.WHITE else -white_value
            attempt.samples.append(
                PgnSample(
                    game_id=game_id,
                    ply=ply,
                    fen=board.fen(),
                    move_uci=move.uci(),
                    policy_index=policy_index,
                    value=value,
                    result=result,
                )
            )
            board.push(move)
    except (ValueError, AssertionError):
        attempt.failed = True
        attempt.samples.clear()

    return attempt
