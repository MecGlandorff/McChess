"""
Stream a PGN file into supervised samples, one per played move:

    {game_id, ply, fen, move_uci, policy_index, value, result}

`value` is the final game result from the side-to-move perspective at
that ply (+1 win / 0 draw / -1 loss). Games with unknown result ("*")
or any parse/legality error are skipped; `counters` is a plain dict
updated in place so the caller can write a manifest.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TextIO, TypedDict

import chess.pgn

from mcchess.board import move_to_index

_RESULT_VALUE = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


class PgnSample(TypedDict):
    game_id: str
    ply: int
    fen: str
    move_uci: str
    policy_index: int
    value: float
    result: str


class PgnCounters(TypedDict):
    games_read: int
    games_used: int
    games_skipped_corrupt: int
    games_skipped_unknown_result: int
    games_skipped_filter: int
    positions_emitted: int


def iter_samples(
    stream: TextIO,
    counters: PgnCounters,
    filters: Mapping[str, object] | None = None,
) -> Iterator[PgnSample]:
    game_index = -1
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        game_index += 1
        counters["games_read"] += 1

        result = game.headers.get("Result", "*")
        if result not in _RESULT_VALUE:
            counters["games_skipped_unknown_result"] += 1
            continue
        if game.errors:
            counters["games_skipped_corrupt"] += 1
            continue
        if not game_passes_filters(game.headers, filters):
            counters["games_skipped_filter"] += 1
            continue

        white_value = _RESULT_VALUE[result]
        game_id = f"g{game_index:06d}"
        board = game.board()
        buf: list[PgnSample] = []
        try:
            for ply, move in enumerate(game.mainline_moves()):
                buf.append(
                    {
                        "game_id": game_id,
                        "ply": ply,
                        "fen": board.fen(),
                        "move_uci": move.uci(),
                        "policy_index": move_to_index(board, move),
                        "value": white_value if board.turn else -white_value,
                        "result": result,
                    }
                )
                board.push(move)
        except ValueError:
            counters["games_skipped_corrupt"] += 1
            continue

        counters["games_used"] += 1
        counters["positions_emitted"] += len(buf)
        yield from buf


def new_counters() -> PgnCounters:
    return {
        "games_read": 0,
        "games_used": 0,
        "games_skipped_corrupt": 0,
        "games_skipped_unknown_result": 0,
        "games_skipped_filter": 0,
        "positions_emitted": 0,
    }


def game_passes_filters(
    headers: Mapping[str, str],
    filters: Mapping[str, object] | None = None,
) -> bool:
    """Return whether PGN headers pass supported dataset filters.

    Unknown filter keys are ignored so provenance-only manifest fields can
    coexist with executable filters. Supported keys:

    - `min_elo`: integer threshold
    - `min_elo_mode`: `both` (default), `either`, `white`, `black`, or `average`
    - `require_rated`: boolean, checked from Lichess-style Event/Rated headers
    """

    if not filters:
        return True

    if bool(filters.get("require_rated", False)) and not _is_rated(headers):
        return False

    min_elo_raw = filters.get("min_elo")
    if min_elo_raw is None:
        return True

    if isinstance(min_elo_raw, bool) or not isinstance(min_elo_raw, int):
        raise ValueError("min_elo must be an integer")
    min_elo = min_elo_raw
    mode = str(filters.get("min_elo_mode", "both"))
    white_elo = _parse_elo(headers.get("WhiteElo"))
    black_elo = _parse_elo(headers.get("BlackElo"))

    if mode == "both":
        return (
            white_elo is not None
            and black_elo is not None
            and min(white_elo, black_elo) >= min_elo
        )
    if mode == "either":
        return (
            (white_elo is not None and white_elo >= min_elo)
            or (black_elo is not None and black_elo >= min_elo)
        )
    if mode == "white":
        return white_elo is not None and white_elo >= min_elo
    if mode == "black":
        return black_elo is not None and black_elo >= min_elo
    if mode == "average":
        return (
            white_elo is not None
            and black_elo is not None
            and (white_elo + black_elo) / 2.0 >= min_elo
        )
    raise ValueError(f"unsupported min_elo_mode: {mode}")


def _parse_elo(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _is_rated(headers: Mapping[str, str]) -> bool:
    rated = headers.get("Rated")
    if rated is not None:
        return rated.lower() in {"1", "true", "yes"}
    return headers.get("Event", "").lower().startswith("rated ")
