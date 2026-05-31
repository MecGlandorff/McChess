"""
Stream a PGN file into supervised samples, one per played move:

    {game_id, ply, fen, move_uci, policy_index, value, result}

`value` is the final game result from the side-to-move perspective at
that ply (+1 win / 0 draw / -1 loss). Games with unknown result ("*")
or any parse/legality error are skipped; `counters` is a plain dict
updated in place so the caller can write a manifest.
"""
from __future__ import annotations

from collections.abc import Iterator
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
    positions_emitted: int


def iter_samples(stream: TextIO, counters: PgnCounters) -> Iterator[PgnSample]:
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
        "positions_emitted": 0,
    }
