"""Opening-position helpers for evaluation schedules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import chess

STANDARD_OPENING_PROTOCOL: Final[str] = "standard_initial_position"
PAIRED_FEN_OPENING_PROTOCOL: Final[str] = "paired_fixed_fens"


def normalize_opening_fens(raw_fens: Sequence[str] | None) -> tuple[str, ...]:
    """Validate optional fixed opening FENs from a config."""

    if raw_fens is None:
        return ()
    if isinstance(raw_fens, str):
        raise ValueError("opening_fens must be a YAML list, not a string")

    fens = []
    for index, raw_fen in enumerate(raw_fens):
        if not isinstance(raw_fen, str) or not raw_fen.strip():
            raise ValueError(f"opening_fens[{index}] must be a non-empty FEN string")
        fen = raw_fen.strip()
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"opening_fens[{index}] is not valid FEN: {fen}") from exc
        if not board.is_valid():
            raise ValueError(f"opening_fens[{index}] is not a valid chess position: {fen}")
        fens.append(board.fen())
    return tuple(fens)


def opening_protocol(opening_fens: Sequence[str]) -> str:
    """Return the protocol label for a configured opening set."""

    return PAIRED_FEN_OPENING_PROTOCOL if opening_fens else STANDARD_OPENING_PROTOCOL


def starting_fen_for_game(opening_fens: Sequence[str], game_index: int) -> tuple[int | None, str]:
    """Return the paired opening FEN for a game index.

    Adjacent games share the same opening so color alternation gives a paired
    comparison when the caller alternates colors globally.
    """

    if not opening_fens:
        return None, chess.STARTING_FEN
    opening_index = (game_index // 2) % len(opening_fens)
    return opening_index, opening_fens[opening_index]
