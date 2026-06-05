import io
import textwrap

import chess

from mcchess.board import move_to_index
from mcchess.data import game_passes_filters, iter_samples, new_counters


SCHOLARS_MATE = textwrap.dedent("""\
    [Event "Scholars"]
    [Result "1-0"]

    1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
    """)

SHORT_DRAW = textwrap.dedent("""\
    [Event "Draw"]
    [Result "1/2-1/2"]

    1. e4 e5 2. Nf3 Nf6 1/2-1/2
    """)

UNKNOWN = textwrap.dedent("""\
    [Event "Unknown"]
    [Result "*"]

    1. e4 e5 *
    """)

CORRUPT = textwrap.dedent("""\
    [Event "Corrupt"]
    [Result "1-0"]

    1. Ke2 e5 1-0
    """)

RATED_2000 = textwrap.dedent("""\
    [Event "Rated Blitz game"]
    [WhiteElo "2100"]
    [BlackElo "2050"]
    [Result "1-0"]

    1. e4 e5 1-0
    """)

RATED_1900 = textwrap.dedent("""\
    [Event "Rated Blitz game"]
    [WhiteElo "2100"]
    [BlackElo "1900"]
    [Result "1-0"]

    1. e4 e5 1-0
    """)


def read(pgn):
    counters = new_counters()
    samples = list(iter_samples(io.StringIO(pgn), counters))
    return samples, counters


def read_filtered(pgn, filters):
    counters = new_counters()
    samples = list(iter_samples(io.StringIO(pgn), counters, filters=filters))
    return samples, counters


def test_scholars_mate_emits_expected_samples():
    samples, c = read(SCHOLARS_MATE)
    assert c["games_used"] == 1
    assert c["positions_emitted"] == 7
    assert [s["move_uci"] for s in samples] == [
        "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7",
    ]
    assert [s["ply"] for s in samples] == list(range(7))
    assert all(s["game_id"] == "g000000" for s in samples)
    assert all(s["result"] == "1-0" for s in samples)


def test_value_perspective_white_wins():
    samples, _ = read(SCHOLARS_MATE)
    assert [s["value"] for s in samples] == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]


def test_value_perspective_draw():
    samples, _ = read(SHORT_DRAW)
    assert len(samples) == 4
    assert all(s["value"] == 0.0 for s in samples)


def test_unknown_result_is_skipped_and_counted():
    samples, c = read(UNKNOWN)
    assert samples == []
    assert c["games_read"] == 1
    assert c["games_used"] == 0
    assert c["games_skipped_unknown_result"] == 1
    assert c["games_skipped_corrupt"] == 0


def test_corrupt_is_skipped_and_reader_continues():
    samples, c = read(CORRUPT + "\n" + SHORT_DRAW)
    assert c["games_read"] == 2
    assert c["games_used"] == 1
    assert c["games_skipped_corrupt"] == 1
    assert all(s["game_id"] == "g000001" for s in samples)
    assert len(samples) == 4


def test_policy_index_and_fen_match_move_to_index():
    samples, _ = read(SCHOLARS_MATE)
    board = chess.Board()
    for s in samples:
        assert s["fen"] == board.fen()
        assert s["policy_index"] == move_to_index(board, chess.Move.from_uci(s["move_uci"]))
        board.push_uci(s["move_uci"])


def test_min_elo_filter_requires_both_players_by_default():
    samples, c = read_filtered(RATED_2000 + "\n" + RATED_1900, {"min_elo": 2000})

    assert c["games_read"] == 2
    assert c["games_used"] == 1
    assert c["games_skipped_filter"] == 1
    assert len(samples) == 2


def test_game_passes_filters_supports_either_player_mode():
    headers = {"Event": "Rated Blitz game", "WhiteElo": "2100", "BlackElo": "1900"}

    assert game_passes_filters(headers, {"min_elo": 2000, "min_elo_mode": "either"})
    assert not game_passes_filters(headers, {"min_elo": 2000, "min_elo_mode": "both"})


def test_require_rated_filter_uses_event_header():
    assert game_passes_filters({"Event": "Rated Blitz game"}, {"require_rated": True})
    assert not game_passes_filters({"Event": "Casual Blitz game"}, {"require_rated": True})
