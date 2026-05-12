import io
import textwrap

import chess

from mcchess.board import move_to_index
from mcchess.data import PgnSample, ReaderCounters, iter_pgn_games, iter_samples


SCHOLARS_MATE_PGN = textwrap.dedent(
    """\
    [Event "Scholars"]
    [Result "1-0"]

    1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
    """
)

SHORT_DRAW_PGN = textwrap.dedent(
    """\
    [Event "Short draw"]
    [Result "1/2-1/2"]

    1. e4 e5 2. Nf3 Nf6 1/2-1/2
    """
)

UNKNOWN_RESULT_PGN = textwrap.dedent(
    """\
    [Event "Unknown"]
    [Result "*"]

    1. e4 e5 *
    """
)

CORRUPT_GAME_PGN = textwrap.dedent(
    """\
    [Event "Corrupt"]
    [Result "1-0"]

    1. Ke2 e5 1-0
    """
)


def _read_samples(pgn: str) -> tuple[list[PgnSample], ReaderCounters]:
    counters = ReaderCounters()
    samples = list(iter_samples(iter_pgn_games(io.StringIO(pgn)), counters=counters))
    return samples, counters


def test_tiny_pgn_emits_expected_samples() -> None:
    samples, counters = _read_samples(SCHOLARS_MATE_PGN)

    assert counters.games_read == 1
    assert counters.games_used == 1
    assert counters.positions_emitted == 7
    assert [s.move_uci for s in samples] == [
        "e2e4",
        "e7e5",
        "f1c4",
        "b8c6",
        "d1h5",
        "g8f6",
        "h5f7",
    ]
    assert all(s.game_id == "g000000" for s in samples)
    assert [s.ply for s in samples] == list(range(7))
    assert all(s.result == "1-0" for s in samples)


def test_value_perspective_white_wins() -> None:
    samples, _ = _read_samples(SCHOLARS_MATE_PGN)

    expected = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    assert [s.value for s in samples] == expected


def test_value_perspective_draw() -> None:
    samples, _ = _read_samples(SHORT_DRAW_PGN)

    assert len(samples) == 4
    assert all(s.value == 0.0 for s in samples)
    assert all(s.result == "1/2-1/2" for s in samples)


def test_unknown_result_is_skipped_and_counted() -> None:
    samples, counters = _read_samples(UNKNOWN_RESULT_PGN)

    assert samples == []
    assert counters.games_read == 1
    assert counters.games_used == 0
    assert counters.games_skipped_unknown_result == 1
    assert counters.games_skipped_corrupt == 0


def test_corrupt_game_is_skipped_and_counted_and_reader_continues() -> None:
    mixed = CORRUPT_GAME_PGN + "\n" + SHORT_DRAW_PGN
    samples, counters = _read_samples(mixed)

    assert counters.games_read == 2
    assert counters.games_used == 1
    assert counters.games_skipped_corrupt == 1
    assert all(s.game_id == "g000001" for s in samples)
    assert len(samples) == 4


def test_policy_index_matches_move_to_index() -> None:
    samples, _ = _read_samples(SCHOLARS_MATE_PGN)
    board = chess.Board()

    for sample in samples:
        expected_index = move_to_index(board, chess.Move.from_uci(sample.move_uci))
        assert sample.policy_index == expected_index
        assert sample.fen == board.fen()
        board.push_uci(sample.move_uci)


def test_fen_is_position_before_move() -> None:
    samples, _ = _read_samples(SCHOLARS_MATE_PGN)

    assert samples[0].fen == chess.Board().fen()
    assert samples[0].move_uci == "e2e4"
