from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import chess.pgn


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "filter_pgn.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("filter_pgn", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_filter_pgn_files_keeps_2000_plus_games(tmp_path: Path) -> None:
    script = load_script_module()
    source = tmp_path / "input.pgn"
    output = tmp_path / "filtered.pgn"
    manifest = tmp_path / "manifest.json"
    source.write_text(
        textwrap.dedent(
            """\
            [Event "Rated Blitz game"]
            [WhiteElo "2100"]
            [BlackElo "2050"]
            [Result "1-0"]

            1. e4 e5 1-0

            [Event "Rated Blitz game"]
            [WhiteElo "2100"]
            [BlackElo "1900"]
            [Result "1-0"]

            1. d4 d5 1-0
            """
        ),
        encoding="utf-8",
    )

    manifest_path = script.filter_pgn_files(
        [source],
        output,
        manifest,
        script.PgnFilterConfig(min_elo=2000, min_elo_mode="both", require_rated=True),
        show_progress=False,
    )

    assert manifest_path == manifest
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["games_read"] == 2
    assert data["games_written"] == 1
    assert data["games_skipped_filter"] == 1

    with output.open(encoding="utf-8") as file:
        game = chess.pgn.read_game(file)
        assert game is not None
        assert game.headers["WhiteElo"] == "2100"
        assert game.headers["BlackElo"] == "2050"
        assert chess.pgn.read_game(file) is None


def test_filter_pgn_files_respects_max_kept_games(tmp_path: Path) -> None:
    script = load_script_module()
    source = tmp_path / "input.pgn"
    output = tmp_path / "filtered.pgn"
    manifest = tmp_path / "manifest.json"
    source.write_text(
        textwrap.dedent(
            """\
            [Event "Rated Blitz game"]
            [WhiteElo "2100"]
            [BlackElo "2050"]
            [Result "1-0"]

            1. e4 e5 1-0

            [Event "Rated Blitz game"]
            [WhiteElo "2200"]
            [BlackElo "2150"]
            [Result "0-1"]

            1. d4 d5 0-1

            [Event "Rated Blitz game"]
            [WhiteElo "2300"]
            [BlackElo "2250"]
            [Result "1/2-1/2"]

            1. c4 c5 1/2-1/2
            """
        ),
        encoding="utf-8",
    )

    script.filter_pgn_files(
        [source],
        output,
        manifest,
        script.PgnFilterConfig(
            min_elo=2000,
            min_elo_mode="both",
            require_rated=True,
            max_kept_games=2,
        ),
        show_progress=False,
    )

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["games_read"] == 2
    assert data["games_written"] == 2
    assert data["stopped_early"] is True
    assert data["filters"]["max_kept_games"] == 2


def test_filter_pgn_files_preserves_matching_movetext_without_validating(
    tmp_path: Path,
) -> None:
    script = load_script_module()
    source = tmp_path / "input.pgn"
    output = tmp_path / "filtered.pgn"
    manifest = tmp_path / "manifest.json"
    source.write_text(
        textwrap.dedent(
            """\
            [Event "Rated Blitz game"]
            [WhiteElo "2100"]
            [BlackElo "2050"]
            [Result "1-0"]

            1. definitely-not-a-legal-move 1-0
            """
        ),
        encoding="utf-8",
    )

    script.filter_pgn_files(
        [source],
        output,
        manifest,
        script.PgnFilterConfig(min_elo=2000, min_elo_mode="both", require_rated=True),
        show_progress=False,
    )

    assert "definitely-not-a-legal-move" in output.read_text(encoding="utf-8")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["games_written"] == 1
    assert data["games_skipped_corrupt"] == 0


def test_parse_header_line_unescapes_pgn_strings() -> None:
    script = load_script_module()

    assert script.parse_header_line('[Event "Rated \\"Blitz\\" game"]') == (
        "Event",
        'Rated "Blitz" game',
    )
    assert script.parse_header_line("[Event Rated Blitz game]") is None
