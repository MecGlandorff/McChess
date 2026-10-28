from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chess
import pytest

from mcchess.eval.stockfish import iter_scheduled_games, load_config
from mcchess.eval.stockfish_rating import estimate_mcchess_elo
from mcchess.eval.stockfish_types import StockfishGameRecord

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models_archive/manifest.json"
BENCHMARK_KEYS = [
    "external_stockfish_benchmark",
    "external_stockfish_wide_benchmark",
    "external_stockfish_high_probe",
]


def _published_result(entry_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = manifest["evaluation"][entry_name]
    raw = (ROOT / entry["result_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["result_sha256"]
    return entry, json.loads(raw)


@pytest.fixture(scope="module")
def published_benchmark() -> dict[str, Any]:
    return _published_result("external_stockfish_benchmark")[1]


def test_published_benchmark_accounting(published_benchmark: dict[str, Any]) -> None:
    games = published_benchmark["games"]
    summary = published_benchmark["summary"]
    assert published_benchmark["run"]["status"] == "completed"
    assert len(games) == summary["games_completed"] == summary["games_scheduled"] == 202
    assert [game["game_index"] for game in games] == list(range(len(games)))
    assert all(game["status"] == "completed" and game["error"] is None for game in games)
    assert sum(bool(game["illegal_move"]) for game in games) == summary["illegal_moves"] == 0
    counts = Counter(game["mcchess_score"] for game in games)
    assert counts == {1.0: summary["wins"], 0.5: summary["draws"], 0.0: summary["losses"]}
    assert sum(game["mcchess_score"] for game in games) / len(games) == pytest.approx(
        summary["score"]
    )

    included = [game for game in games if game["include_in_elo"]]
    excluded = [game for game in games if not game["include_in_elo"]]
    estimate = published_benchmark["metrics"]["elo_estimate"]
    assert len(included) == estimate["included_games"] == 200
    assert len(excluded) == 2
    assert all(game["stockfish_elo"] is None and game["mcchess_score"] == 0 for game in excluded)
    assert Counter(game["stockfish_elo"] for game in included) == {
        level: 20 for level in range(1600, 2501, 100)
    }
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = manifest["evaluation"]["external_stockfish_benchmark"]
    assert Counter(game["mcchess_score"] for game in included) == {
        1.0: entry["included_wins"],
        0.5: entry["included_draws"],
        0.0: entry["included_losses"],
    }
    assert sum(game["mcchess_score"] for game in included) / len(included) == pytest.approx(
        entry["included_score"]
    )
    assert estimate["score"] == entry["included_score"]
    publication = published_benchmark["publication"]
    assert publication["source_checkpoint_sha256"] == manifest["source"]["checkpoint_sha256"]
    assert publication["model_artifact_sha256"] == manifest["artifact"]["sha256"]
    assert (ROOT / publication["model_artifact"]).is_file()
    assert (ROOT / entry["report_path"]).is_file()


@pytest.mark.parametrize("entry_name", BENCHMARK_KEYS)
def test_published_game_records_replay_legally(entry_name: str) -> None:
    _, published_benchmark = _published_result(entry_name)
    for game in published_benchmark["games"]:
        board = chess.Board(game["starting_fen"])
        for move_uci in game["moves"]:
            assert board.outcome(claim_draw=True) is None, game["game_index"]
            move = chess.Move.from_uci(move_uci)
            assert move in board.legal_moves, (game["game_index"], move_uci)
            board.push(move)
        assert len(game["moves"]) == game["ply_count"]
        assert board.fen() == game["final_fen"], game["game_index"]
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            assert game["ply_count"] == published_benchmark["protocol"]["max_ply"]
            assert game["result"] == "1/2-1/2"
            assert game["termination"] == "max_ply"
        else:
            assert outcome.result() == game["result"], game["game_index"]
            assert game["termination"] == outcome.termination.name.lower()
        expected_score = 0.5
        if game["result"] != "1/2-1/2":
            winner = "white" if game["result"] == "1-0" else "black"
            expected_score = float(winner == game["mcchess_color"])
        assert game["mcchess_score"] == expected_score, game["game_index"]


@pytest.mark.parametrize("entry_name", BENCHMARK_KEYS[1:])
def test_october_publication_matches_schedule_and_provenance(entry_name: str) -> None:
    entry, result = _published_result(entry_name)
    games, summary = result["games"], result["summary"]
    config = load_config(ROOT / entry["config_path"])
    schedule = iter_scheduled_games(config)
    assert result["run"]["status"] == "completed"
    assert len(games) == len(schedule) == summary["games_scheduled"] == entry["games"]
    assert summary["games_completed"] == entry["included_games"] == len(games)
    assert summary["illegal_moves"] == 0 and summary["failure"] is None
    counts = Counter(game["mcchess_score"] for game in games)
    assert counts == {
        1.0: entry["included_wins"], 0.5: entry["included_draws"], 0.0: entry["included_losses"]
    }
    assert [summary[key] for key in ("wins", "draws", "losses")] == [
        counts[1.0], counts[0.5], counts[0.0]
    ]
    assert sum(game["mcchess_score"] for game in games) / len(games) == pytest.approx(
        summary["score"]
    )
    assert summary["score"] == entry["included_score"]
    assert sum(game["termination"] == "max_ply" for game in games) == entry["move_cap_draws"]

    replay_config = json.loads(json.dumps(asdict(config)))
    source_config = dict(result["config"])
    for key in ("run_id", "output_dir", "stockfish_path"):
        replay_config.pop(key)
        source_config.pop(key)
    assert replay_config == source_config
    for expected, game in zip(schedule, games, strict=True):
        assert game["status"] == "completed" and game["include_in_elo"]
        assert game["illegal_move"] is None and game["error"] is None
        assert game["game_index"] == expected.game_index
        assert game["level_game_index"] == expected.level_game_index
        assert game["stockfish_elo"] == expected.level.stockfish_elo
        assert game["stockfish_options"] == expected.level.options
        assert game["stockfish_limit"] == expected.level.limit
        assert game["starting_fen"] == expected.starting_fen
        assert game["opening_index"] == expected.opening_index
        assert game["mcchess_color"] == ("white" if expected.mcchess_color else "black")
    estimate = estimate_mcchess_elo([StockfishGameRecord(**game) for game in games])
    assert asdict(estimate) == result["metrics"]["elo_estimate"]

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    publication, provenance = result["publication"], result["provenance"]
    assert publication["model_artifact_sha256"] == manifest["artifact"]["sha256"]
    assert publication["reproduction_config"] == entry["config_path"]
    assert provenance["dataset"] == manifest["dataset"]
    assert provenance["hardware"]["resolved_device"] == "cuda"
    assert provenance["stockfish_sha256"] == (
        "c86215fa1977d53b82ed854540a4c7b025be4cd042276c85ba3de53fb9118911"
    )
    assert result["run"]["git_commit"] == "119ca8f9c4e923df4d40828f9226cf4009a74e45"
    assert (ROOT / entry["report_path"]).is_file()
    if config.opening_fens:
        openings = provenance["opening_manifest"]["openings"]
        assert len(openings) == len(config.opening_fens) == 10
        for opening, fen in zip(openings, config.opening_fens, strict=True):
            board = chess.Board()
            moves = [board.push_san(san).uci() for san in opening["san"].split()]
            assert moves == opening["moves"]
            assert board.fen() == opening["fen"] == fen
        assert entry["move_cap_draws"] == 0
    else:
        assert entry["move_cap_draws"] == 2
        assert [game["game_index"] for game in games if game["termination"] == "max_ply"] == [8, 10]
