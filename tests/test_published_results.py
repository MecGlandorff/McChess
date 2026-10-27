from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models_archive/manifest.json"


@pytest.fixture(scope="module")
def published_benchmark() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = manifest["evaluation"]["external_stockfish_benchmark"]
    raw = (ROOT / entry["result_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["result_sha256"]
    return json.loads(raw)


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


def test_published_game_records_replay_legally(published_benchmark: dict[str, Any]) -> None:
    for game in published_benchmark["games"]:
        board = chess.Board(game["starting_fen"])
        for move_uci in game["moves"]:
            move = chess.Move.from_uci(move_uci)
            assert move in board.legal_moves, (game["game_index"], move_uci)
            board.push(move)
        assert len(game["moves"]) == game["ply_count"]
        assert board.fen() == game["final_fen"], game["game_index"]
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            assert game["ply_count"] == published_benchmark["protocol"]["max_ply"]
            assert game["result"] == "1/2-1/2"
        else:
            assert outcome.result() == game["result"], game["game_index"]
        expected_score = 0.5
        if game["result"] != "1/2-1/2":
            winner = "white" if game["result"] == "1-0" else "black"
            expected_score = float(winner == game["mcchess_color"])
        assert game["mcchess_score"] == expected_score, game["game_index"]
