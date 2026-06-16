from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import chess
import pytest
import yaml  # type: ignore[import-untyped]

from mcchess.eval.arena import ArenaConfig, BotConfig, play_game, run_match

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_arena.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("run_arena", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScriptedBot:
    def __init__(self, name: str, moves: list[str]) -> None:
        self.name = name
        self.moves = [chess.Move.from_uci(move) for move in moves]

    def choose_move(self, board: chess.Board) -> chess.Move:
        move = self.moves.pop(0)
        assert move in board.legal_moves
        return move


class IllegalBot:
    name = "illegal"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return chess.Move.from_uci("e2e5")


def tiny_config(tmp_path: Path, *, seed: int = 7) -> ArenaConfig:
    return ArenaConfig(
        run_id="arena_test",
        output_path=str(tmp_path / "arena.json"),
        seed=seed,
        num_games=4,
        max_ply=6,
        agent=BotConfig(kind="material"),
        opponent=BotConfig(kind="random"),
    )


def test_play_game_records_checkmate_from_agent_perspective() -> None:
    agent = ScriptedBot("agent", ["e7e5", "d8h4"])
    opponent = ScriptedBot("opponent", ["f2f3", "g2g4"])

    game = play_game(
        agent,
        opponent,
        game_index=0,
        agent_color=chess.BLACK,
        max_ply=10,
    )

    assert game.status == "completed"
    assert game.result == "0-1"
    assert game.winner == "black"
    assert game.agent_score == 1.0
    assert game.termination == "checkmate"
    assert game.moves == ["f2f3", "e7e5", "g2g4", "d8h4"]


def test_play_game_reports_move_events() -> None:
    events: list[dict[str, object]] = []

    game = play_game(
        ScriptedBot("agent", ["e2e4"]),
        ScriptedBot("opponent", []),
        game_index=0,
        agent_color=chess.WHITE,
        max_ply=1,
        move_callback=events.append,
    )

    assert game.termination == "max_ply"
    assert events[0]["game_index"] == 0
    assert events[0]["ply"] == 1
    assert events[0]["color"] == "white"
    assert events[0]["bot"] == "agent"
    assert events[0]["uci"] == "e2e4"
    assert events[0]["san"] == "e4"


def test_run_match_alternates_colors_and_counts_draws(tmp_path: Path) -> None:
    result = run_match(
        ArenaConfig(
            run_id="arena_test",
            output_path=str(tmp_path / "arena.json"),
            seed=0,
            num_games=2,
            max_ply=2,
            agent=BotConfig(kind="material"),
            opponent=BotConfig(kind="random"),
        )
    )

    assert result["status"] == "completed"
    assert result["games_completed"] == 2
    assert result["draws"] == 2
    assert result["score"] == 0.5
    assert result["illegal_moves"] == 0
    assert [game["agent_color"] for game in result["games"]] == ["white", "black"]
    assert {game["termination"] for game in result["games"]} == {"max_ply"}


def test_run_match_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = run_match(tiny_config(tmp_path, seed=123))
    second = run_match(tiny_config(tmp_path, seed=123))

    assert [game["moves"] for game in first["games"]] == [
        game["moves"] for game in second["games"]
    ]
    assert first["wins"] == second["wins"]
    assert first["draws"] == second["draws"]
    assert first["losses"] == second["losses"]


def test_illegal_bot_move_fails_the_match(tmp_path: Path) -> None:
    game = play_game(
        IllegalBot(),
        ScriptedBot("opponent", []),
        game_index=0,
        agent_color=chess.WHITE,
        max_ply=10,
    )

    assert game.status == "failed"
    assert game.termination == "illegal_move"
    assert game.illegal_move == {"bot": "illegal", "color": "white", "move": "e2e5"}

    result = run_match(
        ArenaConfig(
            run_id="arena_test",
            output_path=str(tmp_path / "arena.json"),
            seed=0,
            num_games=1,
            max_ply=10,
            agent=BotConfig(kind="material"),
            opponent=BotConfig(kind="random"),
        )
    )
    assert result["illegal_moves"] == 0


def test_bot_config_rejects_policy_only_without_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint_path"):
        BotConfig(kind="policy_only")


def test_arena_config_rejects_negative_move_delay(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="move_delay_seconds"):
        ArenaConfig(
            output_path=str(tmp_path / "arena.json"),
            move_delay_seconds=-1.0,
            agent=BotConfig(kind="material"),
            opponent=BotConfig(kind="random"),
        )


def test_run_arena_script_writes_result_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script_module()
    config_path = tmp_path / "arena.yaml"
    output_path = tmp_path / "arena.json"
    config = {
        "run_id": "arena_script_test",
        "output_path": str(output_path),
        "seed": 3,
        "num_games": 2,
        "max_ply": 2,
        "agent": {"kind": "material"},
        "opponent": {"kind": "random"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(script, "git_commit", lambda: "abc123")

    result_path = script.run_arena(config_path)

    assert result_path == output_path
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["run_id"] == "arena_script_test"
    assert result["config_path"] == str(config_path)
    assert result["git_commit"] == "abc123"
    assert result["config"]["agent"]["kind"] == "material"
    assert len(result["games"]) == 2
