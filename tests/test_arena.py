from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest
import yaml  # type: ignore[import-untyped]

from mcchess.eval import arena as arena_module
from mcchess.eval.arena import ArenaConfig, BotConfig, play_game, run_match
from mcchess.eval.openings import PAIRED_FEN_OPENING_PROTOCOL
from mcchess.eval.schema import validate_result_envelope


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
        output_dir=str(tmp_path / "arena"),
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
            output_dir=str(tmp_path / "arena"),
            seed=0,
            num_games=2,
            max_ply=2,
            agent=BotConfig(kind="material"),
            opponent=BotConfig(kind="random"),
        )
    )

    assert result["schema_version"] == 2
    assert result["run"]["status"] == "completed"
    assert result["summary"]["games_completed"] == 2
    assert result["summary"]["draws"] == 2
    assert result["summary"]["score"] == 0.5
    assert result["summary"]["illegal_moves"] == 0
    assert [game["agent_color"] for game in result["games"]] == ["white", "black"]
    assert {game["termination"] for game in result["games"]} == {"max_ply"}
    validate_result_envelope(result)


def test_run_match_pairs_configured_opening_fens(tmp_path: Path) -> None:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    opening_fen = board.fen()

    result = run_match(
        ArenaConfig(
            run_id="arena_openings_test",
            output_dir=str(tmp_path / "arena"),
            seed=0,
            num_games=2,
            max_ply=1,
            opening_fens=(opening_fen,),
            agent=BotConfig(kind="material"),
            opponent=BotConfig(kind="random"),
        )
    )

    assert result["protocol"]["opening_protocol"] == PAIRED_FEN_OPENING_PROTOCOL
    assert [game["starting_fen"] for game in result["games"]] == [opening_fen, opening_fen]
    assert [game["opening_index"] for game in result["games"]] == [0, 0]


def test_run_match_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = run_match(tiny_config(tmp_path, seed=123))
    second = run_match(tiny_config(tmp_path, seed=123))

    assert [game["moves"] for game in first["games"]] == [
        game["moves"] for game in second["games"]
    ]
    assert first["summary"]["wins"] == second["summary"]["wins"]
    assert first["summary"]["draws"] == second["summary"]["draws"]
    assert first["summary"]["losses"] == second["summary"]["losses"]


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
            output_dir=str(tmp_path / "arena"),
            seed=0,
            num_games=1,
            max_ply=10,
            agent=BotConfig(kind="material"),
            opponent=BotConfig(kind="random"),
        )
    )
    assert result["summary"]["illegal_moves"] == 0


def test_bot_config_rejects_policy_only_without_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint_path"):
        BotConfig(kind="policy_only")


def test_bot_config_rejects_mcts_without_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint_path"):
        BotConfig(kind="mcts")


def test_bot_config_rejects_invalid_mcts_values() -> None:
    with pytest.raises(ValueError, match="simulations"):
        BotConfig(kind="mcts", checkpoint_path="checkpoint.pt", simulations=0)
    with pytest.raises(ValueError, match="c_puct"):
        BotConfig(kind="mcts", checkpoint_path="checkpoint.pt", c_puct=0.0)
    with pytest.raises(ValueError, match="inference_batch_size"):
        BotConfig(kind="mcts", checkpoint_path="checkpoint.pt", inference_batch_size=0)


def test_run_match_records_mcts_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_bot(config: BotConfig, *, default_seed: int) -> arena_module.MaterialBot:
        del default_seed
        return arena_module.MaterialBot(name=config.name or config.kind)

    monkeypatch.setattr(arena_module, "build_bot", fake_build_bot)

    result = run_match(
        ArenaConfig(
            run_id="arena_test",
            output_dir=str(tmp_path / "arena"),
            seed=0,
            num_games=1,
            max_ply=1,
            agent=BotConfig(
                kind="mcts",
                name="mcts_agent",
                checkpoint_path="checkpoint.pt",
                simulations=50,
                c_puct=1.5,
                inference_batch_size=8,
            ),
            opponent=BotConfig(kind="material"),
        )
    )

    assert result["protocol"]["mcts_budget"] == {
        "agent": {"simulations": 50, "c_puct": 1.5, "inference_batch_size": 8},
        "opponent": None,
    }


def test_arena_config_rejects_negative_move_delay(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="move_delay_seconds"):
        ArenaConfig(
            output_dir=str(tmp_path / "arena"),
            move_delay_seconds=-1.0,
            agent=BotConfig(kind="material"),
            opponent=BotConfig(kind="random"),
        )


def test_run_arena_module_writes_result_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "arena.yaml"
    output_dir = tmp_path / "arena_out"
    config = {
        "run_id": "arena_script_test",
        "output_dir": str(output_dir),
        "seed": 3,
        "num_games": 2,
        "max_ply": 2,
        "agent": {"kind": "material"},
        "opponent": {"kind": "random"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(arena_module, "current_git_commit", lambda: "abc123")

    result_path = arena_module.run_arena(config_path)

    assert result_path == output_dir / "result.json"
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "source_config_path.txt").read_text(encoding="utf-8") == (
        str(config_path) + "\n"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 2
    assert result["run"]["status"] == "completed"
    assert result["run"]["id"] == "arena_script_test"
    assert result["run"]["config_path"] == str(config_path)
    assert result["run"]["git_commit"] == "abc123"
    assert result["config"]["agent"]["kind"] == "material"
    assert len(result["games"]) == 2
