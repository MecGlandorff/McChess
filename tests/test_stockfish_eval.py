from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chess
import pytest
import yaml  # type: ignore[import-untyped]

from mcchess.eval import stockfish as stockfish_module
from mcchess.eval import stockfish_ladder as ladder_module
from mcchess.eval.arena import BotConfig
from mcchess.eval.openings import PAIRED_FEN_OPENING_PROTOCOL
from mcchess.eval.stockfish import (
    ScheduledStockfishGame,
    StockfishEvalConfig,
    StockfishGameRecord,
    StockfishLevelConfig,
    estimate_mcchess_elo,
    format_markdown_report,
    game_summary_rows,
    iter_scheduled_games,
    play_stockfish_game,
)
from mcchess.eval.stockfish_gui import StockfishEvalViewer

REPORT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "eval"
    / "stockfish_mcts200_resnet_b_elo_200games.yaml"
)
BATCH8_SMOKE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "eval"
    / "stockfish_mcts200_resnet_c_epoch22_batch8_elo.yaml"
)
BATCH8_MCTS1000_SMOKE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "eval"
    / "stockfish_mcts1000_resnet_c_epoch22_batch8_elo.yaml"
)
BATCH8_REPORT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "eval"
    / "stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games.yaml"
)
LADDER_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "eval"
    / "stockfish_uci_ladder_selfcheck.yaml"
)


class ScriptedBot:
    def __init__(self, name: str, moves: list[str]) -> None:
        self.name = name
        self.moves = [chess.Move.from_uci(move) for move in moves]

    def choose_move(self, board: chess.Board) -> chess.Move:
        move = self.moves.pop(0)
        assert move in board.legal_moves
        return move


class FakePlayResult:
    def __init__(self, move: chess.Move) -> None:
        self.move = move


class ScriptedEngine:
    id = {"name": "fake-stockfish"}
    options = {"UCI_Elo": object(), "UCI_LimitStrength": object()}

    def __init__(self, moves: list[str]) -> None:
        self.moves = [chess.Move.from_uci(move) for move in moves]
        self.configured: list[dict[str, object]] = []

    def configure(self, options: dict[str, object]) -> None:
        self.configured.append(dict(options))

    def play(self, board: chess.Board, limit: object) -> FakePlayResult:
        del limit
        move = self.moves.pop(0)
        assert move in board.legal_moves
        return FakePlayResult(move)


def elo_level(elo: int = 1600, *, games: int = 2) -> StockfishLevelConfig:
    return StockfishLevelConfig(
        name=f"uci_elo_{elo}_t1s",
        games=games,
        include_in_elo=True,
        options={"UCI_LimitStrength": True, "UCI_Elo": elo},
        limit={"time": 1.0},
    )


def sanity_level(*, games: int = 2) -> StockfishLevelConfig:
    return StockfishLevelConfig(
        name="full_stockfish_t1s_sanity",
        games=games,
        include_in_elo=False,
        options={"Skill Level": 20, "UCI_LimitStrength": False},
        limit={"time": 1.0},
    )


def mcts_agent_config(*, simulations: int = 200) -> BotConfig:
    return BotConfig(
        kind="mcts",
        name="resnet_b_mcts_200",
        checkpoint_path="checkpoint.pt",
        simulations=simulations,
        c_puct=1.5,
    )


def eval_config(levels: list[StockfishLevelConfig]) -> StockfishEvalConfig:
    return StockfishEvalConfig(
        run_id="stockfish_test",
        output_dir="runs/external_stockfish/test",
        agent=mcts_agent_config(),
        stockfish_levels=levels,
        max_ply=20,
    )


def test_play_stockfish_game_records_mcchess_black_win() -> None:
    level = elo_level(1600, games=1)
    scheduled = ScheduledStockfishGame(
        game_index=0,
        level_game_index=0,
        level=level,
        mcchess_color=chess.BLACK,
    )
    engine = ScriptedEngine(["f2f3", "g2g4"])
    bot = ScriptedBot("mcchess", ["e7e5", "d8h4"])

    game = play_stockfish_game(bot, engine, scheduled, max_ply=10)

    assert game.status == "completed"
    assert game.white == "uci_elo_1600_t1s"
    assert game.black == "mcchess"
    assert game.result == "0-1"
    assert game.winner == "black"
    assert game.winner_name == "mcchess"
    assert game.mcchess_color == "black"
    assert game.mcchess_score == 1.0
    assert game.termination == "checkmate"
    assert engine.configured == [level.options]


def test_play_stockfish_game_records_mcchess_white_win() -> None:
    level = elo_level(1600, games=1)
    scheduled = ScheduledStockfishGame(
        game_index=0,
        level_game_index=0,
        level=level,
        mcchess_color=chess.WHITE,
    )
    engine = ScriptedEngine(["e7e5", "b8c6", "g8f6"])
    bot = ScriptedBot("mcchess", ["e2e4", "d1h5", "f1c4", "h5f7"])

    game = play_stockfish_game(bot, engine, scheduled, max_ply=10)

    assert game.status == "completed"
    assert game.white == "mcchess"
    assert game.black == "uci_elo_1600_t1s"
    assert game.result == "1-0"
    assert game.winner == "white"
    assert game.winner_name == "mcchess"
    assert game.mcchess_color == "white"
    assert game.mcchess_score == 1.0
    assert game.termination == "checkmate"


def test_schedule_runs_full_stockfish_first_and_alternates_colors() -> None:
    config = eval_config([sanity_level(games=2), elo_level(1600, games=2), elo_level(1700, games=2)])

    scheduled = iter_scheduled_games(config)

    assert [game.level.name for game in scheduled] == [
        "full_stockfish_t1s_sanity",
        "full_stockfish_t1s_sanity",
        "uci_elo_1600_t1s",
        "uci_elo_1600_t1s",
        "uci_elo_1700_t1s",
        "uci_elo_1700_t1s",
    ]
    assert [game.mcchess_color for game in scheduled] == [
        chess.WHITE,
        chess.BLACK,
        chess.WHITE,
        chess.BLACK,
        chess.WHITE,
        chess.BLACK,
    ]


def test_schedule_pairs_configured_opening_fens_within_each_level() -> None:
    first = chess.Board()
    first.push_san("e4")
    first.push_san("e5")
    second = chess.Board()
    second.push_san("d4")
    second.push_san("d5")
    config = StockfishEvalConfig(
        run_id="stockfish_openings_test",
        output_dir="runs/external_stockfish/test",
        agent=mcts_agent_config(),
        stockfish_levels=[elo_level(1600, games=4)],
        opening_fens=(first.fen(), second.fen()),
    )

    scheduled = iter_scheduled_games(config)

    assert stockfish_module.opening_protocol(config.opening_fens) == PAIRED_FEN_OPENING_PROTOCOL
    assert [game.opening_index for game in scheduled] == [0, 0, 1, 1]
    assert [game.starting_fen for game in scheduled] == [
        first.fen(),
        first.fen(),
        second.fen(),
        second.fen(),
    ]


def test_stockfish_config_requires_explicit_mcts_budget() -> None:
    with pytest.raises(ValueError, match="simulations explicitly"):
        StockfishEvalConfig(
            run_id="bad",
            output_dir="runs/external_stockfish/bad",
            agent=mcts_agent_config(simulations=None),
            stockfish_levels=[sanity_level()],
        )


def test_stockfish_config_accepts_non_200_mcts_budget() -> None:
    config = StockfishEvalConfig(
        run_id="mcts_1000",
        output_dir="runs/external_stockfish/mcts_1000",
        agent=mcts_agent_config(simulations=1000),
        stockfish_levels=[sanity_level()],
    )

    assert config.agent.simulations == 1000


def test_elo_level_requires_uci_elo_when_included() -> None:
    with pytest.raises(ValueError, match="UCI_Elo"):
        StockfishLevelConfig(
            name="skill_only",
            games=1,
            include_in_elo=True,
            options={"Skill Level": 0},
            limit={"time": 1.0},
        )


def test_elo_estimate_excludes_sanity_games() -> None:
    estimate = estimate_mcchess_elo(
        [
            game_record(0, elo=None, score=0.0, include_in_elo=False),
            game_record(1, elo=1600, score=0.5),
            game_record(2, elo=1800, score=0.5),
            game_record(3, elo=2000, score=0.5),
        ]
    )

    assert estimate.status == "ok"
    assert estimate.included_games == 3
    assert estimate.score == 0.5
    assert estimate.estimated_elo is not None
    assert 1750 <= estimate.estimated_elo <= 1850


def test_elo_estimate_marks_all_wins_as_upper_bounded() -> None:
    estimate = estimate_mcchess_elo(
        [
            game_record(0, elo=1600, score=1.0),
            game_record(1, elo=1700, score=1.0),
        ]
    )

    assert estimate.status == "ok"
    assert estimate.bounded == "upper"
    assert estimate.upper_95 is None
    assert estimate.estimated_elo is not None
    assert estimate.estimated_elo > 1700


def test_game_summary_rows_show_winner_name_and_mcchess_score() -> None:
    game = game_record(0, elo=1600, score=1.0)

    row = game_summary_rows([asdict(game)])[0]

    assert row["Game"] == "1"
    assert row["Stockfish level"] == "uci_elo_1600_t1s"
    assert row["Winner"] == "white"
    assert row["Winner name"] == "mcchess"
    assert row["McChess score"] == "1.0"
    assert row["Included in Elo"] == "yes"


def test_run_stockfish_match_aggregates_from_mcchess_perspective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = eval_config([elo_level(1600, games=1)])
    engine = ScriptedEngine(["e7e5", "b8c6", "g8f6"])

    def fake_build_bot(config: BotConfig, *, default_seed: int) -> ScriptedBot:
        del config, default_seed
        return ScriptedBot("mcchess", ["e2e4", "d1h5", "f1c4", "h5f7"])

    monkeypatch.setattr(stockfish_module, "build_bot", fake_build_bot)

    result = stockfish_module.run_stockfish_match(config, engine)

    assert result["schema_version"] == 2
    assert result["run"]["status"] == "completed"
    assert result["summary"]["wins"] == 1
    assert result["summary"]["draws"] == 0
    assert result["summary"]["losses"] == 0
    assert result["summary"]["score"] == 1.0
    assert result["games"][0]["winner_name"] == "mcchess"
    assert result["games"][0]["mcchess_score"] == 1.0


def test_run_stockfish_eval_module_loads_config_and_resolves_path(tmp_path: Path) -> None:
    binary_path = tmp_path / "stockfish.exe"
    binary_path.write_text("", encoding="utf-8")
    config_path = tmp_path / "stockfish.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "stockfish_script_test",
                "output_dir": str(tmp_path / "out"),
                "stockfish_path": str(binary_path),
                "seed": 0,
                "max_ply": 10,
                "agent": {
                    "kind": "mcts",
                    "name": "resnet_b_mcts_200",
                    "checkpoint_path": "checkpoint.pt",
                    "simulations": 200,
                    "c_puct": 1.5,
                },
                "stockfish_levels": [
                    {
                        "name": "full_stockfish_t1s_sanity",
                        "games": 2,
                        "include_in_elo": False,
                        "options": {"Skill Level": 20, "UCI_LimitStrength": False},
                        "limit": {"time": 1.0},
                    },
                    {
                        "name": "uci_elo_1600_t1s",
                        "games": 2,
                        "include_in_elo": True,
                        "options": {"UCI_LimitStrength": True, "UCI_Elo": 1600},
                        "limit": {"time": 1.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = stockfish_module.load_config(config_path)

    assert config.run_id == "stockfish_script_test"
    assert config.num_games == 4
    assert stockfish_module.resolve_stockfish_path(config) == str(binary_path)


def test_run_stockfish_eval_scopes_keep_awake_around_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    expected_path = tmp_path / "result.json"

    class FakeKeepAwake:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, *exc_info: object) -> None:
            events.append(("exit", exc_info[0]))

    def fake_keep_system_awake(*, enabled: bool) -> FakeKeepAwake:
        events.append(("enabled", enabled))
        return FakeKeepAwake()

    def fake_run_stockfish_eval(*args: object, **kwargs: object) -> Path:
        events.append(("run", args, kwargs))
        return expected_path

    monkeypatch.setattr(stockfish_module, "keep_system_awake", fake_keep_system_awake)
    monkeypatch.setattr(stockfish_module, "_run_stockfish_eval", fake_run_stockfish_eval)

    result = stockfish_module.run_stockfish_eval("config.yaml", keep_awake=True)

    assert result == expected_path
    assert events[0:2] == [("enabled", True), "enter"]
    assert events[2][0] == "run"
    assert events[3] == ("exit", None)


def test_report_scale_stockfish_config_has_200_elo_games() -> None:
    config = stockfish_module.load_config(REPORT_CONFIG_PATH)

    assert config.num_games == 202
    included_games = sum(level.games for level in config.stockfish_levels if level.include_in_elo)
    included_elos = [level.stockfish_elo for level in config.stockfish_levels if level.include_in_elo]
    assert included_games == 200
    assert included_elos == [1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]


def test_resnet_c_batch8_configs_keep_smoke_and_report_runs_separate() -> None:
    smoke = stockfish_module.load_config(BATCH8_SMOKE_CONFIG_PATH)
    mcts1000_smoke = stockfish_module.load_config(BATCH8_MCTS1000_SMOKE_CONFIG_PATH)
    report = stockfish_module.load_config(BATCH8_REPORT_CONFIG_PATH)

    assert smoke.num_games == 22
    assert smoke.agent.simulations == 200
    assert smoke.agent.inference_batch_size == 8
    assert "batch8" in smoke.run_id
    assert "batch8" in smoke.output_dir

    assert mcts1000_smoke.num_games == 22
    assert mcts1000_smoke.agent.simulations == 1000
    assert mcts1000_smoke.agent.inference_batch_size == 8
    assert "batch8" in mcts1000_smoke.run_id
    assert "batch8" in mcts1000_smoke.output_dir

    assert report.num_games == 202
    assert report.agent.simulations == 1000
    assert report.agent.inference_batch_size == 8
    assert "batch8" in report.run_id
    assert "batch8" in report.output_dir
    assert report.output_dir != smoke.output_dir
    assert report.output_dir != mcts1000_smoke.output_dir


def test_stockfish_report_labels_uci_elo_and_time_limit() -> None:
    game = game_record(0, elo=1800, score=1.0)
    result = {
        "schema_version": 2,
        "run": {"id": "stockfish_test", "status": "completed"},
        "summary": {
            "games_completed": 1,
            "games_scheduled": 1,
            "wins": 1,
            "draws": 0,
            "losses": 0,
            "score": 1.0,
        },
        "protocol": {
            "max_ply": 180,
            "draw_rule": "python_chess_outcome_or_max_ply_draw",
            "color_policy": "alternating",
            "opening_protocol": "standard_initial_position",
        },
        "metrics": {"elo_estimate": asdict(estimate_mcchess_elo([game]))},
        "games": [asdict(game)],
    }

    report = format_markdown_report(result)

    assert "UCI_Elo" in report
    assert "time=1.0s/move" in report
    assert "not an online or official rating" in report
    assert "Do not use this report as training data" in report


def test_run_stockfish_eval_show_uses_viewer_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_path = tmp_path / "stockfish.exe"
    binary_path.write_text("", encoding="utf-8")
    config_path = tmp_path / "stockfish.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "stockfish_show_test",
                "output_dir": str(tmp_path / "out"),
                "stockfish_path": str(binary_path),
                "seed": 0,
                "max_ply": 10,
                "agent": {
                    "kind": "mcts",
                    "name": "resnet_b_mcts_200",
                    "checkpoint_path": "checkpoint.pt",
                    "simulations": 200,
                    "c_puct": 1.5,
                },
                "stockfish_levels": [
                    {
                        "name": "uci_elo_1600_t1s",
                        "games": 1,
                        "include_in_elo": True,
                        "options": {"UCI_LimitStrength": True, "UCI_Elo": 1600},
                        "limit": {"time": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    viewers = []

    class FakeViewer:
        def __init__(self) -> None:
            self.moves = []
            self.games = []
            self.completed_path = None
            self.waited = False
            viewers.append(self)

        def on_move(self, event: dict[str, object]) -> None:
            self.moves.append(event)

        def on_game(self, game: StockfishGameRecord) -> None:
            self.games.append(game)

        def mark_complete(self, result_path: str) -> None:
            self.completed_path = result_path

        def wait_until_closed(self) -> None:
            self.waited = True

    class FakeEngine:
        id = {"name": "fake-stockfish"}
        options = {"UCI_Elo": object(), "UCI_LimitStrength": object()}

        def quit(self) -> None:
            pass

    def fake_popen_uci(path: str) -> FakeEngine:
        assert path == str(binary_path)
        return FakeEngine()

    def fake_run_stockfish_match(
        config: StockfishEvalConfig,
        engine: object,
        **kwargs: object,
    ) -> dict[str, object]:
        del config, engine
        move_callback = kwargs["move_callback"]
        game_callback = kwargs["game_callback"]
        assert callable(move_callback)
        assert callable(game_callback)
        move_callback(
            {
                "game_index": 0,
                "ply": 1,
                "level": "uci_elo_1600_t1s",
                "color": "white",
                "bot": "mcchess",
                "san": "e4",
                "uci": "e2e4",
                "fen": chess.Board().fen(),
            }
        )
        game = game_record(0, elo=1600, score=1.0)
        game_callback(game)
        return {
            "schema_version": 2,
            "run": {"id": "stockfish_show_test", "status": "completed"},
            "summary": {
                "games_completed": 1,
                "games_scheduled": 1,
                "wins": 1,
                "draws": 0,
                "losses": 0,
                "score": 1.0,
            },
            "protocol": {
                "max_ply": 10,
                "draw_rule": "python_chess_outcome_or_max_ply_draw",
                "color_policy": "alternating",
                "opening_protocol": "standard_initial_position",
            },
            "metrics": {"elo_estimate": asdict(estimate_mcchess_elo([game]))},
            "games": [asdict(game)],
        }

    monkeypatch.setattr(stockfish_module, "StockfishEvalViewer", FakeViewer)
    monkeypatch.setattr(stockfish_module.chess.engine.SimpleEngine, "popen_uci", fake_popen_uci)
    monkeypatch.setattr(stockfish_module, "run_stockfish_match", fake_run_stockfish_match)

    result_path = stockfish_module.run_stockfish_eval(config_path, show=True)

    assert result_path.exists()
    assert len(viewers) == 1
    assert len(viewers[0].moves) == 1
    assert len(viewers[0].games) == 1
    assert viewers[0].completed_path == str(result_path)
    assert viewers[0].waited is True


def test_run_stockfish_eval_can_disable_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_path = tmp_path / "stockfish.exe"
    binary_path.write_text("", encoding="utf-8")
    config_path = tmp_path / "stockfish.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "stockfish_progress_test",
                "output_dir": str(tmp_path / "out"),
                "stockfish_path": str(binary_path),
                "seed": 0,
                "max_ply": 10,
                "agent": {
                    "kind": "mcts",
                    "name": "resnet_b_mcts_200",
                    "checkpoint_path": "checkpoint.pt",
                    "simulations": 200,
                    "c_puct": 1.5,
                },
                "stockfish_levels": [
                    {
                        "name": "uci_elo_1600_t1s",
                        "games": 1,
                        "include_in_elo": True,
                        "options": {"UCI_LimitStrength": True, "UCI_Elo": 1600},
                        "limit": {"time": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    progress_disabled = []

    class FakeProgress:
        def __init__(self, **kwargs: object) -> None:
            progress_disabled.append(kwargs["disable"])

        def update(self, amount: int) -> None:
            assert amount == 1

        def set_postfix(self, **kwargs: object) -> None:
            assert kwargs["level"] == "uci_elo_1600_t1s"

        def close(self) -> None:
            pass

    class FakeEngine:
        id = {"name": "fake-stockfish"}
        options = {"UCI_Elo": object(), "UCI_LimitStrength": object()}

        def quit(self) -> None:
            pass

    def fake_run_stockfish_match(
        config: StockfishEvalConfig,
        engine: object,
        **kwargs: object,
    ) -> dict[str, object]:
        del config, engine
        game = game_record(0, elo=1600, score=1.0)
        kwargs["game_callback"](game)
        return {
            "schema_version": 2,
            "run": {"id": "stockfish_progress_test", "status": "completed"},
            "summary": {
                "games_completed": 1,
                "games_scheduled": 1,
                "wins": 1,
                "draws": 0,
                "losses": 0,
                "score": 1.0,
            },
            "protocol": {
                "max_ply": 10,
                "draw_rule": "python_chess_outcome_or_max_ply_draw",
                "color_policy": "alternating",
                "opening_protocol": "standard_initial_position",
            },
            "metrics": {"elo_estimate": asdict(estimate_mcchess_elo([game]))},
            "games": [asdict(game)],
        }

    monkeypatch.setattr(stockfish_module, "tqdm", lambda **kwargs: FakeProgress(**kwargs))
    monkeypatch.setattr(
        stockfish_module.chess.engine.SimpleEngine,
        "popen_uci",
        lambda _path: FakeEngine(),
    )
    monkeypatch.setattr(stockfish_module, "run_stockfish_match", fake_run_stockfish_match)

    stockfish_module.run_stockfish_eval(config_path, show_progress=False)

    assert progress_disabled == [True]


def test_run_stockfish_eval_writes_failed_artifact_on_engine_startup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_path = tmp_path / "stockfish.exe"
    binary_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "out"
    config_path = tmp_path / "stockfish.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "stockfish_startup_failure_test",
                "output_dir": str(output_dir),
                "stockfish_path": str(binary_path),
                "seed": 0,
                "max_ply": 10,
                "agent": {
                    "kind": "mcts",
                    "name": "resnet_b_mcts_200",
                    "checkpoint_path": "checkpoint.pt",
                    "simulations": 200,
                    "c_puct": 1.5,
                },
                "stockfish_levels": [
                    {
                        "name": "uci_elo_1600_t1s",
                        "games": 1,
                        "include_in_elo": True,
                        "options": {"UCI_LimitStrength": True, "UCI_Elo": 1600},
                        "limit": {"time": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeProgress:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def update(self, amount: int) -> None:
            del amount

        def set_postfix(self, **kwargs: object) -> None:
            del kwargs

        def close(self) -> None:
            pass

    monkeypatch.setattr(stockfish_module, "tqdm", lambda **kwargs: FakeProgress(**kwargs))

    def fail_popen_uci(path: str) -> object:
        assert path == str(binary_path)
        raise PermissionError("cannot start engine")

    monkeypatch.setattr(stockfish_module.chess.engine.SimpleEngine, "popen_uci", fail_popen_uci)

    with pytest.raises(PermissionError, match="cannot start engine"):
        stockfish_module.run_stockfish_eval(config_path)

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["run"]["status"] == "failed"
    assert result["summary"]["games_completed"] == 0
    assert result["summary"]["failure"]["stage"] == "setup"
    assert "PermissionError" in result["summary"]["failure"]["error"]


def test_stockfish_viewer_draws_unicode_piece_symbols() -> None:
    class FakeCanvas:
        def __init__(self) -> None:
            self.texts: list[str] = []

        def delete(self, tag: str) -> None:
            assert tag == "all"

        def create_rectangle(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def create_text(self, *args: object, **kwargs: object) -> None:
            del args
            self.texts.append(str(kwargs["text"]))

    viewer = object.__new__(StockfishEvalViewer)
    viewer.canvas = FakeCanvas()

    viewer._draw_board(chess.Board(), last_move_uci=None)

    white_pawn = chess.Piece(chess.PAWN, chess.WHITE).unicode_symbol()
    black_king = chess.Piece(chess.KING, chess.BLACK).unicode_symbol()
    assert white_pawn in viewer.canvas.texts
    assert black_king in viewer.canvas.texts
    assert "P" not in viewer.canvas.texts


def test_stockfish_ladder_config_schedules_adjacent_paired_games() -> None:
    config = ladder_module.load_config(LADDER_CONFIG_PATH)
    scheduled = ladder_module.iter_ladder_games(config)

    assert config.num_games == 18
    assert [(game.lower_elo, game.higher_elo) for game in scheduled[:4]] == [
        (1600, 1700),
        (1600, 1700),
        (1700, 1800),
        (1700, 1800),
    ]
    assert [(game.white_elo, game.black_elo) for game in scheduled[:4]] == [
        (1600, 1700),
        (1700, 1600),
        (1700, 1800),
        (1800, 1700),
    ]


def test_stockfish_ladder_report_rejects_real_elo_calibration_claim() -> None:
    result = {
        "schema_version": 2,
        "run": {"id": "ladder_test", "status": "completed"},
        "summary": {
            "games_completed": 0,
            "games_scheduled": 0,
            "higher_elo_score": None,
        },
        "protocol": {
            "max_ply": 180,
            "draw_rule": "python_chess_outcome_or_max_ply_draw",
            "color_policy": "paired_colors_per_adjacent_uci_elo_pair",
            "opening_protocol": "standard_initial_position",
        },
        "games": [],
    }

    report = ladder_module.format_report(result)

    assert "not a calibration to Lichess Elo, FIDE Elo, CCRL Elo" in report
    assert "Do not use Stockfish moves" in report


def test_stockfish_ladder_progress_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ladder_module.StockfishLadderConfig(
        run_id="ladder_progress_test",
        output_dir="runs/external_stockfish/test",
        stockfish_elos=[1600, 1700],
        limit={"time": 1.0},
    )
    progress_disabled = []
    postfixes = []

    class FakeProgress:
        def __init__(self, iterable: object, **kwargs: object) -> None:
            self.iterable = iterable
            progress_disabled.append(kwargs["disable"])

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs: object) -> None:
            postfixes.append(kwargs)

    class FakeEngine:
        id = {"name": "fake-stockfish"}
        options = {"UCI_Elo": object(), "UCI_LimitStrength": object()}

    def fake_play_ladder_game(*args: object, **kwargs: object):
        scheduled = args[2]
        return ladder_module.LadderGameRecord(
            game_index=scheduled.game_index,
            pair_index=scheduled.pair_index,
            lower_elo=scheduled.lower_elo,
            higher_elo=scheduled.higher_elo,
            white_elo=scheduled.white_elo,
            black_elo=scheduled.black_elo,
            status="completed",
            result="1/2-1/2",
            winner=None,
            winner_elo=None,
            higher_elo_score=0.5,
            termination="max_ply",
            ply_count=180,
            final_fen=chess.Board().fen(),
            moves=[],
            white_options={},
            black_options={},
            stockfish_limit={"time": 1.0},
        )

    monkeypatch.setattr(ladder_module, "tqdm", FakeProgress)
    monkeypatch.setattr(ladder_module, "play_ladder_game", fake_play_ladder_game)

    result = ladder_module.run_ladder_match(config, FakeEngine(), FakeEngine(), show_progress=True)

    assert result["summary"]["games_completed"] == 2
    assert progress_disabled == [False]
    assert postfixes[-1]["pair"] == "1600-1700"
    assert postfixes[-1]["result"] == "1/2-1/2"


def game_record(
    game_index: int,
    *,
    elo: int | None,
    score: float,
    include_in_elo: bool = True,
) -> StockfishGameRecord:
    if score == 1.0:
        result = "1-0"
        winner = "white"
        winner_name = "mcchess"
    elif score == 0.0:
        result = "0-1"
        winner = "black"
        winner_name = "stockfish"
    else:
        result = "1/2-1/2"
        winner = None
        winner_name = None
    return StockfishGameRecord(
        game_index=game_index,
        level=f"uci_elo_{elo}_t1s" if elo is not None else "full_stockfish_t1s_sanity",
        level_game_index=0,
        stockfish_elo=elo,
        include_in_elo=include_in_elo,
        status="completed",
        mcchess_color="white",
        white="mcchess",
        black="stockfish",
        result=result,
        winner=winner,
        winner_name=winner_name,
        mcchess_score=score,
        termination="max_ply",
        ply_count=180,
        final_fen=chess.Board().fen(),
        moves=[],
        stockfish_options={},
        stockfish_limit={"time": 1.0},
    )
