"""Artifact and report writers for external Stockfish benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, cast

import yaml  # type: ignore[import-untyped]

from mcchess.eval.common import write_csv, write_json_atomic, write_text_atomic
from mcchess.eval.stockfish_protocol import SCOPE_NOTE, StockfishEvalConfig

GAME_CSV_FIELDS: Final[list[str]] = [
    "Game",
    "Stockfish level",
    "White",
    "Black",
    "Result",
    "Winner",
    "Winner name",
    "McChess score",
    "Included in Elo",
]


def write_artifacts(
    *,
    output_dir: Path,
    config_path: Path,
    config: StockfishEvalConfig,
    result: dict[str, Any],
) -> Path:
    """Write config copy, JSON result, CSV table, and Markdown report."""

    write_text_atomic(
        output_dir / "config.yaml",
        yaml.safe_dump(asdict(config), sort_keys=False),
    )
    result_path = output_dir / "result.json"
    write_json_atomic(result_path, result)
    rows = game_summary_rows(cast(Sequence[Mapping[str, Any]], result.get("games", [])))
    write_csv(output_dir / "games.csv", GAME_CSV_FIELDS, rows)
    write_text_atomic(output_dir / "report.md", format_markdown_report(result))
    write_text_atomic(output_dir / "source_config_path.txt", str(config_path) + "\n")
    return result_path


def game_summary_rows(games: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return display-friendly per-game rows for CSV and Markdown reports."""

    rows: list[dict[str, str]] = []
    for game in games:
        winner = game.get("winner")
        winner_name = game.get("winner_name")
        mcchess_score = game.get("mcchess_score")
        rows.append(
            {
                "Game": str(int(game["game_index"]) + 1),
                "Stockfish level": str(game["level"]),
                "White": str(game["white"]),
                "Black": str(game["black"]),
                "Result": str(game["result"]),
                "Winner": str(winner if winner is not None else "draw"),
                "Winner name": str(winner_name if winner_name is not None else "draw"),
                "McChess score": "" if mcchess_score is None else f"{float(mcchess_score):.1f}",
                "Included in Elo": "yes" if game.get("include_in_elo") else "no",
            }
        )
    return rows


def format_markdown_report(result: Mapping[str, Any]) -> str:
    """Format a compact Markdown report for an external benchmark run."""

    run = _mapping(result.get("run"))
    summary = _mapping(result.get("summary"))
    protocol = _mapping(result.get("protocol"))
    metrics = _mapping(result.get("metrics"))
    elo = metrics.get("elo_estimate", result.get("elo_estimate", {}))
    games = cast(Sequence[Mapping[str, Any]], result.get("games", []))
    rows = game_summary_rows(games)
    lines = [
        "# Stockfish External Benchmark Report",
        "",
        f"Run ID: `{run.get('id', result.get('run_id'))}`",
        "",
        f"Scope: {SCOPE_NOTE}",
        "",
        "## Summary",
        "",
        f"- Status: `{run.get('status', result.get('status'))}`",
        "- Games completed: "
        f"{summary.get('games_completed', result.get('games_completed'))} / "
        f"{summary.get('games_scheduled', result.get('num_games'))}",
        "- McChess W/D/L: "
        f"{summary.get('wins', result.get('wins'))} / "
        f"{summary.get('draws', result.get('draws'))} / "
        f"{summary.get('losses', result.get('losses'))}",
        f"- McChess score: {float(summary.get('score', result.get('score', 0.0))):.3f}",
        f"- Max ply: {protocol.get('max_ply', result.get('max_ply'))}",
        f"- Draw rule: `{protocol.get('draw_rule', result.get('draw_rule'))}`",
        f"- Color policy: `{protocol.get('color_policy', result.get('color_policy'))}`",
        f"- Opening protocol: `{protocol.get('opening_protocol', result.get('opening_protocol'))}`",
        f"- Stockfish search limit: `{_format_limit_summary(games)}`",
        "",
        "## Elo Estimate",
        "",
        _format_elo_summary(elo),
        "",
        "## Game Table",
        "",
        "| Game | Stockfish level | White | Black | Result | Winner | Winner name | McChess score | Included in Elo |",
        "|---:|---|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {Game} | {Stockfish level} | {White} | {Black} | {Result} | {Winner} | "
            "{Winner name} | {McChess score} | {Included in Elo} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- Stockfish UCI_Elo is an engine handicap setting, not an online or official rating.",
            "- Interpret each level together with its recorded Stockfish limit, such as `time=1.0s/move`.",
            "- Stockfish source describes the handicap range as approximate CCRL Blitz calibration and uses weakened move selection.",
            "- Source: https://github.com/official-stockfish/Stockfish/blob/master/src/search.h",
            "- Source: https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp",
            "- The estimate is only meaningful for the exact checkpoint, MCTS budget, Stockfish binary, and config recorded in this run.",
            "- Do not use this report as training data or as evidence of Lichess/FIDE Elo.",
        ]
    )
    return "\n".join(lines) + "\n"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _format_limit_summary(games: Sequence[Mapping[str, Any]]) -> str:
    limits: set[str] = set()
    for game in games:
        raw_limit = game.get("stockfish_limit")
        if not isinstance(raw_limit, Mapping):
            continue
        parts: list[str] = []
        if "time" in raw_limit:
            parts.append(f"time={raw_limit['time']}s/move")
        if "depth" in raw_limit:
            parts.append(f"depth={int(raw_limit['depth'])}")
        if "nodes" in raw_limit:
            parts.append(f"nodes={int(raw_limit['nodes'])}")
        if parts:
            limits.add(", ".join(parts))
    if not limits:
        return "not recorded"
    if len(limits) == 1:
        return next(iter(limits))
    return "; ".join(sorted(limits))


def _format_elo_summary(elo: object) -> str:
    if not isinstance(elo, Mapping) or elo.get("status") != "ok":
        return f"- Status: `{getattr(elo, 'get', lambda _key, default=None: default)('status', 'unknown')}`"

    estimated = elo.get("estimated_elo")
    lower = elo.get("lower_95")
    upper = elo.get("upper_95")
    bounded = elo.get("bounded")
    if bounded == "lower":
        interval = f"below {upper}" if upper is not None else "lower-bounded by tested bracket"
    elif bounded == "upper":
        interval = f"above {lower}" if lower is not None else "upper-bounded by tested bracket"
    else:
        interval = f"{lower} to {upper}"
    return (
        f"- Estimated McChess Elo: `{estimated}`\n"
        f"- Rough 95% interval: `{interval}`\n"
        f"- Included games: {elo.get('included_games')}\n"
        f"- Score in included games: {float(elo.get('score', 0.0)):.3f}\n"
        f"- Note: {elo.get('note')}"
    )
