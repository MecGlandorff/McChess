#!/usr/bin/env python3
"""Run a reproducible McChess bot-vs-bot arena."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from mcchess.eval.arena import ArenaConfig, BotConfig, run_match


def load_config(path: str | Path) -> ArenaConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    agent_raw = raw.pop("agent", None)
    opponent_raw = raw.pop("opponent", None)
    if not isinstance(agent_raw, dict):
        raise ValueError("agent must be a YAML mapping")
    if not isinstance(opponent_raw, dict):
        raise ValueError("opponent must be a YAML mapping")

    return ArenaConfig(
        **raw,
        agent=BotConfig(**agent_raw),
        opponent=BotConfig(**opponent_raw),
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def run_arena(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    result = run_match(config)
    result["config_path"] = str(config_path)
    result["config"] = asdict(config)
    result["git_commit"] = git_commit()
    output_path = Path(config.output_path)
    write_json_atomic(output_path, result)
    print(f"saved arena result to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a McChess bot-vs-bot arena.")
    parser.add_argument("config", type=Path, help="YAML arena config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_arena(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
