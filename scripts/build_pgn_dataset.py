#!/usr/bin/env python3
"""Build a supervised JSONL dataset from a YAML config."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from mcchess.data.dataset_builder import build_dataset


@dataclass(frozen=True)
class DatasetBuildConfig:
    source: str
    output_dir: str
    manifest_path: str
    source_description: str = ""
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)
    split_seed: int = 0
    filters: Mapping[str, object] | None = None
    show_progress: bool = True


def parse_split_ratios(raw: object) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("split_ratios must contain exactly three values")

    ratios = (float(raw[0]), float(raw[1]), float(raw[2]))
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("split_ratios cannot contain negative values")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split_ratios must sum to 1.0")
    return ratios


def load_config(path: str | Path) -> DatasetBuildConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"{config_path} must be a YAML config file")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    data: dict[str, Any] = dict(raw)
    if "split_ratios" in data:
        data["split_ratios"] = parse_split_ratios(data["split_ratios"])
    filters = data.get("filters")
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("filters must be a mapping when set")
    return DatasetBuildConfig(**data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a supervised McChess PGN dataset.")
    parser.add_argument("config", type=Path, help="YAML dataset build config.")
    parser.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        default=None,
        help="Show terminal progress while reading the PGN.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable terminal progress even if the config enables it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    show_progress = config.show_progress if args.show_progress is None else args.show_progress
    manifest_path = build_dataset(
        source=config.source,
        output_dir=config.output_dir,
        manifest_path=config.manifest_path,
        source_description=config.source_description,
        split_ratios=config.split_ratios,
        split_seed=config.split_seed,
        filters=config.filters,
        show_progress=show_progress,
    )
    print(f"wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
