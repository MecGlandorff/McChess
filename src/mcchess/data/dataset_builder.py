"""Build a supervised dataset from a PGN source.

Two-pass design: pass one enumerates games and assigns each to train/val/test
by seeded RNG; pass two streams samples through the reader and routes them to
the appropriate shard file. A manifest JSON is written alongside the shards
with the fields required by `docs/DATASET_PROTOCOL.md`.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, TextIO

import mcchess
from mcchess.data.pgn_reader import ReaderCounters, iter_pgn_games, iter_samples

SCHEMA_VERSION: Final[int] = 1
_SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")


@dataclass
class BuildConfig:
    source: Path
    source_description: str
    output_dir: Path
    manifest_path: Path
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)
    split_seed: int = 0
    filters: dict[str, Any] = field(default_factory=dict)


def build_dataset(config: BuildConfig) -> Path:
    """Materialize JSONL shards and a manifest. Returns the manifest path."""

    _validate_ratios(config.split_ratios)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    game_split = _assign_splits(config)

    shard_paths = {name: config.output_dir / f"{name}.jsonl" for name in _SPLIT_NAMES}
    per_split_position_counts = {name: 0 for name in _SPLIT_NAMES}
    used_game_ids_per_split: dict[str, set[str]] = {name: set() for name in _SPLIT_NAMES}
    counters = ReaderCounters()

    with _open_shards(shard_paths) as shards:
        with config.source.open("r", encoding="utf-8", errors="replace") as stream:
            for sample in iter_samples(iter_pgn_games(stream), counters=counters):
                split = game_split.get(sample.game_id)
                if split is None:
                    # Game was skipped during the assignment pass; should not happen
                    # because both passes use the same enumeration, but guard anyway.
                    continue
                sample.split = split
                shards[split].write(json.dumps(_sample_to_dict(sample)) + "\n")
                per_split_position_counts[split] += 1
                used_game_ids_per_split[split].add(sample.game_id)

    manifest = _build_manifest(
        config=config,
        counters=counters,
        per_split_position_counts=per_split_position_counts,
        used_game_ids_per_split=used_game_ids_per_split,
    )
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config.manifest_path


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    total = sum(ratios)
    if not (0.999 < total < 1.001):
        raise ValueError(f"split_ratios must sum to 1.0, got {total}")
    if any(r < 0 for r in ratios):
        raise ValueError("split_ratios must be non-negative")


def _assign_splits(config: BuildConfig) -> dict[str, str]:
    """First pass: walk the PGN once to enumerate games and assign each a split."""

    rng = random.Random(config.split_seed)
    train_p, val_p, _test_p = config.split_ratios
    assignments: dict[str, str] = {}

    with config.source.open("r", encoding="utf-8", errors="replace") as stream:
        for game_index, _game in enumerate(iter_pgn_games(stream)):
            game_id = f"g{game_index:06d}"
            r = rng.random()
            if r < train_p:
                assignments[game_id] = "train"
            elif r < train_p + val_p:
                assignments[game_id] = "val"
            else:
                assignments[game_id] = "test"
    return assignments


class _ShardWriters:
    def __init__(self, paths: dict[str, Path]):
        self._paths = paths
        self._files: dict[str, TextIO] = {}

    def __enter__(self) -> dict[str, TextIO]:
        for name, path in self._paths.items():
            self._files[name] = path.open("w", encoding="utf-8")
        return self._files

    def __exit__(self, *exc: object) -> None:
        for f in self._files.values():
            f.close()


def _open_shards(paths: dict[str, Path]) -> _ShardWriters:
    return _ShardWriters(paths)


def _sample_to_dict(sample: Any) -> dict[str, Any]:
    return {
        "game_id": sample.game_id,
        "ply": sample.ply,
        "fen": sample.fen,
        "move_uci": sample.move_uci,
        "policy_index": sample.policy_index,
        "value": sample.value,
        "result": sample.result,
        "split": sample.split,
    }


def _build_manifest(
    *,
    config: BuildConfig,
    counters: ReaderCounters,
    per_split_position_counts: dict[str, int],
    used_game_ids_per_split: dict[str, set[str]],
) -> dict[str, Any]:
    games_per_split = {name: len(ids) for name, ids in used_game_ids_per_split.items()}

    filters = dict(config.filters)
    filters.setdefault("duplicate_handling", "not_implemented")

    return {
        "source": str(config.source),
        "source_description": config.source_description,
        "source_checksum": _sha256_of_file(config.source),
        "num_games_raw": counters.games_read,
        "num_games_used": counters.games_used,
        "num_games_skipped": (
            counters.games_skipped_corrupt + counters.games_skipped_unknown_result
        ),
        "num_games_skipped_corrupt": counters.games_skipped_corrupt,
        "num_games_skipped_unknown_result": counters.games_skipped_unknown_result,
        "num_duplicate_games": 0,
        "num_positions": counters.positions_emitted,
        "filters": filters,
        "split": {
            "ratios": {
                "train": config.split_ratios[0],
                "val": config.split_ratios[1],
                "test": config.split_ratios[2],
            },
            "games_per_split": games_per_split,
            "positions_per_split": per_split_position_counts,
        },
        "split_seed": config.split_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _code_version(),
        "schema_version": SCHEMA_VERSION,
    }


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version() -> str:
    base = mcchess.__version__
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return base
    return f"{base}+{sha}" if sha else base

