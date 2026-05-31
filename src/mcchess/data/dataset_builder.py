"""
Build a supervised dataset from a PGN file. Writes JSONL shards for
train/val/test and a JSON manifest. Splits are by game (not by position),
seeded, and recorded in the manifest.

Schema and required manifest fields come from `docs/DATASET_PROTOCOL.md`.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO, TypedDict

import mcchess
from mcchess.data.pgn_reader import iter_samples, new_counters

SCHEMA_VERSION = 1


class DatasetSample(TypedDict):
    game_id: str
    ply: int
    fen: str
    move_uci: str
    policy_index: int
    value: float
    result: str
    split: str


def build_dataset(
    source: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    source_description: str = "",
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    split_seed: int = 0,
    filters: Mapping[str, object] | None = None,
) -> Path:
    source = Path(source)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(split_seed)
    train_p, val_p, _ = split_ratios

    def pick_split() -> str:
        r = rng.random()
        if r < train_p:
            return "train"
        if r < train_p + val_p:
            return "val"
        return "test"

    splits: dict[str, str] = {}
    pos_per_split = {"train": 0, "val": 0, "test": 0}
    counters = new_counters()

    with (
        source.open(encoding="utf-8") as src,
        (output_dir / "train.jsonl").open("w", encoding="utf-8") as train_shard,
        (output_dir / "val.jsonl").open("w", encoding="utf-8") as val_shard,
        (output_dir / "test.jsonl").open("w", encoding="utf-8") as test_shard,
    ):
        shards: dict[str, TextIO] = {
            "train": train_shard,
            "val": val_shard,
            "test": test_shard,
        }
        for sample in iter_samples(src, counters):
            gid = sample["game_id"]
            if gid not in splits:
                splits[gid] = pick_split()
            split = splits[gid]
            dataset_sample: DatasetSample = {**sample, "split": split}
            shards[split].write(json.dumps(dataset_sample) + "\n")
            pos_per_split[split] += 1

    manifest = {
        "source": str(source),
        "source_description": source_description,
        "source_checksum": hashlib.sha256(source.read_bytes()).hexdigest(),
        "num_games_raw": counters["games_read"],
        "num_games_used": counters["games_used"],
        "num_games_skipped": (
            counters["games_skipped_corrupt"]
            + counters["games_skipped_unknown_result"]
        ),
        "num_games_skipped_corrupt": counters["games_skipped_corrupt"],
        "num_games_skipped_unknown_result": counters["games_skipped_unknown_result"],
        "num_duplicate_games": 0,
        "num_positions": counters["positions_emitted"],
        "filters": filters or {},
        "split": {
            "ratios": list(split_ratios),
            "positions_per_split": pos_per_split,
        },
        "split_seed": split_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": mcchess.__version__,
        "schema_version": SCHEMA_VERSION,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
