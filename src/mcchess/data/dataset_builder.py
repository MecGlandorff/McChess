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
from typing import Protocol, TextIO, TypedDict, cast

import mcchess
from mcchess.data.pgn_reader import iter_samples, new_counters
from tqdm.auto import tqdm  # type: ignore[import-untyped]

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


class _ProgressBar(Protocol):
    n: int | float

    def update(self, n: int | float = 1) -> object:
        ...

    def set_postfix(self, **kwargs: object) -> object:
        ...

    def close(self) -> None:
        ...


class _ProgressTextStream:
    def __init__(self, stream: TextIO, progress: _ProgressBar, encoding: str) -> None:
        self._stream = stream
        self._progress = progress
        self._encoding = encoding

    def readline(self, size: int = -1) -> str:
        line = self._stream.readline(size)
        if line:
            self._progress.update(len(line.encode(self._encoding, errors="replace")))
        return line

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)


def build_dataset(
    source: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    source_description: str = "",
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    split_seed: int = 0,
    filters: Mapping[str, object] | None = None,
    show_progress: bool = False,
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

    progress: _ProgressBar | None = None
    source_size = source.stat().st_size if show_progress else 0

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
        sample_stream: TextIO = src
        if show_progress:
            progress = tqdm(
                total=source_size,
                desc=f"processing {source.name}",
                unit="B",
                unit_scale=True,
                dynamic_ncols=True,
            )
            sample_stream = cast(
                TextIO,
                _ProgressTextStream(src, progress, src.encoding or "utf-8"),
            )

        try:
            samples_written = 0
            completed = False
            for sample in iter_samples(sample_stream, counters, filters=filters):
                gid = sample["game_id"]
                if gid not in splits:
                    splits[gid] = pick_split()
                split = splits[gid]
                dataset_sample: DatasetSample = {**sample, "split": split}
                shards[split].write(json.dumps(dataset_sample) + "\n")
                pos_per_split[split] += 1
                samples_written += 1

                if progress is not None and (
                    samples_written == 1 or samples_written % 1000 == 0
                ):
                    progress.set_postfix(
                        games=counters["games_read"],
                        used=counters["games_used"],
                        skipped=(
                            counters["games_skipped_corrupt"]
                            + counters["games_skipped_unknown_result"]
                            + counters["games_skipped_filter"]
                        ),
                        positions=counters["positions_emitted"],
                    )
            completed = True
        finally:
            if progress is not None:
                if completed and progress.n < source_size:
                    progress.update(source_size - progress.n)
                progress.close()

    manifest = {
        "source": str(source),
        "source_description": source_description,
        "source_checksum": hashlib.sha256(source.read_bytes()).hexdigest(),
        "num_games_raw": counters["games_read"],
        "num_games_used": counters["games_used"],
        "num_games_skipped": (
            counters["games_skipped_corrupt"]
            + counters["games_skipped_unknown_result"]
            + counters["games_skipped_filter"]
        ),
        "num_games_skipped_corrupt": counters["games_skipped_corrupt"],
        "num_games_skipped_unknown_result": counters["games_skipped_unknown_result"],
        "num_games_skipped_filter": counters["games_skipped_filter"],
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
