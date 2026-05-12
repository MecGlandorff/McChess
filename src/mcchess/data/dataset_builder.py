"""
Build a supervised dataset from a PGN file. Writes JSONL shards for
train/val/test and a JSON manifest. Splits are by game (not by position),
seeded, and recorded in the manifest.

Schema and required manifest fields come from `docs/DATASET_PROTOCOL.md`.
"""
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import mcchess
from mcchess.data.pgn_reader import iter_samples, new_counters

SCHEMA_VERSION = 1


def build_dataset(
    source,
    output_dir,
    manifest_path,
    *,
    source_description="",
    split_ratios=(0.9, 0.05, 0.05),
    split_seed=0,
    filters=None,
):
    source = Path(source)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(split_seed)
    train_p, val_p, _ = split_ratios

    def pick_split():
        r = rng.random()
        if r < train_p:
            return "train"
        if r < train_p + val_p:
            return "val"
        return "test"

    splits = {}                                          # game_id -> split
    pos_per_split = {"train": 0, "val": 0, "test": 0}
    counters = new_counters()

    with open(source) as src, \
         open(output_dir / "train.jsonl", "w") as t, \
         open(output_dir / "val.jsonl", "w") as v, \
         open(output_dir / "test.jsonl", "w") as e:
        shards = {"train": t, "val": v, "test": e}
        for s in iter_samples(src, counters):
            gid = s["game_id"]
            if gid not in splits:
                splits[gid] = pick_split()
            s["split"] = splits[gid]
            shards[splits[gid]].write(json.dumps(s) + "\n")
            pos_per_split[splits[gid]] += 1

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
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
