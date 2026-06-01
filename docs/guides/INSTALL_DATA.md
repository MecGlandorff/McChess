# Installing The First Lichess Dataset

This guide creates the first real McChess supervised dataset from the Lichess
open database.

Source:

- https://database.lichess.org/
- Dataset: standard rated games
- License: CC0

Important: Lichess PGNs may contain Stockfish eval comments such as
`[%eval ...]`. McChess must not use those as labels. The dataset path below
uses only human moves, game metadata, and final game results.

## Prerequisites

Install project dependencies:

```bash
poetry install --with dev,notebook
```

Install `zstd` for decompressing `.pgn.zst` archives:

```bash
brew install zstd
```

## Download The Raw Archive

Start with the small January 2013 Lichess standard rated archive:

```bash
poetry run python scripts/download_lichess.py 2013-01 \
  --output-dir data/raw/lichess \
  --manifest data/manifests/lichess_downloads.jsonl \
  --checksum
```

This writes:

- `data/raw/lichess/lichess_db_standard_rated_2013-01.pgn.zst`
- `data/raw/lichess/lichess_db_standard_rated_2013-01.pgn.zst.json`
- `data/manifests/lichess_downloads.jsonl`

## Decompress The PGN

```bash
zstd -d data/raw/lichess/lichess_db_standard_rated_2013-01.pgn.zst \
  -o data/raw/lichess/lichess_db_standard_rated_2013-01.pgn
```

## Create A 1,000-Game Sample

Use a bounded sample first so local training and smoke tests stay manageable:

```bash
poetry run python -c '
from pathlib import Path
import chess.pgn

src = Path("data/raw/lichess/lichess_db_standard_rated_2013-01.pgn")
dst = Path("data/raw/lichess/lichess_db_standard_rated_2013-01.sample1000.pgn")
limit = 1000
count = 0

with src.open(encoding="utf-8", errors="replace") as inp, dst.open("w", encoding="utf-8") as out:
    while count < limit:
        game = chess.pgn.read_game(inp)
        if game is None:
            break
        print(game, file=out, end="\n\n")
        count += 1

print(f"wrote {count} games to {dst}")
'
```

## Build The Processed Dataset

```bash
poetry run python -c '
from mcchess.data.dataset_builder import build_dataset

build_dataset(
    source="data/raw/lichess/lichess_db_standard_rated_2013-01.sample1000.pgn",
    output_dir="data/processed/lichess_2013_01_sample1000",
    manifest_path="data/manifests/lichess_2013_01_sample1000_manifest.json",
    source_description=(
        "First 1000 games from Lichess standard rated 2013-01 PGN archive; "
        "human moves/results only; PGN comments ignored by parser."
    ),
    split_ratios=(0.9, 0.05, 0.05),
    split_seed=20260601,
    filters={
        "source": "lichess standard rated",
        "month": "2013-01",
        "sample": "first_1000_games",
        "engine_labels_used": False,
        "pgn_comments_used": False,
    },
)
'
```

Expected outputs:

- `data/processed/lichess_2013_01_sample1000/train.jsonl`
- `data/processed/lichess_2013_01_sample1000/val.jsonl`
- `data/processed/lichess_2013_01_sample1000/test.jsonl`
- `data/manifests/lichess_2013_01_sample1000_manifest.json`

The first local sample created from this guide should contain about 1,000 games
and 61,000 supervised positions.

## Smoke Check The Loader

```bash
poetry run python -c '
from mcchess.data.torch_dataset import SupervisedChessDataset

dataset = SupervisedChessDataset("data/processed/lichess_2013_01_sample1000/train.jsonl")
sample = dataset[0]

print(len(dataset))
print(tuple(sample["board"].shape), sample["board"].dtype)
print(sample["policy_index"].item(), sample["value"].item())
'
```

Expected tensor shape:

```text
(18, 8, 8)
```

## Git Tracking

Raw data, processed data, manifests, and checkpoints are local artifacts and are
ignored by git. Keep the commands and manifests for reproducibility; do not
commit large dataset files.
