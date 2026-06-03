# Dataset Protocol

## Allowed Data

Allowed:

- human PGN games
- game metadata
- final game result

Not allowed:

- Stockfish analysis
- engine evaluations
- tablebase labels
- external best-move labels

## Provenance

Every raw dataset should have a provenance note:

- source path or URL
- source description
- acquisition date
- license or terms note when available
- file checksum when practical
- filter assumptions

Do not mix raw data sources without recording which source each processed sample came from or which manifest produced it.

## PGN Filtering

Recommended filters:

- rated games only
- exclude games with illegal/corrupt moves
- optionally filter by player rating
- optionally exclude ultra-bullet
- optionally include only rapid/classical

Filtering choices must be saved in the dataset manifest. If the project uses rating, time-control, date, termination, or variant filters, record exact thresholds and skipped-game counters.

## Splitting

Split by game, not by position.

Reason:

Positions from the same game are highly correlated. Splitting by position causes train/validation leakage.

Recommended split:

- train: 98%
- val: 1%
- test: 1%

For small experiments:

- train: 90%
- val: 5%
- test: 5%

The split assignment should be reproducible from a seed and saved to disk when practical.

Never split by position.

## Duplicate Handling

Record whether duplicate games are:

- kept
- removed by exact PGN text
- removed by normalized move sequence
- removed by another documented rule

If duplicate removal is implemented, save duplicate counts in the manifest.

## Supervised Sample

For each move in a game:

- board before move
- played move
- final result
- side to move
- metadata

Targets:

- policy target = `move_to_index(board, played_move)`
- value target = final result from side-to-move perspective

Recommended serialized fields:

```json
{
  "game_id": "",
  "ply": 0,
  "fen": "",
  "move_uci": "",
  "policy_index": 0,
  "value": 0.0,
  "result": "",
  "split": ""
}
```

## PyTorch Loader

`SupervisedChessDataset` reads one JSONL shard, decodes each `fen` with
`python-chess`, calls `encode_board`, and returns model-ready tensors:

```text
board: [18, 8, 8] float32
policy_index: scalar int64
value: scalar float32
```

The loader does not store encoded tensors on disk. FEN remains the serialized
dataset source of truth.

## Optional Tensor Cache

For larger CUDA runs, JSONL shards may be converted into a local tensor cache
with:

```powershell
poetry run python scripts\build_tensor_cache.py data/processed/name/train.jsonl data/tensor_cache/name/train
```

The cache is an acceleration artifact, not a replacement source of truth. It
must be rebuildable from the JSONL shard and current encoder implementation.

Cache files:

```text
boards.npy          uint8   [num_samples, 18, 8, 8]
policy_indices.npy  int64   [num_samples]
values.npy          float32 [num_samples]
manifest.json
```

Boards are stored as `uint8` because the current 18 documented planes are
binary. Training casts cached boards to `float32` on the selected device before
model inference. If board planes ever become non-binary, this cache format must
be revised with a new schema version.

Training configs may set:

```yaml
train_cache_dir: data/tensor_cache/name/train
val_cache_dir: data/tensor_cache/name/val
```

When cache directories are configured, training reads cached tensors instead of
decoding FENs in the hot path.

Cache builders must write array files through temporary paths and write
`manifest.json` last, after all arrays are complete. Cache readers must reject
missing files, unsupported schema versions, and unexpected shapes or dtypes
before using cached tensors.

## Manifest

Every processed dataset should save a manifest:

```json
{
  "source": "example.pgn",
  "source_description": "",
  "source_checksum": "",
  "num_games_raw": 0,
  "num_games_used": 0,
  "num_games_skipped": 0,
  "num_games_skipped_corrupt": 0,
  "num_games_skipped_unknown_result": 0,
  "num_duplicate_games": 0,
  "num_positions": 0,
  "filters": {},
  "split": {},
  "split_seed": 0,
  "created_at": "",
  "code_version": "",
  "schema_version": 1
}
```

## Data Quality Counters

Track:

- games read
- games skipped
- unknown result games
- illegal move errors
- positions emitted
- checkmates
- draws
- average game length

## Dataset Acceptance Gate

A dataset builder change is not complete unless tests cover:

- tiny valid PGN
- unknown result handling
- corrupt or illegal game skipping
- game-level split behavior
- side-to-move value perspective
- policy target index generation

If a dataset artifact is produced manually, record the command and manifest path in the relevant experiment notes.
