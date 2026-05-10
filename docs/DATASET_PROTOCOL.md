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
