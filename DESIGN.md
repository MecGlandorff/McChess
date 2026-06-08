# Design

## System Overview

```text
PGN games
  -> dataset builder
  -> board tensors, policy targets, value targets
  -> policy/value model
  -> policy-only bot
  -> MCTS-enhanced bot
  -> arena evaluation
```

## Contract Source

`INVARIANTS.md` is the source of truth for implementation contracts that should not drift silently.

When changing board tensors, move indexing, value perspective, model outputs, dataset splits, or MCTS semantics, update:

- the implementation
- tests
- this design document
- `INVARIANTS.md`
- any affected config or protocol docs

## Board Representation

Implemented encoder: single-position tensor with shape:

```text
[18, 8, 8]
```

Implemented plane order:

1. white pawns
2. white knights
3. white bishops
4. white rooks
5. white queens
6. white king
7. black pawns
8. black knights
9. black bishops
10. black rooks
11. black queens
12. black king
13. white to move
14. white kingside castling right
15. white queenside castling right
16. black kingside castling right
17. black queenside castling right
18. en-passant target square

Future metadata planes may include:

- halfmove clock, normalized
- fullmove number, normalized or clipped

Piece planes are one-hot occupancy planes.

Side-to-move and castling metadata planes are constant planes filled with `1.0`
when true and `0.0` when false.

The en-passant target plane is sparse. It is all zeros unless
`python-chess` reports that an en-passant capture is legal in the current
position. When legal, the plane contains `1.0` only on the en-passant target
square. For example, after `1. e4 a6 2. e5 d5`, White can play `exd6 e.p.`, so
the target square `d6` is marked. A FEN en-passant square that does not permit a
legal en-passant capture is not encoded.

Rationale:

- Including legal en-passant availability removes an input alias where two
  positions with identical pieces, castling rights, and side to move can have
  different legal policy targets.
- The legal policy mask remains the source of truth for move legality; the
  neural network is not responsible for deciding whether en passant is legal.

Caveats:

- This changes the model input shape, so models trained with the older
  17-plane encoder would not be checkpoint-compatible.
- The plane encodes current legal en-passant availability only. It does not
  encode longer history, repetition state, the halfmove clock, or the fullmove
  number.

Square orientation:

- tensor row 0 is rank 8
- tensor row 7 is rank 1
- tensor column 0 is file a
- tensor column 7 is file h

Examples:

- `a8 -> (0, 0)`
- `h8 -> (0, 7)`
- `a1 -> (7, 0)`
- `h1 -> (7, 7)`

The encoder implementation must also document square orientation. Tests should verify that pieces land on expected squares, not only that piece counts match.

## Value Convention

The value target is always from the side-to-move perspective at the current position.

- side to move eventually wins: `+1`
- side to move eventually loses: `-1`
- draw: `0`

For example:

- White to move and White wins: `+1`
- Black to move and White wins: `-1`
- White to move and Black wins: `-1`
- Black to move and Black wins: `+1`

## Move Encoding

Implemented policy space: `8 x 8 x 73 = 4672`.

Policy index formula:

```text
index = from_square * 73 + move_plane
```

`from_square` uses the `python-chess` square index:

- `a1 = 0`
- `h1 = 7`
- `a8 = 56`
- `h8 = 63`

Move planes:

```text
0-55   queen-like moves: 8 directions x distances 1-7
56-63  knight moves: 8 possible offsets
64-72  underpromotions: knight, bishop, rook x forward, left, right
```

Queen-like direction order:

1. north `(0, +1)`
2. north-east `(+1, +1)`
3. east `(+1, 0)`
4. south-east `(+1, -1)`
5. south `(0, -1)`
6. south-west `(-1, -1)`
7. west `(-1, 0)`
8. north-west `(-1, +1)`

Knight direction order:

1. `(+1, +2)`
2. `(+2, +1)`
3. `(+2, -1)`
4. `(+1, -2)`
5. `(-1, -2)`
6. `(-2, -1)`
7. `(-2, +1)`
8. `(-1, +2)`

Queen promotions are encoded as ordinary queen-like pawn moves.

Underpromotion planes encode only knight, bishop, and rook promotions. Promotion directions are from the moving side's perspective:

- forward
- left capture
- right capture

Castling is encoded as the king's ordinary two-square horizontal move.

En passant is encoded as the pawn's ordinary diagonal capture geometry.

The move-indexing layer must provide:

```python
move_to_index(board, move) -> int
index_to_move(board, index) -> chess.Move | None
legal_policy_mask(board) -> np.ndarray
```

`move_to_index` raises `ValueError` for illegal or unrepresentable moves.

`index_to_move` returns `None` when the index is out of range or does not decode to a legal move in the current position.

`legal_policy_mask` returns a `float32` array of shape `[4672]` with `1.0` at legal move indices.

Legal masking must use python-chess.

The neural model should never be responsible for deciding legality.

## Training Targets

Supervised learning sample:

- input: board tensor
- policy target: move played in human game
- value target: final result from side-to-move perspective

Search-distillation sample:

- input: board tensor
- policy target: MCTS visit distribution
- value target: final game result or MCTS value target

Self-play sample:

- input: board tensor
- policy target: MCTS visit distribution
- value target: final self-play result

## Training Artifacts

Supervised training run directories contain the copied config, `metrics.jsonl`,
`batch_metrics.jsonl`, `status.json`, checkpoints, `loss.svg`, and
`batch_loss.svg`.

`metrics.jsonl` records one row per completed epoch, including train and
validation policy/value/total losses. `batch_metrics.jsonl` records train-only
losses at the configured `log_every_steps` interval, including the latest batch
loss and the running average within the current epoch. Full validation still
runs at epoch boundaries by default.

Checkpoint files use PyTorch serialization and contain:

- `model_state_dict`
- `model_config`
- `train_config`
- `epoch`
- `metrics`
- `saved_at`
- `completed_at`

The trainer refreshes `batch_loss.svg` while an epoch is running. It writes
`checkpoint_epoch_###.pt`, `checkpoint_latest.pt`, and `loss.svg` after each
completed epoch. `checkpoint.pt` is the final completed-run checkpoint and is
written only after all configured epochs finish.

## Playable Bots

Initial playable agents use a small `choose_move(board)` bot interface.

- random legal-move bot
- one-ply material bot
- policy-only checkpoint bot

The policy-only bot loads a supervised `PolicyValueResNet` checkpoint, encodes
the current board, runs the policy head, masks illegal moves with
`legal_policy_mask(board)`, and chooses the highest-logit legal move. It does
not use MCTS or the value head for move selection.

The notebook play helper provides a click-source, click-target board for local
inspection. It is an interactive debugging aid, not an arena evaluation or
strength result.

## Model Families

Baseline:

- `PolicyValueResNet`: compact single-board ResNet policy/value network.
  Current input shape is `[batch, 18, 8, 8]`; outputs are
  `policy_logits: [batch, 4672]` and `value: [batch]`. No normalization layers
  are used in the current baseline. This keeps the first supervised ResNet
  minimal and makes later normalization changes measurable as matched
  ablations. BatchNorm, GroupNorm, or other normalization variants should be
  compared explicitly rather than assumed as part of the baseline.
- Named ResNet presets currently include `resnet_baseline` and `resnet_b`.

Ablations:

- History ResNet
- ResNet + square attention
- LSTM over board history
- LSTM + temporal attention
- Temporal Transformer

Future optional:

- NNUE-style sparse accumulator. If added, this is a McChess-defined neural
  architecture, not imported Stockfish NNUE. It must be trained only from
  allowed project targets, keep explicit legal move masking, and preserve the
  model output contract:

```text
policy_logits: [batch, 4672]
value: [batch]
```

An NNUE-style implementation would need a documented sparse feature schema and
tests showing that any incremental accumulator state matches full feature
recomputation. A value-only NNUE scorer would be a separate experimental bot
type and must not silently replace the policy/value model contract.

## MCTS

Use PUCT-style selection:

```text
score = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Requirements:

- expand legal moves only
- mask illegal policy logits
- evaluate leaves with value head
- flip value sign at each ply during backup
- support fixed simulation budget
- handle terminal positions without ordinary network evaluation

## Dataset Pipeline

`src/mcchess/data/pgn_reader.py` streams a PGN file and yields one sample
per played move. Games with unknown result (`*`) or any parse/legality
error are skipped and counted; no partial samples are emitted.

`src/mcchess/data/dataset_builder.py` calls the reader, assigns each new
`game_id` a split via a seeded `random.Random`, and writes
`train.jsonl`, `val.jsonl`, `test.jsonl` plus a manifest JSON.

JSONL sample fields match `docs/DATASET_PROTOCOL.md`:

```json
{
  "game_id": "g000000",
  "ply": 0,
  "fen": "rnbqkbnr/pppppppp/...",
  "move_uci": "e2e4",
  "policy_index": 748,
  "value": 0.0,
  "result": "1/2-1/2",
  "split": "train"
}
```

The manifest written next to the shards records source, sha256 checksum,
raw/used/skipped game counts (with separate corrupt and unknown-result
counters), position counts per split, the configured ratios and seed,
filters, `created_at`, `code_version`, and a `schema_version` integer.

The dataset builder does not encode board tensors to disk; downstream
training code is expected to decode the FEN and call `encode_board`. This
keeps shards portable and decoupled from any encoder revisions.

For throughput-sensitive local CUDA runs, a JSONL shard may be converted into
an optional tensor cache documented in `docs/DATASET_PROTOCOL.md`. The JSONL
shard remains the source of truth; the cache stores encoded `uint8` board planes
and policy/value targets so training can avoid per-sample FEN parsing in the
hot path.

## Evaluation

Evaluation should use:

- fixed random seed
- alternating colors
- fixed number of games
- max ply limit
- same opening positions when comparing models
- fixed node budgets for MCTS
- win/draw/loss table
- illegal move count
- checkpoint and config identifiers
- runtime or speed metrics when practical

Do not claim Elo strength without a careful protocol.
