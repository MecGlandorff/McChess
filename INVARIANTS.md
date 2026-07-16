# Invariants

This file records technical contracts that should not drift silently.

If an implementation changes one of these contracts, update this file, `DESIGN.md`, tests, and any affected configs in the same change.

## Chess Rules

- Legal move generation must come from `python-chess`.
- The neural network must not be trusted to learn legality.
- Illegal moves must be masked before move selection and before MCTS expansion.
- Stockfish may be used only as an external evaluation opponent. Its moves,
  evaluations, and game outcomes must not be used for training supervision,
  distillation, checkpoint selection, or hyperparameter tuning.
- Syzygy tablebases, Leela labels, tablebase labels, and external engine move
  recommendations are not allowed.

## Board Encoding

- Single-board tensors use shape `[planes, 8, 8]`.
- The current single-board encoder uses shape `[18, 8, 8]`.
- Batched single-board tensors use shape `[batch, planes, 8, 8]`.
- Plane dtype is `float32`.
- Board orientation is row 0 = rank 8, row 7 = rank 1, column 0 = file a, column 7 = file h.
- The exact plane order must be documented in `DESIGN.md`.
- Side-to-move metadata must be explicit in the tensor representation.
- Castling-right metadata must be explicit if the current encoder uses it.
- En-passant metadata must be explicit in the current tensor representation.
- The current en-passant plane marks the legal target square only when
  `python-chess` reports that an en-passant capture is legal; stale FEN
  en-passant squares are encoded as all zeros.
- Halfmove and fullmove metadata must be documented if added.

## Value Convention

Value targets and value predictions are from the side-to-move perspective at the current position.

- side to move eventually wins: `+1`
- side to move eventually loses: `-1`
- draw: `0`

The value head output must be a scalar per position and must be bounded to `[-1, 1]`.

## Move Encoding

- The implemented policy space is `8 x 8 x 73 = 4672`.
- Policy index formula is `from_square * 73 + move_plane`.
- `from_square` uses the `python-chess` square index, where `a1 = 0` and `h8 = 63`.
- Policy logits use shape `[batch, 4672]`.
- Legal masks use shape `[4672]` for one board and `[batch, 4672]` only when explicitly batched.
- Legal masks use dtype `float32`.
- Legal masks should mark legal indices only, using `1.0` for legal moves and `0.0` otherwise.
- `move_to_index(board, move)` and `index_to_move(board, index)` must round trip every legal move in tested positions.
- `move_to_index(board, move)` must reject illegal moves.
- `index_to_move(board, index)` must return `None` for out-of-range indices or indices that do not decode to legal moves.
- Queen promotions are encoded as queen-like moves.
- Knight, bishop, and rook underpromotions use underpromotion planes.
- Castling and en passant are encoded by ordinary move geometry.
- Promotions, castling, and en passant must be covered by tests.

## Model Output

All model families must return:

```text
policy_logits: [batch, 4672]
value: [batch]
```

Outputs must be finite for normal inputs.

Future NNUE-style architectures are subject to the same output contract unless
they are explicitly documented as a separate value-only experimental bot. They
must not use imported engine weights, engine evaluations, tablebase labels, or
external best-move labels.

## MCTS

- MCTS may expand legal moves only.
- Policy priors must be generated from legally masked policy logits.
- Leaf values are interpreted from the side-to-move perspective at the leaf.
- Backup must flip the value sign at every ply.
- Terminal states must not be evaluated as ordinary nonterminal leaves.
- Non-default batched MCTS inference must be recorded as part of the evaluation
  protocol because virtual visit reservations can change search trajectories.

## Dataset Splits

- Split by game, not by position.
- Positions from one game must not appear in multiple splits.
- Dataset manifests must include source description, filters, counts, split sizes, and code version if available.
- Executable PGN filters must record skipped-by-filter game counts separately
  from corrupt and unknown-result skips.
- Header-only prefilters may reduce large PGN archives by metadata, but they
  must not replace full legal-move validation in the dataset builder.
- Optional tensor caches are derived artifacts. They must be rebuildable from
  JSONL shards and must record schema version, sample count, tensor shape, and
  dtypes. A board encoding shape or plane semantics change invalidates existing
  tensor caches.
- Tensor cache readers must require `manifest.json` as the completed-cache
  marker. In-progress `*.tmp` arrays and `progress.json` may be used only by
  cache builders to resume interrupted builds.

## Evaluation

- Reported arena results must include opponent, checkpoint, seed, number of games, color policy, max ply, draw rules, and MCTS budget.
- Illegal move count must be zero.
- Do not claim Elo unless measured by a documented rating protocol.
