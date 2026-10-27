# Design

```text
human PGNs -> JSONL shards + manifest -> board tensors and targets
           -> supervised policy/value ResNet -> policy-only or MCTS bot
           -> arena and held-out evaluation
```

[INVARIANTS.md](INVARIANTS.md) is the contract source. Changes to encodings,
value perspective, model outputs, splits, or search semantics require matching
implementation, tests, and documentation changes.

## Board Encoding

`encode_board` returns a `float32` tensor of shape `[18, 8, 8]`.

| Plane (1-based) | Meaning |
|---|---|
| 1-6 | White pawn, knight, bishop, rook, queen, king occupancy |
| 7-12 | Black pawn, knight, bishop, rook, queen, king occupancy |
| 13 | White to move |
| 14-15 | White kingside, queenside castling rights |
| 16-17 | Black kingside, queenside castling rights |
| 18 | Legal en-passant target square |

Occupancy and en-passant planes are sparse binary planes. Side-to-move and
castling planes are filled with `1.0` when true and `0.0` otherwise.
The en-passant plane is zero unless `python-chess` reports a legal en-passant
capture. A stale FEN target is not encoded; after `1. e4 a6 2. e5 d5`, `d6`
is marked. This distinguishes positions with different legal pawn captures
without making the model responsible for legality.

Rows run rank 8 to rank 1; columns run file a to file h:
`a8=(0,0)`, `h8=(0,7)`, `a1=(7,0)`, `h1=(7,7)`.
The encoder has no history, repetition state, halfmove clock, or fullmove
number. Older 17-plane checkpoints are incompatible.

## Move Indexing And Legality

The policy space is `8 * 8 * 73 = 4672`:

```text
index = from_square * 73 + move_plane
```

`from_square` follows `python-chess`: `a1=0`, `h1=7`, `a8=56`, `h8=63`.
It does not use the tensor row orientation.

| Planes | Geometry |
|---|---|
| 0-55 | Eight queen-like directions, each with distances 1-7 |
| 56-63 | Eight knight offsets |
| 64-72 | Knight, bishop, rook underpromotions; each with forward, left, right |

Queen-like direction order in `(file, rank)` offsets:
`(0,+1), (+1,+1), (+1,0), (+1,-1), (0,-1), (-1,-1), (-1,0), (-1,+1)`.

Knight offset order:
`(+1,+2), (+2,+1), (+2,-1), (+1,-2), (-1,-2), (-2,-1), (-2,+1), (-1,+2)`.

Underpromotion directions are from the moving side's perspective. Queen
promotions use ordinary queen-like pawn geometry. Castling uses the king's
two-square horizontal move; en passant uses the pawn's diagonal capture.

| API | Contract |
|---|---|
| `move_to_index(board, move)` | Integer index; raises `ValueError` for illegal or unrepresentable moves. |
| `index_to_move(board, index)` | Legal `chess.Move`, or `None` for out-of-range or illegal indices. |
| `legal_policy_mask(board)` | `float32 [4672]`, with `1.0` at legal indices and `0.0` elsewhere. |
| `legal_moves_with_policy_indices(board)` | Legal moves and indices computed in one pass for masking and expansion. |

All legality comes from `python-chess`. Mask illegal actions before move
selection and MCTS expansion.

## Targets And Model

Supervised policy targets are the played human move. Value targets are the
final result from the current side-to-move perspective: win `+1`, loss `-1`,
draw `0`. For a White win, White-to-move positions have target `+1` and
Black-to-move positions have target `-1`.

The current model is `PolicyValueResNet`:

```text
input:         [batch, 18, 8, 8]
policy_logits: [batch, 4672]
value:         [batch], bounded to [-1, 1]
```

[ARCHITECTURES.md](docs/ARCHITECTURES.md) describes the presets and deferred
families. Future distillation would use McChess visit distributions; future
self-play would use McChess game results. Neither pipeline is implemented.

## Data And Training Artifacts

The PGN reader emits one sample per played move, rejects corrupt/illegal or
unknown-result games without partial samples, and counts rejected games.
The builder assigns splits by game using a seeded RNG, writes
`train.jsonl`, `val.jsonl`, `test.jsonl`, and a manifest with source checksum,
filters, counts, split settings, creation time, and code version.

JSONL FEN records remain the source of truth. Optional tensor caches avoid
repeated parsing and encoding during training; shapes, dtypes, completion
markers, and provenance are specified in the
[dataset protocol](docs/DATASET_PROTOCOL.md).

Training saves copied config, epoch and batch metrics, status, loss plots, and
checkpoints. Current training checkpoints include model/optimizer state, model
and train configs, epoch, global step, metrics, and timestamps. With
`resume_from_checkpoint`, `epochs` is the final target epoch. Older checkpoints
without optimizer state warm-start weights with a reset optimizer.

Checkpoint lifecycle and inference-only export are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Bots And MCTS

Bots expose `choose_move(board)`. Baselines are random, one-ply material, and
negamax. Policy-only play selects the highest-logit legal move and does not use
the value head. MCTS uses legal priors and side-to-move leaf values:

```text
PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Search evaluates the root, runs a fixed number of simulations, and selects a
root move by visits, then mean value and prior. Values change sign at every
backup ply. Terminal outcomes use game rules rather than neural evaluation.
Board copies retain move history for repetition and claimable draws.

Model calls use `torch.inference_mode()`; checkpoint loading also sets eval
mode. Direct `MCTSSearch` callers must provide an eval-mode model themselves.
Legal moves are indexed once per expansion, softmax covers only legal logits,
and priors/values are transferred to the host together.

The default inference batch size is 1. Larger batches reserve several paths
before inference. Paths reaching one pending node share a neural evaluation;
simulation count is therefore distinct from model calls or distinct evaluated
positions. Reservations can change search trajectories, so the batch size is
part of the evaluation protocol. Each move currently starts a fresh tree.

## Evaluation

The arena alternates colors with the named agent White first and reports
W/D/L from that agent's perspective. Nonterminal games at `max_ply` are draws.
Saved games include colors, result, termination, ply count, final FEN, moves,
and search budgets. Fixed opening lists pair adjacent games with colors swapped.

Held-out supervised evaluation reconstructs JSONL FENs for legal top-k
metrics. External Stockfish games are evaluation-only. Protocol, budget,
opening, and reporting requirements are in
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
