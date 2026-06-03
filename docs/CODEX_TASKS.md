# Codex Tasks

Use one task per Codex session.

## General Instruction To Prepend For Nontrivial Tasks

Read `AGENTS.md`, `PROJECT.md`, `ROADMAP.md`, and the relevant design docs before editing.

Before editing, inspect the repository and propose a short plan.

Implement only the smallest useful version of the requested task.

Do not implement future milestones.

Run relevant tests through Poetry before claiming completion.

## Acceptance Criteria Template

Each Codex task should end with:

- changed files listed
- tests added or updated
- docs updated if contracts, shapes, configs, or protocols changed
- commands run and their result
- artifacts produced, if any
- limitations or intentionally deferred work stated

Do not mark a task complete only because code was written.

## Task 0 - Repository Skeleton

Set up the initial Python repository skeleton for McChess.

Create:

- `pyproject.toml`
- `README.md`
- package structure under `src/mcchess`
- `tests` directory
- `configs` directory
- placeholder docs if missing

Use Python 3.11+, pytest, python-chess, torch, numpy, PyYAML, ruff, and mypy.

Do not implement chess logic yet.

Make the package importable.

Add a minimal smoke test.

Run `poetry run pytest`.

Acceptance criteria:

- package imports from `src`
- `poetry run pytest` passes
- no chess logic is implemented yet

## Task 1 - Board Encoding

Implement the first version of board tensor encoding.

Requirements:

- Add `src/mcchess/board/encoding.py`
- Use python-chess `Board` objects.
- Encode board into a NumPy `float32` tensor.
- Use 12 piece planes.
- Add side-to-move plane.
- Add castling-right planes.
- Document exact plane order in `DESIGN.md`.
- Add tests in `tests/test_board_encoding.py`.

Tests:

- initial position shape
- initial piece counts
- side-to-move plane changes after a move
- castling rights update after king move

Do not implement move indexing yet.

Run `poetry run pytest`.

Acceptance criteria:

- tests verify shape, piece counts, square orientation, side to move, and castling rights
- `DESIGN.md` and `INVARIANTS.md` document the plane contract
- no move-indexing code is introduced

## Task 2 - Move Indexing And Legal Mask

Implement move indexing and legal policy masking.

Requirements:

- Add `src/mcchess/board/move_index.py`
- Add `src/mcchess/board/legal_mask.py`
- Use fixed policy size `8x8x73 = 4672` if practical.
- Implement:
  - `move_to_index(board, move) -> int`
  - `index_to_move(board, index) -> chess.Move | None`
  - `legal_policy_mask(board) -> np.ndarray`
- Mask shape must be `(4672,)`.
- Handle normal moves, captures, promotions, castling, and en passant.
- Add round-trip tests over many generated legal positions.
- Update `DESIGN.md`.

Do not use Stockfish or engine evaluation.

Run `poetry run pytest`.

Acceptance criteria:

- generated legal positions round trip through `move_to_index` and `index_to_move`
- legal mask has shape `(4672,)`
- castling, promotion, and en passant tests pass
- `DESIGN.md` and `INVARIANTS.md` document the policy contract

## Task 3 - PGN Dataset Builder

Implement PGN parsing into supervised samples.

Requirements:

- Add `src/mcchess/data/pgn_reader.py`
- Add `src/mcchess/data/dataset_builder.py`
- Parse PGN files with python-chess.
- For each move, emit:
  - board FEN before move
  - policy target index
  - value target from side-to-move perspective
  - game result metadata
- Split by game, not by position.
- Skip corrupt games with counters.
- Write dataset manifest JSON.
- Add tests using tiny inline PGN strings.
- Update `DESIGN.md` and `docs/DATASET_PROTOCOL.md` if needed.

Do not implement training yet.

Run `poetry run pytest`.

Acceptance criteria:

- fixture PGN emits expected positions and targets
- corrupt or unknown-result games are counted
- split is by game
- manifest contains provenance, counts, filters, split, and code version if available

## Task 4 - Model Baseline

Implement a small PyTorch policy/value ResNet.

Requirements:

- Add `src/mcchess/model/network.py`
- Add `src/mcchess/model/loss.py`
- Policy output size: `4672`
- Value output: scalar in `[-1, 1]`
- Combined loss:
  - cross entropy for policy
  - MSE for value
  - configurable value weight
- Add shape tests.
- Add forward/backward smoke test.

Do not add attention, LSTM, MCTS, or self-play yet.

Run `poetry run pytest`.

Acceptance criteria:

- policy logits have shape `[batch, 4672]`
- value has shape `[batch]` and is bounded to `[-1, 1]`
- forward/backward smoke test passes
- tiny overfit or loss-decrease smoke test exists when training code is available

## Task 5 - Training Script

Implement a minimal supervised training script.

Requirements:

- Add `scripts/train_supervised.py`
- Load YAML config.
- Train policy/value model.
- Save:
  - config copy
  - `metrics.jsonl`
  - `checkpoint.pt`
- Support CPU and CUDA.
- Set deterministic seeds.
- Log policy loss, value loss, and total loss.
- Add a tiny overfit or smoke test.

Keep implementation simple.

Run `poetry run pytest`.

Acceptance criteria:

- run directory contains config copy, metrics, checkpoint, and status
- deterministic seed is set
- checkpoint can be loaded after saving
- tiny training smoke test passes

## Task 6 - Baseline Bots

Implement baseline chess bots.

Requirements:

- Add bot base interface.
- Add random legal move bot.
- Add material-count bot.
- Add shallow minimax bot.
- Add policy-only neural bot if checkpoint loading exists.
- All bots expose `choose_move(board)`.
- Tests verify returned moves are legal.

Do not implement MCTS yet.

Run `poetry run pytest`.

Acceptance criteria:

- every bot returns legal moves across tested positions
- seeded bots are deterministic when expected
- no MCTS code is introduced

## Task 7 - Arena

Implement arena evaluation.

Requirements:

- Bot-vs-bot matches.
- Alternating colors.
- Fixed seed.
- Max ply limit.
- Results JSON.
- Win/draw/loss counts.
- Tests with random-vs-random and material-vs-random smoke matches.

Do not claim Elo.

Run `poetry run pytest`.

Acceptance criteria:

- arena output includes W/D/L, score, seed, max ply, draw rule, and illegal move count
- baseline smoke matches complete
- no Elo claims are introduced

## Task 8 - MCTS

Implement simple PUCT MCTS.

Requirements:

- Add search node and MCTS implementation.
- Use policy priors masked to legal moves.
- Use value head for leaf evaluation.
- Flip value sign during backup at each ply.
- Add MCTS bot.
- Fixed simulations per move.
- Tests:
  - legal move returned
  - terminal handling
  - backup sign convention
  - visit counts update

Do not add transposition tables yet.

Run `poetry run pytest`.

Acceptance criteria:

- MCTS returns legal moves
- backup sign flip is tested
- terminal handling is tested
- visit counts update and are inspectable

## Task 9 - History Encoding

Add multi-position history encoding.

Requirements:

- Keep single-board encoder unchanged.
- Add `encode_board_history`.
- Accept boards ordered current-first.
- Support configurable `history_length`.
- Encode 12 piece planes per board.
- Pad missing history positions with zero planes.
- Keep value perspective anchored to the current board.
- Add tests for shape, padding, and current-board consistency.
- Update `DESIGN.md`.

Do not change model architecture yet.

Run `poetry run pytest`.

Acceptance criteria:

- history encoding shape and padding are tested
- current-position perspective is preserved
- `DESIGN.md` and `INVARIANTS.md` document history shape
- no architecture changes are introduced

## Task 10 - LSTM History Architecture

Add optional LSTM-over-board-history architecture.

Requirements:

- Architecture name: `lstm_history`.
- Input shape: `[batch, history_length, board_planes, 8, 8]`.
- Use shared CNN board encoder.
- Feed board embeddings to `nn.LSTM`.
- Use oldest-to-newest order inside the model.
- Use final output for policy/value heads.
- Policy size `4672`.
- Value scalar in `[-1, 1]`.
- Add config support.
- Add shape and backward tests.
- Update `docs/ARCHITECTURES.md`.

Do not modify MCTS or self-play.

Run `poetry run pytest`.

Acceptance criteria:

- shape, finite-output, and backward tests pass
- invalid input shape fails clearly
- config example exists
- no MCTS, self-play, or data parsing changes are introduced

## Task 11 - LSTM With Temporal Attention

Add optional LSTM-with-temporal-attention architecture.

Requirements:

- Architecture name: `lstm_attention_history`.
- Input shape: `[batch, history_length, board_planes, 8, 8]`.
- Use shared CNN board encoder.
- Feed embeddings into `nn.LSTM`.
- Use current/newest LSTM output as query.
- Use all LSTM outputs as keys and values.
- Use `nn.MultiheadAttention`.
- Add residual connection and `LayerNorm`.
- Policy size `4672`.
- Value scalar in `[-1, 1]`.
- Add config support.
- Tests:
  - forward shape
  - backward smoke test
  - invalid history length raises `ValueError`
  - `attention_heads` divides `lstm_hidden_dim`
- Update `docs/ARCHITECTURES.md`.

Do not modify MCTS, self-play, or data parsing.

Run `poetry run pytest`.

Acceptance criteria:

- shape, finite-output, and backward tests pass
- invalid history length raises `ValueError`
- attention head divisibility is validated
- docs describe query/key/value semantics

## Task 12 - Search Distillation

Add initial search-distillation support.

Requirements:

- Generate MCTS visit-count distributions for selected positions.
- Store targets in a processed dataset format.
- Add KL-divergence or cross-entropy loss against visit distribution.
- Keep human supervised training path working.
- Add small tests using fake visit distributions.
- Update `EXPERIMENTS.md`.

Do not add full self-play yet.

Run `poetry run pytest`.

Acceptance criteria:

- generated visit distributions are normalized and aligned to policy indices
- distillation loss works on fake targets
- supervised human-move training path still works
- experiment docs describe how the targets were produced
- no self-play loop is introduced

## Task 13 - Optional Self-Play

Add the first self-play data-generation path.

Requirements:

- Generate games using project bots only.
- Save game records and metadata.
- Store replay samples with policy/value targets.
- Save config, seed, checkpoint identifiers, and git commit if available.
- Add before/after arena evaluation support.
- Update `EXPERIMENTS.md` and `REPRODUCIBILITY.md`.

Do not claim improvement unless the before/after arena was run under the same protocol.

Run `poetry run pytest`.

Acceptance criteria:

- self-play artifacts have a documented schema
- replay buffer format is documented
- before/after evaluation uses the same arena config
- regressions against baselines are recorded, not hidden
- no external engine labels are introduced

## Task 14 - Optional NNUE-Style Architecture Study

Add a McChess-defined NNUE-style sparse accumulator architecture.

Requirements:

- Define the sparse feature schema in docs before coding.
- Do not import Stockfish NNUE weights.
- Do not use engine evaluations, tablebase labels, or external best-move labels.
- Keep legal move masking external and mandatory.
- Preserve the default policy/value output contract:
  - `policy_logits: [batch, 4672]`
  - `value: [batch]`
- If an incremental accumulator is implemented, test it against full
  recomputation.
- Add parameter count and inference speed reporting.
- Compare against the supervised ResNet under matched dataset, seed, and
  training budget.

Run `poetry run pytest`.

Acceptance criteria:

- feature encoding tests pass
- accumulator update tests pass, if incremental updates are implemented
- shape, finite-output, and backward tests pass
- architecture docs describe inputs, outputs, and limitations
- no external engine labels, weights, or tablebase labels are introduced
