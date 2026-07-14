# Roadmap

## Milestone Gate

Each milestone should finish with:

- implementation deliverables complete
- relevant unit or smoke tests added
- affected docs updated
- artifacts written when the milestone produces data, checkpoints, or results
- `poetry run pytest` run, or a clear explanation of why it could not run
- no future milestone features included without an explicit request

## Milestone 0 - Repository Foundation

Goal: create a clean, importable Python project.

Deliverables:

- package structure
- `pyproject.toml`
- basic `README.md`
- design docs
- test setup

Exit criteria:

- package imports successfully
- `poetry run pytest` runs
- README explains the project scope and non-goals

## Milestone 1 - Board Encoding

Goal: convert python-chess boards into tensors.

Deliverables:

- single-board encoder
- documented plane order
- tests for shape, piece counts, side to move, castling rights

Exit criteria:

- board encoding tests pass
- square orientation is tested
- `DESIGN.md` documents the exact plane order
- `INVARIANTS.md` documents the board-encoding contract

## Milestone 2 - Move Indexing And Legal Mask

Goal: fixed policy space and reliable legality masking.

Deliverables:

- `move_to_index`
- `index_to_move`
- `legal_policy_mask`
- fixed 4672 policy space if practical
- round-trip tests

Exit criteria:

- all legal moves in generated positions round-trip correctly
- legal mask only marks legal moves
- special moves are tested: castling, promotions, en passant
- policy size and mask dtype/shape are documented

## Milestone 3 - PGN Dataset Builder

Goal: turn human games into supervised samples.

Deliverables:

- PGN parser
- game-level train/val/test split
- value targets from side-to-move perspective
- dataset manifest

Exit criteria:

- tiny PGN fixture creates correct samples
- corrupt games are skipped with counters
- split is by game, not by position
- manifest includes source, filters, counts, split sizes, and code version if available

## Milestone 4 - Supervised Policy/Value Baseline

Goal: train a small ResNet policy/value model.

Deliverables:

- PyTorch dataset
- model
- loss
- training script
- metrics logging
- checkpoint saving

Exit criteria:

- model can overfit a tiny dataset
- forward/backward tests pass
- checkpoint save/load works
- parameter count is logged or easy to compute
- config examples exist

## Milestone 5 - Handmade Baselines

Goal: compare against simple bots.

Deliverables:

- random bot
- material bot
- shallow minimax bot
- policy-only bot

Exit criteria:

- every bot returns legal moves
- smoke matches complete
- baseline behavior is deterministic when seeded

## Milestone 6 - Arena Evaluation

Goal: reproducible bot-vs-bot evaluation.

Deliverables:

- arena runner
- alternating colors
- fixed seeds
- max ply limit
- results JSON

Exit criteria:

- baseline-vs-baseline matches run reproducibly
- results JSON includes seed, colors, max ply, draw rules, and illegal move count
- no Elo claims are made

## Milestone 7 - MCTS

Goal: improve model play using neural search.

Deliverables:

- PUCT MCTS
- policy priors
- value backup with sign flip
- MCTS bot

Exit criteria:

- MCTS bot returns legal moves
- sign flip tests pass
- MCTS can beat or match policy-only in smoke evaluation
- terminal handling is tested
- visit counts and priors can be inspected or saved

Planned follow-up work is documented in
[`planned_upgrades/mcts_performance.md`](planned_upgrades/mcts_performance.md).

## Milestone 8 - History Encoders

Goal: add temporal context.

Deliverables:

- history-board encoder
- history-plane ResNet support
- tests for padding and current-position perspective

Exit criteria:

- history models train/evaluate with same pipeline
- padding and current-position perspective are tested
- history shape is documented in `DESIGN.md` and `INVARIANTS.md`

## Milestone 9 - Architecture Ablations

Goal: compare temporal and attention models.

Architectures:

- ResNet
- History ResNet
- ResNet + square attention
- LSTM history
- LSTM + temporal attention
- Temporal Transformer

Exit criteria:

- each architecture has shape tests
- each can run a small training smoke test
- parameter count and inference speed are reported when practical
- comparisons use matched dataset, seed, and training budget

## Milestone 10 - Search Distillation

Goal: distill MCTS visit distributions into the policy.

Deliverables:

- MCTS target generation
- distillation loss
- comparison against supervised-only model

Exit criteria:

- measurable supervised-only vs distillation evaluation
- MCTS target generation is reproducible from config
- KL divergence or equivalent policy-target metric is reported

## Milestone 11 - Optional Self-Play

Goal: generate self-play games and retrain.

Deliverables:

- self-play worker
- replay buffer
- training loop
- before/after evaluation

Exit criteria:

- self-play does not regress against baselines without being detected
- replay buffer format is documented
- before/after evaluation uses the same arena protocol

## Later Optional - NNUE-Style Sparse Accumulator

Goal: test whether a compact sparse-feature accumulator gives a better
speed/quality tradeoff than convolutional or temporal models under the same
data, seed, and evaluation protocol.

This is optional future work. It is not part of the current supervised ResNet,
MCTS, history, attention, distillation, or self-play milestones.

Allowed:

- project-defined sparse chess features
- human PGN policy targets
- final-result value targets
- self-play targets generated by McChess
- MCTS visit counts generated by McChess

Not allowed:

- Stockfish NNUE weights
- Stockfish or other engine evaluations
- Syzygy or tablebase labels
- external best-move labels
- pretrained chess-engine policy/value targets

Deliverables:

- documented sparse feature encoding
- accumulator/update implementation or clearly documented non-incremental
  baseline
- policy/value outputs matching the project model contract
- legal policy masking unchanged
- parameter count and inference speed reporting
- comparison against the supervised ResNet under matched data and budget

Exit criteria:

- feature encoding tests pass
- incremental accumulator, if implemented, matches full recomputation
- model shape and backward tests pass
- no external engine labels or weights are used
- architecture comparison records speed as well as accuracy or arena score
