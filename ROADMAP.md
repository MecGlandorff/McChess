# Roadmap

Implement one milestone at a time. A milestone finishes with relevant tests,
updated contracts/docs, saved artifacts where applicable, and the checks in
[AGENTS.md](AGENTS.md). Implemented infrastructure does not imply that all
research comparisons have been completed.

## Implemented Foundation

| Milestone | Current implementation |
|---|---|
| 0. Repository | Python package, Poetry environment, tests, and CI. |
| 1. Board encoding | 18-plane single-board encoder with orientation and metadata tests. |
| 2. Move indexing | 4,672-action policy, legal masking, round trips, and special-move tests. |
| 3. Dataset builder | Human PGN parsing, game-level splits, counters, JSONL shards, manifests, and optional tensor caches. |
| 4. Supervised baseline | ResNet presets, policy/value loss, training, metrics, resume checkpoints, and an epoch-30 inference artifact. |
| 5. Baseline bots | Random, material, negamax, and policy-only agents. |
| 6. Arena | Alternating colors, fixed seeds/openings, draw adjudication, and result artifacts. |
| 7. MCTS | Fixed-budget PUCT, masked priors, sign-flip backup, terminal handling, and batched leaf inference. |

Completed measurements and their limits are indexed in [RESULTS.md](RESULTS.md).
The shared model's export and provenance workflow is documented in
[models_archive](models_archive/README.md).

## Current Evaluation And Performance Work

- Compare policy-only and MCTS under fixed simulation and wall-clock budgets,
  with paired colors and broader saved opening coverage.
- Diagnose value calibration and tactical/endgame failures; use validation
  data for tuning and reserve test data for fixed final evaluations.
- Add reproducible search counters before further optimization:
  [MCTS plan](planned_upgrades/mcts_performance.md).
- Profile remaining PGN preparation costs:
  [data plan](planned_upgrades/data_processing_speed_up.md).
- Measure opt-in precision and training-throughput changes independently:
  [training plan](planned_upgrades/training_throughput.md).

The plans describe deferred work and do not authorize implementation.

## 8. History Encoding

Add a separate history encoder without changing the single-board contract.
The proposed caller order is current-first, with 12 piece planes per historical
board and zero padding for missing history. Keep value perspective anchored to
the current board; explicitly document the metadata and resulting tensor shape.

Gate: shape, padding, current-position consistency, and perspective tests;
history models use the existing training/evaluation workflow.

## 9. Architecture Ablations

Compare History ResNet, square attention, LSTM history, LSTM with temporal
attention, and Temporal Transformer against the single-board ResNet.
[ARCHITECTURES.md](docs/ARCHITECTURES.md) records the proposed interfaces.

Gate: config, shape and invalid-input tests, finite outputs, backward smoke
test, parameter count, and inference speed for each model. Compare matched
dataset, seed, and training budget; normalization changes remain explicit.

## 10. Search Distillation

Generate normalized, policy-index-aligned MCTS visit targets from McChess
checkpoints. Keep human-supervised training available and document the target
schema, config, seed, and source checkpoint.

Gate: target/loss tests, reproducible generation, KL divergence or equivalent
target metric, and supervised-only versus distilled evaluation under one
protocol. Self-play is a separate milestone.

## 11. Optional Self-Play

Generate games using project bots, save replay samples and their schema, and
retrain with recorded configs, seeds, and checkpoint identifiers.

Gate: before/after arena evaluation under the same protocol, including
regressions against baselines. Document replay-buffer behavior and retain
failed or inconclusive runs.

## Later Optional: NNUE-Style Sparse Accumulator

Test a McChess-defined sparse feature schema and accumulator against ResNet
under matched data and budget. Use allowed project targets only; imported
engine weights, engine evaluations, and tablebase labels remain excluded.

Gate: explicit side-to-move features, documented policy-relevant metadata,
feature tests, incremental-versus-full recomputation tests if incremental,
policy/value shape and backward tests, and measured parameter count and speed.
A value-only scorer must be a separately documented experimental bot.
