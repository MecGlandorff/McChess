# Coding Standard

McChess should read like a small research codebase: compact, direct, and easy
to modify, with explicit contracts wherever chess correctness, tensor shapes,
configs, datasets, checkpoints, or evaluation claims are involved.

The preferred style is nanochat-like in spirit: simple files, obvious control
flow, minimal framework machinery, and scripts that show the real workflow.
McChess adds stricter research discipline because chess legality, value
perspective, and reproducibility errors can silently invalidate results.

## Core Principles

- Prefer correctness over cleverness.
- Keep modules small enough to scan, but split files only when it improves
  comprehension.
- Use the project contracts in `DESIGN.md` and `INVARIANTS.md` as the source of
  truth for shapes, move indexing, value perspective, dataset formats, and MCTS
  semantics.
- Make research runs reproducible from configs before treating results as
  reportable.
- Do not add future milestone code, engine supervision, or strength claims
  without explicit scope and documented measurements.

## Abstraction

Default to simple functions plus dataclasses for structured configs and result
objects.

Use classes when they own meaningful state or lifecycle:

- PyTorch modules
- trainers
- arena/evaluation runners
- MCTS/search state
- checkpoint managers

Avoid formal interfaces, inheritance trees, registries, or dependency injection
unless multiple real implementations need the same contract. A small function is
usually better than a premature framework.

## Public APIs

Public functions must have type hints and clear inputs/outputs.

Prefer APIs that expose chess and tensor contracts directly:

```python
def encode_board(board: chess.Board) -> np.ndarray:
    ...
```

Functions that accept user-provided paths, configs, PGNs, checkpoints, or
dataset records should raise explicit exceptions or return structured counters
for invalid inputs. Use assertions for programmer errors and internal invariants.

## Shapes And Contracts

Tensor-producing and model-facing code must make shapes obvious near the code
path. Use short comments when shape is part of the contract:

```python
# policy_logits: [batch, 4672]
# value: [batch]
```

When changing any of these contracts, update implementation, tests, and docs in
the same change:

- board tensor shape or plane order
- policy index space
- legal mask shape or dtype
- value target perspective
- model output shape
- dataset schema or manifest fields
- checkpoint format
- evaluation result format
- MCTS backup semantics

## Chess Correctness

Use `python-chess` for legal move generation and game rules. Neural code may
rank legal moves; it must not define legality.

Legal move masking is mandatory before move selection and before MCTS expansion.
Special moves and terminal states must be covered by tests when touched:

- castling
- promotion and underpromotion
- en passant
- checkmate
- stalemate
- side-to-move value perspective
- MCTS backup sign flip

## Configs And Workflows

Use Poetry for dependency management and command execution.

Use plain YAML configs for reproducible workflows. Dataclasses are preferred for
typed config objects inside Python. Avoid larger config frameworks unless simple
YAML plus dataclasses becomes a real bottleneck.

Training, dataset building, evaluation, self-play, and distillation workflows
should be runnable from scripts or modules, not only notebooks. Reportable runs
must save enough metadata to replay or audit the result.

## Typing

Type public functions and production functions that are created or modified.
Move toward stricter typing over time, but do not make tensor code unreadable to
appease the type checker.

Use tests and concise shape comments for tensor-rank assumptions that Python
types cannot express well.

## Testing

Every public API should have tests for its contract or shape assumptions.

Use seeded randomized tests for chess move and board logic when practical. Use
smoke tests for scripts or workflows that create artifacts, train models,
evaluate games, or mutate datasets.

Tests should be focused and cheap by default. Broaden coverage when a change
touches shared contracts, chess rules, model outputs, or reproducibility.

## Comments

Comments should explain reasoning, tensor shapes, chess edge cases, or research
tradeoffs. Avoid comments that restate the next line of code.

Good comments answer questions like:

- why this target is from the side-to-move perspective
- why a promotion maps to a specific policy plane
- why an evaluation field is required for reproducibility
- why a performance shortcut preserves the documented contract

## Experiments

Exploratory local runs can stay lightweight. Reportable runs, saved evaluations,
checkpoints, and results should record:

- config copy
- random seed
- metrics
- checkpoint path when applicable
- dataset manifest path when applicable
- git commit if available
- evaluation config when applicable
- failure status if the run did not complete

Negative and inconclusive results should remain documented when they answer a
research question or reveal a limitation.

## Dependencies

Poetry is the dependency boundary. New dependencies should be rare, justified,
and added only when they remove meaningful complexity or enable a required
capability.

Do not introduce large frameworks for ordinary scripting, configuration,
logging, or experiment tracking unless the project has outgrown the simpler
approach.

## Performance

Clarity comes first until profiling or scale proves otherwise.

Vectorize obvious NumPy and PyTorch hot paths. Design training, data loading,
and MCTS with performance in mind, but do not make code dense before there is a
measured reason.

When optimizing, keep the contract visible and add tests that protect behavior.

## Documentation

Docs should be updated when behavior, tensor shapes, file formats, configs,
protocols, or research claims change.

Project documentation should sound like it was written by a careful engineer,
not like product marketing or generated filler. Use direct, concrete language:
what exists, how to run it, what output it writes, what the result does and
does not prove, and what limitations remain.

Avoid vague persuasive phrasing, hype, and AI-flavored summary language. Do not
describe features as signals, showcases, unlocks, seamless experiences, or
proof of quality. Let runnable commands, tests, configs, and measured results
speak for themselves.

Keep docs honest and measured. Prefer:

```text
In local arena config X, checkpoint Y scored Z over N games against baseline B.
```

Avoid unsupported claims about Elo, engine strength, or competitiveness.
