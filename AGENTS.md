# AGENTS.md

## Role

You are working on `McChess`, a compact neural chess research system trained without Stockfish supervision or external engine labels.

Act like a careful research software engineer:

- prefer correctness over cleverness
- implement one milestone at a time
- keep changes small and reviewable
- write tests for chess rules and model shape assumptions
- do not silently change research scope

## Read These First

Before making nontrivial changes, read:

- `PROJECT.md`
- `ROADMAP.md`
- `DESIGN.md`
- `INVARIANTS.md`
- relevant files under `docs/`

If the task touches experiments or evaluation, also read:

- `EXPERIMENTS.md`
- `docs/EVALUATION_PROTOCOL.md`
- `REPRODUCIBILITY.md`

If the task touches data processing, also read:

- `docs/DATASET_PROTOCOL.md`

If the task touches model architecture, also read:

- `docs/ARCHITECTURES.md`

## Hard Constraints

- Do not use Stockfish for supervision; external evaluation only.
- Do not use Syzygy tablebases.
- Do not use external engine labels.
- Human PGN games are allowed.
- Use `python-chess` for legal move generation and game rules.
- Use explicit legal move masking.
- Do not rely on the neural network to learn legality.
- Do not implement future milestones unless explicitly asked.
- Do not rewrite unrelated files.
- Do not introduce large frameworks without a clear reason.

## Engineering Standards

- Python 3.11+
- Poetry for environment and dependency management
- PyTorch for neural networks
- NumPy for tensor preprocessing
- `python-chess` for chess rules
- `pytest` for tests
- `ruff` for linting if configured
- `mypy` for type checking if configured

Follow `docs/CODING_STANDARD.md` for project style, abstraction level, testing
expectations, configs, comments, dependencies, and performance tradeoffs.

Use type hints for public functions.
Prefer small modules with clear APIs.

## Validation

Before claiming completion, run the most relevant checks.

At minimum:

```bash
poetry run pytest
```

If configured:

```bash
poetry run ruff check .
poetry run mypy src
```

If a check cannot be run, explain why.

## Definition Of Done

A task is not complete unless:

- relevant tests were added or updated
- `poetry run pytest` was run, or the reason it could not run is stated
- `poetry run ruff check .` and `poetry run mypy src` were run if configured and relevant
- public APIs have type hints
- docs were updated if behavior, tensor shapes, file formats, configs, or protocols changed
- no future milestone code was added without an explicit request
- no unsupported strength, Elo, or engine-comparison claims were added
- remaining limitations or follow-up work are stated clearly

## Chess Correctness Requirements

Write tests for:

- board encoding
- move-index round trip
- legal move mask
- castling
- promotion
- en passant
- checkmate
- stalemate
- side-to-move value perspective
- MCTS backup sign flip

## Research Discipline

Every experiment should be reproducible from a config.

Each run should save:

- config copy
- random seed
- metrics
- checkpoint path
- dataset manifest path
- git commit if available

For reportable experiments, also save or record:

- hardware notes
- wall-clock runtime
- evaluation config
- exact opponent/checkpoint identifiers
- failure status if the run did not complete

Negative, failed, or inconclusive results should stay documented. Do not delete weak results to make the project look stronger.

## Claims Discipline

Do not write that McChess is strong, high-Elo, or competitive with modern engines unless that exact claim was measured by a documented protocol.

Prefer measured claims:

```text
In local arena config X, checkpoint Y scored Z over N games against baseline B.
```

Avoid unsupported claims:

```text
McChess plays at 2000 Elo.
McChess is close to leading chess engines.
McChess is stronger than Stockfish.
```

## Writing Voice

Write documentation, README copy, PR descriptions, comments, and other
user-facing text in plain engineering prose.

Avoid AI-flavored or marketing-like phrasing such as "it works signal",
"seamless", "powerful", "game-changing", "unlock", "showcase", or vague claims
of quality. Prefer concrete descriptions of what exists, how to run it, what
artifact it writes, and what limitation remains.

Prefer:

```text
Open `notebooks/bot_vs_bot.ipynb` to watch two local policy-only checkpoints
play a live board.
```

Avoid:

```text
This notebook provides an immediate "it works" signal.
```

## Preferred Workflow

For nontrivial tasks:

1. Inspect relevant files.
2. Propose a short plan.
3. Implement the smallest useful version.
4. Add or update tests.
5. Run checks.
6. Summarize changed files and remaining limitations.

## Important Reminder

This is a research platform, not a maximum-strength chess engine.

Optimize for clean ablations, reproducible results, and honest evaluation.
