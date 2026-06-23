# Reviewer Guide

McChess is a compact neural chess research project trained without Stockfish,
Syzygy, or external engine labels. It is milestone-based: supervised-learning
components, a local arena runner, and fixed-budget neural MCTS are implemented
and tested. Temporal models, attention models, self-play, and distillation are
future work.

Start with:

- `PROJECT.md` for the research goal and non-goals
- `INVARIANTS.md` for contracts that must not drift
- `DESIGN.md` for the system design
- `ROADMAP.md` for what is done and planned
- `RESULTS.md` for measured results only

Implemented core:

- board encoding
- move indexing
- legal move masking
- PGN dataset building
- supervised ResNet training
- checkpoint loading
- policy-only play
- local arena evaluation with random, material, negamax, and policy-only bots
- fixed-budget neural MCTS with masked priors and side-to-move value backup

Not yet implemented:

- temporal and attention model families
- search distillation
- self-play

Important tests:

- `tests/test_board_encoding.py`
- `tests/test_move_indexing.py`
- `tests/test_pgn_reader.py`
- `tests/test_model_network.py`
- `tests/test_train_supervised_script.py`
- `tests/test_checkpoint_and_bots.py`
- `tests/test_arena.py`
- `tests/test_mcts.py`

Verify with:

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Current reportable results are limited to the experiments recorded in
`RESULTS.md`, `EXPERIMENTS.md`, and `reports/`. Do not infer Elo, engine
strength, or broad arena performance unless the result was produced and recorded
under the evaluation protocol.
