# Reviewer Guide

McChess is a compact neural chess research project trained without Stockfish,
Syzygy, or external engine labels. It is milestone-based: some
supervised-learning components are implemented and tested, while MCTS, arena
evaluation, self-play, and distillation are future work.

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

Not yet implemented:

- arena evaluation
- neural MCTS
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

Verify with:

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Current results are supervised-learning results only. Do not infer Elo, engine
strength, or arena performance from them.
