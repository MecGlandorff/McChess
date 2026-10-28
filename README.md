# McChess

[![CI](https://github.com/MecGlandorff/McChess/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MecGlandorff/McChess/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

McChess is a compact neural chess research system trained on human games.
It combines a PyTorch policy/value ResNet with explicit legal move masking and
fixed-budget MCTS. The repository includes a **3.6M-parameter epoch-30
ResNet-C inference model**, so playing does not require training first.

The research question is how board representation, policy/value learning, and
search interact under limited compute. Stockfish is used only as an external
evaluation opponent. Training excludes external engine labels, imported engine
weights, and tablebases; `python-chess` supplies legal moves and game rules.

## Quick Start

Use Python 3.11+ and Poetry from the repository root:

```powershell
poetry install
poetry run mcchess-play
```

Terminal play defaults to MCTS-800. For faster policy-only play:

```powershell
poetry run mcchess-play --mode policy
```

Use `poetry run mcchess-play --help` for color, device, checkpoint, and search
options. The bundled artifact's checksum and provenance are in
[models_archive](models_archive/README.md).

Poetry creates `.venv/` through the included `poetry.toml`. If it selects the
wrong interpreter, deactivate any active Conda environment and use
`poetry env use <path-to-python-3.11-or-newer>` before installing.

For optional visual play, install `poetry install --with notebook`, then run
`poetry run jupyter nbclassic` and open
[play_mcts_bot.ipynb](notebooks/play_mcts_bot.ipynb) or
[play_policy_bot.ipynb](notebooks/play_policy_bot.ipynb).
See [How to play](docs/guides/HOW_TO_PLAY.md) for notebook setup and controls.

## Current Evidence

Both rows below evaluate the published epoch-30 ResNet-C model.

| Evaluation | Scope | Result |
|---|---|---|
| Held-out human moves | First 40,000 positions of the test split | Legal top-1 **50.225%**, top-3 **78.3075%**, top-5 **88.0975%** |
| External Stockfish benchmark | 200 games, `UCI_Elo` 1400–3190, 10 paired openings | **112 wins / 46 draws / 42 losses; 67.5% score** |

The [supervised report](reports/2026-08-11-supervised-resnet-c-epoch30-test-40000.md)
also records value MSE **0.857878**, against **0.941850** for a constant-zero
predictor. This is a bounded test prefix, not the entire held-out split.

The [October benchmark](reports/2026-10-28-stockfish-resnet-c-epoch30-wide-ladder.md)
used MCTS-1000, inference batch size 8, `c_puct=1.5`, and Stockfish 18 at
1.0 second per opponent move. Each level had 20 games: ten fixed openings,
each played with both colors. McChess scored **60% at UCI 2600** and
**37.5% at UCI 2800**. No game reached the 400-ply draw cap; all 200 game
records passed legal replay and result checks.

The runner's rough local handicap estimate is **2653** (model interval
2571–2736). This is not a FIDE, Lichess, or general engine rating; the interval
does not cover all opening dependencies or calibration uncertainty. The same
model's earlier runs used different openings, levels, and move caps, so their
scores do not establish model improvement. [RESULTS.md](RESULTS.md) retains
those runs, including the preliminary 12-game probe and full-strength losses.

## What Exists And What Remains

Implemented: PGN processing with game-level splits and manifests, optional
tensor caches, supervised ResNet training, checkpoint export, policy-only
play, fixed-budget PUCT MCTS, baseline bots, and reproducible evaluation.
Inputs are `[batch, 18, 8, 8]`; outputs are 4,672 policy logits and one
side-to-move value per board.

The current model has no board history, halfmove-clock input, or repetition
input. Search uses the board's move history for draw rules. Broader opening
coverage, matched search-budget comparisons, and value calibration remain
evaluation work. Temporal models, attention, distillation, and self-play
remain unimplemented; see [ROADMAP.md](ROADMAP.md).

## Development And Navigation

```powershell
poetry install --with dev
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

For a small arena without neural checkpoints:

```powershell
poetry run python -m mcchess.eval.arena configs/eval/arena_smoke_material_vs_random.yaml
```

The arena writes `result.json` under the config's `output_dir`, from the named
agent's perspective. GitHub Actions runs Poetry validation, tests, Ruff, and mypy.

- [PROJECT.md](PROJECT.md): scope, supervision rules, and research questions.
- [DESIGN.md](DESIGN.md) and [INVARIANTS.md](INVARIANTS.md): system and tensor contracts.
- [Architectures](docs/ARCHITECTURES.md): implemented presets and future ablations.
- [Dataset protocol](docs/DATASET_PROTOCOL.md) and [data guide](docs/guides/INSTALL_DATA.md): data preparation.
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md) and [Stockfish guide](docs/guides/STOCKFISH_ELO_EVAL.md): evaluation commands and interpretation.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): run artifacts, export, and provenance.
- [Coding standard](docs/CODING_STANDARD.md): implementation and test expectations.
- [MCTS explainer](reports/2026-06-17-mcts-puct-explainer.md): PUCT and backup signs.
- [Hardware controls](docs/guides/HARDWARE_PROTECT.md): optional local CPU/GPU limits.

Code lives under `src/mcchess/`: `board`, `data`, `model`, `bots`, `search`,
and `eval`. Runnable workflows are in `scripts/` and `configs/`; correctness
checks are in `tests/`. McChess is released under the [MIT License](LICENSE).
