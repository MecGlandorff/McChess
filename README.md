# McChess

[![CI](https://github.com/MecGlandorff/McChess/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MecGlandorff/McChess/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

McChess is a compact neural-chess research system: raw human PGNs in, explicit
chess-rule tensors out, PyTorch policy/value checkpoints, reproducible metrics,
and playable policy-only bots. It is built around one question:

> How far can a small neural chess agent go using human games, architectural
> inductive bias, search, and self-play without engine supervision?

The project deliberately avoids Stockfish, Syzygy, Leela, tablebase labels,
external best-move labels, and imported engine weights. `python-chess` owns the
rules. Neural networks rank and evaluate positions only after legal move
masking.

For a quick inspection path through the repo, see [REVIEWER.md](REVIEWER.md).

McChess is not a claim of engine strength. It is a research platform for clean
ablations, honest evaluation, and compact neural search under limited compute.

## What Works Now

McChess already has the core supervised-learning stack needed for neural chess
experiments:

| Area | Current capability |
|---|---|
| Chess contracts | 18-plane `[18, 8, 8]` board tensors, side-to-move values, and a fixed 4,672-action policy space |
| Legality | Explicit legal move masks from `python-chess`; the model is never trusted to learn legality |
| Data | PGN streaming, corrupt/unknown-result accounting, game-level splits, JSONL shards, and manifests |
| Training input | JSONL-backed datasets plus optional tensor caches for faster local CUDA training |
| Models | Compact PyTorch policy/value ResNet presets: `resnet_a` and `resnet_b` |
| Training | YAML-configured supervised training with epoch metrics, batch metrics, checkpoints, and loss plots |
| Evaluation metrics | Legal-masked supervised top-k evaluation and value diagnostics via `scripts/eval_top1.py` |
| Bots | Random, material, negamax alpha-beta, and policy-only checkpoint bots |
| Play | Clickable notebook widget for playing a local policy-only checkpoint |
| Reproducibility | Project invariants, dataset protocol, evaluation protocol, model card, configs, and CI checks |
| Tests | Coverage for board encoding, move indexing, legal masks, PGNs, datasets, model shapes, losses, checkpoints, bots, and scripts |

Not yet implemented:

- neural MCTS
- temporal and attention model families
- search distillation
- self-play

## Current Evidence

These are supervised-learning measurements only. They do not imply Elo,
Lichess strength, or engine competitiveness.

### Reported Baseline

The first reportable supervised baseline in `RESULTS.md` trained a compact
ResNet on a 600k-position prefix of Lichess 2013-01 and evaluated a 40k-position
held-out test slice:

| Experiment | Model | Legal top-1 | Legal top-3 | Legal top-5 | Value MSE |
|---|---|---:|---:|---:|---:|
| `supervised_resnet32x2_lichess2013_01` | ResNet 32ch x2, ~1.2M params | 0.262 | 0.492 | 0.610 | 0.858 |

That run used human moves and final game results only. The raw unmasked argmax
was legal for 88.8% of evaluated positions, which is useful evidence that the
model learned board structure, but move selection still requires explicit legal
masking.

### Full-Data Supervised Runs

The current matched ResNet-A vs ResNet-B comparison uses the May 2026 2000+
Lichess dataset with 73,553,382 train positions and 747,498 validation
positions. Both runs use batch size 2048, AdamW, learning rate `5e-4`, seed
`20260501`, CUDA, and the tensor cache.

| Run | Model | Params | Epochs | Validation total loss | Final validation policy loss | Status |
|---|---|---:|---:|---:|---:|---|
| ResNet-A | 32 channels, 1 block | ~0.63M | 20 of 20 | `3.3176 -> 3.0549` | `2.1960` | completed 2026-06-07 |
| ResNet-B | 64 channels, 6 blocks | ~1.06M | 20 of 20 | `2.8403 -> 2.5447` | `1.7047` | completed 2026-06-11 |

Under matched data, seed, and optimizer settings, ResNet-B's epoch-1 validation
total loss (`2.8403`) was already below ResNet-A's final epoch-20 loss
(`3.0549`). The final ResNet-B validation total loss was `2.5447`.

![ResNet-A loss curve](reports/assets/lichess_2026_05_resnet_a_loss_curve.svg)

![ResNet-B loss curve](reports/assets/lichess_2026_05_resnet_b_loss_curve.svg)

No arena or play-strength evaluation has been run for these full-data
checkpoints yet.

## Core Contracts

McChess keeps chess and tensor assumptions explicit so experiments do not drift.

| Contract | Current value |
|---|---|
| Board tensor | `[18, 8, 8]`, `float32` |
| Batched input | `[batch, 18, 8, 8]` |
| Policy space | `8 x 8 x 73 = 4672` |
| Policy logits | `[batch, 4672]` |
| Value output | `[batch]`, side-to-move perspective, bounded to `[-1, 1]` |
| Legal mask | `[4672]`, `float32`, generated from `python-chess` |
| Split rule | split by game, never by position |

The source of truth for these contracts is `INVARIANTS.md`, with implementation
details in `DESIGN.md`.

## System Shape

```text
PGN games
  -> dataset builder                         implemented
  -> board tensors, policy targets, values   implemented
  -> policy/value ResNet                     implemented
  -> policy-only bot                         implemented
  -> arena evaluation                        implemented
  -> MCTS-enhanced bot                       future
  -> history, attention, distillation        future
  -> self-play                               future
```

The current supervised sample is:

```json
{
  "game_id": "g000000",
  "ply": 0,
  "fen": "rnbqkbnr/pppppppp/...",
  "move_uci": "e2e4",
  "policy_index": 748,
  "value": 0.0,
  "result": "1/2-1/2",
  "split": "train"
}
```

Training artifacts include copied configs, metrics JSONL, batch metrics JSONL,
status files, checkpoints, and plots.

## Repository Map

- `src/mcchess/board/`: board tensors, move indexing, and legal masks
- `src/mcchess/data/`: PGN parsing, dataset shards, tensor caches, and PyTorch datasets
- `src/mcchess/model/`: policy/value ResNets, presets, checkpoints, and supervised losses
- `src/mcchess/bots/`: baseline bots and policy-only checkpoint play
- `scripts/`: dataset building, data filtering/downloading, tensor-cache building, training, and top-k evaluation
- `configs/`: reproducible data, training, evaluation, and future self-play configs
- `tests/`: chess-rule, data, model-shape, loss, checkpoint, bot, and script tests
- `docs/`: architecture notes, coding standard, dataset protocol, evaluation protocol, and guides

Run a small local arena:

```bash
poetry run python scripts/run_arena.py configs/eval/arena_smoke_material_vs_random.yaml
```

Arena results are written as JSON from the named agent's perspective with
alternating colors and max-ply draw adjudication. They are local evaluation
artifacts, not Elo estimates.

To watch local ResNet-A and ResNet-B policy-only checkpoints play with a
four-second pause after each move:

```bash
poetry run python scripts/run_arena.py configs/eval/arena_watch_resnet_a_vs_resnet_b.yaml
```

The delay is for pacing printed moves only; policy-only bots do not spend that
time searching.
- `reports/`: development reports and diagnostic plots

## Setup

McChess uses Poetry with Python 3.11+.

```bash
poetry env use python3.12
poetry install --with dev,notebook
poetry run python --version
```

This repository includes `poetry.toml`, so Poetry creates the virtual
environment at `.venv/`.

If Poetry picks up an old Conda Python, deactivate Conda first:

```bash
conda deactivate
poetry env use python3.12
poetry install --with dev,notebook
```

Run the standard checks:

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

GitHub Actions runs `poetry check`, `pytest`, `ruff`, and `mypy` on push and
pull request.

## Local Workflows

Useful guides:

- `docs/guides/INSTALL_DATA.md`
- `docs/guides/GPU_POWER_LIMIT.md`
- `docs/guides/HOW_TO_PLAY.md`

Build or filter datasets through the scripts in `scripts/` and the YAML configs
under `configs/`. Large CUDA runs can use the optional tensor cache documented
in `docs/DATASET_PROTOCOL.md`.

Start notebooks through Poetry:

```bash
poetry run jupyter nbclassic
```

To play a policy-only checkpoint locally:

```powershell
New-Item -ItemType Directory -Force .local\jupyter\nbclassic-runtime, .local\ipython | Out-Null
$env:JUPYTER_RUNTIME_DIR = "$PWD\.local\jupyter\nbclassic-runtime"
$env:IPYTHONDIR = "$PWD\.local\ipython"
poetry run jupyter nbclassic --no-browser --notebook-dir="$PWD" --port=8888 --ServerApp.token=mcchess
```

Then open `notebooks/play_policy_bot.ipynb` with the `McChess (.venv)` kernel.
The notebook is a manual inspection tool, not an arena evaluation or a
strength result.

GPU power limiting for NVIDIA cards can be toggled around CUDA training runs:

```powershell
poetry run python gpu_protect --status
poetry run python gpu_protect --on
poetry run python gpu_protect --off
```

The limiter only matters when training resolves to `device=cuda`.

## Research Boundaries

Allowed supervision:

- human PGN games
- final game results
- self-play games generated by this project
- MCTS visit counts generated by this project's own model

Disallowed supervision:

- Stockfish evaluations or best moves
- Leela evaluations or best moves
- Syzygy tablebases
- external engine labels
- imported engine weights or pretrained chess-engine policy/value targets

Strength claims require documented evaluation. Preferred:

```text
In local arena config X, checkpoint Y scored Z over N games against baseline B.
```

Unsupported:

```text
McChess is 2000 Elo.
McChess is close to leading engines.
McChess is stronger than Stockfish.
```

## Roadmap

The next milestones are intentionally narrow:

1. Run and record initial policy-only arena comparisons under fixed configs.
2. Add neural MCTS with legal expansion, masked priors, terminal handling, and
   backup sign-flip tests.
3. Compare policy-only and MCTS play under fixed budgets.
4. Add history and attention architectures as matched ablations.
5. Test search distillation and optional self-play only after the arena and MCTS
   protocols are stable.

The full milestone list is in `ROADMAP.md`.

## Development Philosophy

Build one milestone at a time.

Prioritize:

- correctness
- tests
- reproducibility
- honest evaluation
- small reviewable changes

Avoid:

- hidden engine supervision
- unsupported Elo claims
- future milestone code mixed into current work
- giant untested rewrites

The project coding standard is documented in `docs/CODING_STANDARD.md`.

## License

McChess is released under the MIT License. See [LICENSE](LICENSE).
