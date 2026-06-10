# McChess

[![CI](https://github.com/MecGlandorff/McChess/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MecGlandorff/McChess/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A compact neural chess research system trained without Stockfish, Syzygy, or external engine labels.

McChess currently has a working supervised-learning path: chess-rule contracts,
PGN data processing, an 18-plane board encoder, compact policy/value ResNets,
config-driven training, checkpoint loading, policy-only play, tests, and smoke
reports. Search, arena evaluation, temporal/attention models, distillation, and
self-play are planned next steps.

## Current Status

Implemented:

- repository foundation, Poetry environment, CI, and project documentation
- 18-plane board tensor encoder backed by `python-chess`
- fixed 4672-action policy space with move-index round trips
- explicit legal move masking using `python-chess`
- PGN reader with corrupt-game and unknown-result accounting
- game-level dataset builder with JSONL shards and manifests
- optional tensor cache for faster local training input pipelines
- PyTorch dataset support for JSONL shards and tensor caches
- compact ResNet policy/value model with supervised loss
- supervised training script with epoch/batch metrics, per-epoch checkpoints, and plots
- checkpoint loading plus random, material, negamax alpha-beta, and policy-only bots
- clickable notebook play helper for policy-only checkpoints
- local data, training, and smoke-test configs
- tests for chess/tensor contracts, data processing, model shapes, losses, and scripts

Not yet implemented:

- arena evaluation
- neural MCTS
- temporal and attention model families
- search distillation
- self-play

## Current Capabilities

| Area | Implemented capability |
|---|---|
| Board representation | 18-plane `[planes, 8, 8]` encoder with side-to-move, castling, and legal en-passant metadata |
| Move representation | Fixed `8 x 8 x 73 = 4672` policy space with legal move round-trip tests |
| Legality | Explicit `python-chess` legal policy mask; the model is not trusted to infer legal moves |
| Data pipeline | PGN streaming, final-result value targets, game-level splits, JSONL shards, dataset manifests |
| Training input | JSONL-backed dataset plus optional encoded tensor cache for local throughput |
| Model | Compact PyTorch ResNet presets returning `policy_logits: [batch, 4672]` and `value: [batch]` |
| Training | Config-driven supervised training script with epoch/batch metrics, per-epoch checkpoints, and diagnostic plots |
| Play | Policy-only checkpoint bot with explicit legal masking and clickable notebook play helper |
| Reproducibility | YAML configs, dataset protocols, evaluation protocols, model card, invariants, and CI checks |
| Validation | pytest coverage for board encoding, move indexing, legal masks, PGN handling, datasets, model outputs, losses, and scripts |

## Project Goal

McChess studies how far a small neural chess agent can go using:

- human PGN supervised learning
- board tensor encoding
- legal move masking
- policy/value neural networks
- Monte Carlo Tree Search
- temporal modeling
- attention mechanisms
- optional search distillation
- optional self-play

The goal is not to beat modern engines. The goal is to build a clean, reproducible research platform for neural chess search under limited compute.

## Core Research Question

How far can a compact neural chess engine go using only human games, architectural inductive biases, search, and self-play without engine supervision?

## Research And Engineering Scope

- deep learning fundamentals
- chess board representation
- move-indexing design
- legal move masking
- policy/value modeling
- MCTS experiments
- sequence-modeling experiments
- attention-mechanism experiments
- controlled ablation studies
- reproducible evaluation
- careful research engineering

## Non-Goals

This project does not use:

- Stockfish labels
- Leela labels
- Syzygy tablebases
- external engine evaluations
- pretrained engine recommendations

Strength claims need documented evaluations, not estimates.

## System Overview

```text
PGN games
  -> dataset builder                         implemented
  -> board tensors, policy targets, values   implemented
  -> policy/value ResNet                     implemented
  -> policy-only bot                         implemented
  -> MCTS-enhanced bot                       future
  -> arena evaluation                        future
```

## Model Families

Implemented:

- ResNet policy/value model
- named ResNet presets: `resnet_baseline` and `resnet_b`

Planned ablations:

- optional NNUE-style sparse accumulator
- History ResNet
- ResNet + square attention
- LSTM history
- LSTM + temporal attention
- Temporal Transformer

## Future Evaluation Baselines

- random legal-move bot
- material-count bot
- negamax alpha-beta bot
- policy-only neural bot
- policy/value + MCTS bot

## Repository Map

- `src/mcchess/board/`: board tensors, move indexing, and legal masks
- `src/mcchess/data/`: PGN parsing, dataset shards, tensor caches, and PyTorch datasets
- `src/mcchess/model/`: policy/value ResNets, model presets, and supervised losses
- `scripts/`: dataset building, tensor-cache building, data download, and supervised training
- `configs/`: reproducible data and training configurations
- `tests/`: chess-rule, data, model-shape, loss, and script tests
- `docs/`: architecture notes, coding standard, dataset protocol, and evaluation protocol
- `reports/`: development smoke reports and diagnostic plots

## Validation Snapshot

The test suite covers board orientation, piece planes, castling metadata, legal en-passant encoding, move-index round trips, legal policy masks, PGN parsing, corrupt and unknown-result game handling, game-level dataset splits, tensor cache loading, model output shapes, finite outputs, supervised loss computation, and training-script smoke behavior.

## Development Philosophy

Build one milestone at a time.

Prioritize:

- correctness
- tests
- reproducibility
- honest evaluation
- small reviewable changes

Avoid:

- overbuilding early
- hidden engine supervision
- unsupported Elo claims
- giant untested code dumps

The project coding standard is documented in `docs/CODING_STANDARD.md`. In
short: keep the code compact and hackable, make chess/tensor contracts explicit,
prefer simple functions plus dataclasses, use Poetry and reproducible YAML
configs, and test every public chess or model-shape contract.

## Setup

This project uses Poetry with Python 3.11+.

Recommended local setup:

```bash
poetry env use python3.12
poetry install --with dev,notebook
```

This repository includes `poetry.toml`, so Poetry creates the virtual environment at `.venv/`.

If Poetry complains that the current Python is `3.9.x`, deactivate Conda first:

```bash
conda deactivate
poetry env use python3.12
poetry install --with dev,notebook
```

Verify the interpreter:

```bash
poetry run python --version
```

## Data And Local Training Guides

Useful local workflow guides:

- `docs/guides/INSTALL_DATA.md`
- `docs/guides/GPU_POWER_LIMIT.md`
- `docs/guides/HOW_TO_PLAY.md`

GPU power limiting for NVIDIA cards can be toggled around CUDA training runs:

```powershell
poetry run python gpu_protect --status
poetry run python gpu_protect --on
poetry run python gpu_protect --off
```

The limiter only matters when training resolves to `device=cuda`.

Run checks through Poetry:

```bash
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

GitHub Actions CI runs `poetry check`, `pytest`, `ruff`, and `mypy` on push and pull request.

Start notebooks through Poetry:

```bash
poetry run jupyter nbclassic
```

Play against a local policy checkpoint in the notebook:

```powershell
New-Item -ItemType Directory -Force .local\jupyter\nbclassic-runtime | Out-Null
$env:JUPYTER_RUNTIME_DIR = "$PWD\.local\jupyter\nbclassic-runtime"
poetry run jupyter nbclassic --no-browser --notebook-dir="$PWD" --port=8888 --ServerApp.token=mcchess
```

Then open `notebooks/play_policy_bot.ipynb` and select the `McChess (.venv)`
kernel. See `docs/guides/HOW_TO_PLAY.md` for setup, expected behavior, and
troubleshooting.

Current smoke notebooks:

- `notebooks/encoding_smoke_test.ipynb`
- `notebooks/move_indexing_smoke_test.ipynb`
- `notebooks/dataset_builder_smoke_test.ipynb`
- `notebooks/play_policy_bot.ipynb`

## Research Discipline

The project keeps technical contracts and reproducibility rules in:

- `INVARIANTS.md`
- `REPRODUCIBILITY.md`
- `docs/DATASET_PROTOCOL.md`
- `docs/EVALUATION_PROTOCOL.md`
- `EXPERIMENTS.md`

Reportable results should include configs, seeds, manifests, checkpoints, metrics, and evaluation metadata. Weak, failed, and inconclusive runs should be documented when they answer a research question or expose a limitation.

## Current Supervised Runs

The first matched ResNet-A vs ResNet-B comparison is running on the May 2026
2000+ Lichess dataset: 73,553,382 train positions, 747,498 validation
positions, batch size 2048, AdamW with learning rate `5e-4`, seed `20260501`,
CUDA with the tensor cache.

| Run | Model | Params | Epochs | Val total loss | Final val policy loss |
|---|---|---:|---:|---:|---:|
| ResNet-A | 32 channels, 1 block | ~0.63M | 20 of 20 | `3.3176 -> 3.0549` | `2.1960` |
| ResNet-B | `resnet_b`: 64 channels, 6 blocks | ~1.06M | 16 of 20 | `2.8403 -> 2.5603` | `1.7199` |

ResNet-B is still training as of 2026-06-11 at roughly 80 to 90 minutes per
epoch; its numbers are through epoch 16. Under matched data, seed, and
optimizer settings, ResNet-B's epoch-1 validation total loss (`2.8403`) is
already below ResNet-A's final epoch-20 loss (`3.0549`).

![ResNet-A loss curve](reports/assets/lichess_2026_05_resnet_a_loss_curve.svg)

![ResNet-B loss curve](reports/assets/lichess_2026_05_resnet_b_loss_curve.svg)

These are loss metrics only. No arena or play-strength evaluation has been run
for these checkpoints, and the comparison becomes a reportable result only
after the ResNet-B run completes. Older development smoke reports live in
`reports/`.

## License

McChess is released under the MIT License. See [LICENSE](LICENSE).
