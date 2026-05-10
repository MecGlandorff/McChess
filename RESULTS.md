# Results

This file records measured results.

Do not put aspirational claims here. Only report results that were actually run.

## Current Status

No reportable experiments yet.

## Reporting Standards

Each result should include:

- dataset name and manifest
- model config
- training config
- evaluation config
- random seed
- number of games
- MCTS budget, if applicable
- hardware notes, if available
- date run
- checkpoint path or identifier
- metrics path
- results path
- git commit if available
- run status

Do not copy a result into the tables below unless the underlying experiment entry has enough metadata to reproduce it.

For arena results, also include:

- color policy
- max ply
- draw adjudication rule
- opening protocol
- illegal move count
- search budget or policy-only declaration

## Supervised Learning Results

| Experiment | Model | Dataset | Top-1 | Top-3 | Top-5 | Value MSE |
|---|---|---|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Arena Results

| Experiment | Agent | Opponent | Games | Wins | Draws | Losses | Score |
|---|---|---|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## MCTS Scaling Results

| Model | Budget | Games | Score | Nodes/sec | Notes |
|---|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD |

## Architecture Ablation Results

| Architecture | Params | History | Attention | Top-1 | Value MSE | Arena Score | Speed |
|---|---:|---|---|---:|---:|---:|---:|
| ResNet | TBD | No | No | TBD | TBD | TBD | TBD |
| History ResNet | TBD | Yes | No | TBD | TBD | TBD | TBD |
| ResNet + Square Attention | TBD | No | Yes | TBD | TBD | TBD | TBD |
| LSTM History | TBD | Yes | No | TBD | TBD | TBD | TBD |
| LSTM + Temporal Attention | TBD | Yes | Yes | TBD | TBD | TBD | TBD |
| Temporal Transformer | TBD | Yes | Yes | TBD | TBD | TBD | TBD |

## Failure Analysis

Track examples where models fail:

- hanging pieces
- missed mate in one
- unsafe captures
- poor king safety
- endgame conversion failures
- repetition/draw handling
- tactical horizon problems

## Limitations

Current limitations should be recorded honestly here.
