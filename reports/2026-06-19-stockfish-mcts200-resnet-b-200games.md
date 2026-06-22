# ResNet-B MCTS-200 Stockfish-UCI Benchmark

Date: 2026-06-19

## Scope

This is an external Stockfish benchmark only. Do not use Stockfish moves,
evaluations, or game outcomes for McChess training data, labels, distillation
targets, checkpoint selection, or self-play targets.

The estimate below is a local result against Stockfish `UCI_Elo` handicap
levels under one exact config. It is not Lichess Elo, FIDE Elo, CCRL Elo, or a
general engine-strength claim.

## Setup

- Config: `configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml`
- Command:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml
```

- Run ID: `stockfish_mcts200_resnet_b_elo_200games`
- Agent: `resnet_b_mcts_200`
- Checkpoint:
  `runs/lichess_2026_05_2000plus_resnet_b_epoch20_cached_batchmetrics/checkpoint.pt`
- MCTS budget: 200 simulations per move, `c_puct = 1.5`
- Opponent: Stockfish 18 through UCI
- Stockfish limit: `time=1.0s/move`
- Stockfish handicap levels: `UCI_Elo=1600` through `2500`
- Full-strength sanity games: 2, excluded from the rating estimate
- Handicap games: 200, included in the rating estimate
- Opening protocol: standard initial position
- Color policy: alternating McChess White first over the global game schedule
- Max ply: 180
- Draw rule: `python_chess_outcome_or_max_ply_draw`
- Seed: 0
- Git commit recorded by the run:
  `9e261a35711c08d5afd6ba5a683ad0af563717a7`
- Output directory:
  `runs/external_stockfish/stockfish_mcts200_resnet_b_elo_200games/`
- Runtime: 26,181.7 seconds
- Status: completed

## Summary

The run completed all 202 scheduled games.

| Game set | Games | McChess wins | Draws | McChess losses | McChess score | Illegal moves |
|---|---:|---:|---:|---:|---:|---:|
| All games, including full-strength sanity games | 202 | 95 | 53 | 54 | 0.601 | 0 |
| `UCI_Elo` handicap games only | 200 | 95 | 53 | 52 | 0.608 | 0 |
| Full-strength sanity games only | 2 | 0 | 0 | 2 | 0.000 | 0 |

## Rating Estimate

| Included games | Opponent range | Estimate | Rough 95% interval | Method |
|---:|---|---:|---|---|
| 200 | Stockfish `UCI_Elo=1600..2500` | 2171 | 2110 to 2233 | rough logistic MLE against Stockfish `UCI_Elo` |

Interpretation: under this local benchmark only, ResNet-B with MCTS-200 scored
`0.6075` over 200 included games against Stockfish `UCI_Elo` handicap levels
from 1600 through 2500 at `time=1.0s/move`.

## Per-Level Results

| Stockfish `UCI_Elo` | Games | McChess wins | Draws | McChess losses | McChess score |
|---:|---:|---:|---:|---:|---:|
| 1600 | 20 | 15 | 3 | 2 | 0.825 |
| 1700 | 20 | 17 | 1 | 2 | 0.875 |
| 1800 | 20 | 11 | 5 | 4 | 0.675 |
| 1900 | 20 | 11 | 3 | 6 | 0.625 |
| 2000 | 20 | 11 | 6 | 3 | 0.700 |
| 2100 | 20 | 12 | 5 | 3 | 0.725 |
| 2200 | 20 | 4 | 7 | 9 | 0.375 |
| 2300 | 20 | 4 | 11 | 5 | 0.475 |
| 2400 | 20 | 3 | 8 | 9 | 0.350 |
| 2500 | 20 | 7 | 4 | 9 | 0.450 |

## Limitations

- Stockfish `UCI_Elo` is a handicap setting, not an online, FIDE, or CCRL rating.
- Results are not monotonic by handicap level; 20 games per level is still a
  small sample for per-level conclusions.
- The benchmark used the standard initial position only, not an opening suite.
- The Stockfish side used a one-second-per-move limit while McChess used a fixed
  200-simulation MCTS budget, so this is not a symmetric compute comparison.
- Hardware notes were not recorded in the result JSON.
- The result applies to this checkpoint, Stockfish binary, config, and run only.

## Artifact Notes

The full generated `result.json`, `games.csv`, copied `config.yaml`, and
machine-generated `report.md` are stored under the ignored local run directory:

```text
runs/external_stockfish/stockfish_mcts200_resnet_b_elo_200games/
```

Those files are local run artifacts and are not committed to the repository.
