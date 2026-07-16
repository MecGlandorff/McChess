# ResNet-C Epoch-22 MCTS-1000 Stockfish-UCI Benchmark

Date: 2026-07-16

## Scope

This is an external Stockfish benchmark only. Do not use Stockfish moves,
evaluations, or game outcomes for McChess training data, labels, distillation
targets, checkpoint selection, or self-play targets.

The estimate below applies only to one local Stockfish `UCI_Elo` handicap
protocol. It is not an official FIDE rating, Lichess Elo, CCRL Elo, or a
general engine-strength claim.

## Setup

- Config:
  `configs/eval/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games.yaml`
- Command:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games.yaml --keep-awake
```

- Run ID: `stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games`
- Agent: `resnet_c_epoch22_mcts_1000_batch8`
- Checkpoint:
  `runs/lichess_2026_05_2000plus_resnet_c_epoch30_from_epoch12_cached_batchmetrics/checkpoint_epoch_022.pt`
- MCTS budget: 1000 simulations per move, `c_puct = 1.5`, inference batch size 8
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
  `a0b1a7a56ba7081e9975282be2b0afde9a4c5d1e`
- Output directory:
  `runs/external_stockfish/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games/`
- Runtime: 23,375.3 seconds, approximately 6 hours 29 minutes
- Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM
- Status: completed

## Summary

The run completed all 202 scheduled games without an illegal move or recorded
game error.

| Game set | Games | McChess wins | Draws | McChess losses | McChess score | Illegal moves |
|---|---:|---:|---:|---:|---:|---:|
| All games, including full-strength sanity games | 202 | 153 | 33 | 16 | 0.839 | 0 |
| `UCI_Elo` handicap games only | 200 | 153 | 33 | 14 | 0.8475 | 0 |
| Full-strength sanity games only | 2 | 0 | 0 | 2 | 0.000 | 0 |

## Rating Estimate

| Included games | Opponent range | Estimate | Rough 95% interval | Method |
|---:|---|---:|---|---|
| 200 | Stockfish `UCI_Elo=1600..2500` | 2489 | 2415 to 2566 | rough logistic MLE against Stockfish `UCI_Elo` |

Interpretation: under this exact local benchmark, the ResNet-C epoch-22
checkpoint with MCTS-1000 scored `0.8475` over the 200 included handicap games.

## Per-Level Results

| Stockfish `UCI_Elo` | Games | McChess wins | Draws | McChess losses | McChess score |
|---:|---:|---:|---:|---:|---:|
| 1600 | 20 | 20 | 0 | 0 | 1.000 |
| 1700 | 20 | 19 | 0 | 1 | 0.950 |
| 1800 | 20 | 17 | 1 | 2 | 0.875 |
| 1900 | 20 | 19 | 1 | 0 | 0.975 |
| 2000 | 20 | 17 | 3 | 0 | 0.925 |
| 2100 | 20 | 14 | 4 | 2 | 0.800 |
| 2200 | 20 | 16 | 3 | 1 | 0.875 |
| 2300 | 20 | 15 | 3 | 2 | 0.825 |
| 2400 | 20 | 10 | 8 | 2 | 0.700 |
| 2500 | 20 | 6 | 10 | 4 | 0.550 |

## Integrity Checks

- `result.json` and `games.csv` each contain 202 game records.
- All 18,107 stored moves replayed as legal moves with `python-chess`.
- Replayed final positions and terminal results matched the stored records.
- Terminations were 169 checkmates, 23 threefold repetitions, 7 max-ply draws,
  and 3 stalemates.
- All 200 included games had distinct complete move sequences; 193 had distinct
  first-ten-ply sequences.
- SHA-256 of the local `result.json`:
  `1c1507a2f752dc9d347be718ca4132e9c0fb3bed2846be8b2c7b75af2f660222`

## Limitations

- Stockfish `UCI_Elo` is an engine handicap setting, not an official rating.
- The benchmark used the standard initial position only, not an opening suite.
- Twenty games per level leave visible sampling noise; the per-level scores are
  not monotonic.
- Stockfish used a one-second-per-move limit while McChess used a fixed MCTS
  simulation budget, so this is not a symmetric compute comparison.
- Inference batching changes MCTS selection trajectories. Compare this result
  only with runs that record their inference batch size.
- The local hardware identity was checked separately; it was not serialized in
  the generated result JSON.

## Artifact Notes

The full generated `result.json`, `games.csv`, copied `config.yaml`, and
machine-generated `report.md` are stored under the ignored local run directory:

```text
runs/external_stockfish/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games/
```

Those generated files are local run artifacts and are not committed to the
repository.
