# ResNet-C Epoch 30: Stockfish Development Benchmark

This is a 22-game post-training development benchmark for the final epoch-30
ResNet-C model. It was not used as training data, a training target, or a
checkpoint-selection signal.

## Protocol

- Run ID: `stockfish_mcts1000_resnet_c_epoch30_batch8_elo`
- Config: `configs/eval/stockfish_mcts1000_resnet_c_epoch30_batch8_elo.yaml`
- Model: epoch 30, with weights identical to `models_archive/resnet_c_epoch_030.pt`
- Agent: fixed-budget MCTS-1000, `c_puct=1.5`, inference batch size 8
- Opponent: Stockfish 18
- Opponent limit: 1 second per move
- Schedule: two full-strength sanity games, then two games at each Stockfish
  `UCI_Elo` setting from 1600 through 2500
- Colors: alternating, McChess White first
- Openings: standard initial position only
- Maximum game length: 180 ply, then adjudicated draw
- Seed: 0
- Recorded Git commit: `6cbf8d7b3aa11437e8aa856cc73c828ae73863fc`
- Started: 2026-08-11 17:30:14 UTC
- Completed: 2026-08-11 18:07:43 UTC

## Result

| Games | Wins | Draws | Losses | Score | Illegal moves |
|---:|---:|---:|---:|---:|---:|
| 22 | 17 | 1 | 4 | 0.795 | 0 |

The two full-strength sanity games were both losses and were excluded from the
handicap estimate. Across the 20 `UCI_Elo` handicap games, McChess scored 0.875.
The evaluator produced a rough local estimate of 2539 with a 2307–2835 interval.

## Interpretation

This sample is small, uses one opening position, and has wide uncertainty. The
estimate describes only this checkpoint, MCTS budget, Stockfish binary,
handicap configuration, and move limit. Stockfish `UCI_Elo` is not Lichess Elo,
FIDE Elo, or a general engine-strength measurement. The planned 200-game config
is `configs/eval/stockfish_mcts1000_resnet_c_epoch30_batch8_elo_200games.yaml`.
