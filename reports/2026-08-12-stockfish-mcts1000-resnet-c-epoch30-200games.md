# ResNet-C Epoch 30: Completed Stockfish-UCI Benchmark

This post-training benchmark evaluates the final scheduled epoch-30 checkpoint.
Stockfish moves, evaluations, and outcomes were not used for training,
distillation, checkpoint selection, or hyperparameter tuning.

## Protocol And Provenance

- Run: `stockfish_mcts1000_resnet_c_epoch30_batch8_elo_200games`, completed.
- Recorded code commit: `6cbf8d7b3aa11437e8aa856cc73c828ae73863fc`.
- Started: 2026-08-11 18:11:25 UTC; completed: 2026-08-12 01:07:00 UTC.
- Runtime: 24,934.86 seconds (6.93 hours).
- Model: ResNet-C epoch 30; its 144 state tensors and model config were verified
  identical to the [published inference artifact](../models_archive/README.md).
  The original run used the training checkpoint recorded in the result JSON;
  the current config uses the exported artifact without optimizer state.
- Search: MCTS-1000, `c_puct=1.5`, inference batch size 8, device selection `auto`.
- Opponent: Stockfish 18, `stockfish-windows-x86-64-avx2.exe`, one second per move.
- Schedule: two full-strength sanity games, then 20 games at each `UCI_Elo`
  setting from 1600 through 2500. Sanity games are excluded from the estimate.
- Seed: 0. Standard initial position only; McChess White first, alternating
  colors across the schedule. Draw at 180 ply unless already terminal.
- Hardware: the result did not record a hardware snapshot or the resolved
  `auto` device. The training manifest's hardware is not substituted for this.
- [Config](../configs/eval/stockfish_mcts1000_resnet_c_epoch30_batch8_elo_200games.yaml),
  [published result JSON](data/stockfish_mcts1000_resnet_c_epoch30_batch8_elo_200games.json),
  [model manifest](../models_archive/manifest.json).

## Results

| Population | Games | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|---:|
| All scheduled games | 202 | 150 | 37 | 15 | 83.42% |
| Included handicap games | 200 | 150 | 37 | 13 | 84.25% |
| Full-strength sanity games | 2 | 0 | 0 | 2 | 0.00% |

All games completed; recorded illegal moves: **0**.

| Stockfish `UCI_Elo` | Wins | Draws | Losses | Score |
|---:|---:|---:|---:|---:|
| 1600 | 20 | 0 | 0 | 100.0% |
| 1700 | 18 | 2 | 0 | 95.0% |
| 1800 | 17 | 3 | 0 | 92.5% |
| 1900 | 19 | 1 | 0 | 97.5% |
| 2000 | 17 | 2 | 1 | 90.0% |
| 2100 | 15 | 5 | 0 | 87.5% |
| 2200 | 13 | 4 | 3 | 75.0% |
| 2300 | 11 | 6 | 3 | 70.0% |
| 2400 | 10 | 5 | 5 | 62.5% |
| 2500 | 10 | 9 | 1 | 72.5% |

The saved logistic estimate is **2480**, with a rough 95% interval of
**2408–2557**. This is a local Stockfish-UCI handicap estimate under the stated
limit, not FIDE, Lichess, CCRL, or general engine strength. One opening and one
schedule seed limit the result. The opponent binary's name was recorded, but
the run did not record its checksum.

The earlier [epoch-22 run](2026-07-16-stockfish-mcts1000-resnet-c-epoch22-200games.md)
scored 84.75% with an estimate of 2489 (2415–2566). The intervals overlap and
these runs do not establish a checkpoint improvement or regression. Both
results remain published; epoch 30 was selected by its training schedule.

## Artifact Handling

The full 202 game records, moves, settings, and scores are preserved in the
published JSON. Only the machine-specific parent directories of the Stockfish
executable were removed from `participants.stockfish.path` and
`config.stockfish_path`; its filename is retained. A `publication` object
records that transformation, the source JSON checksum, and both checkpoint
checksums. The [manifest](../models_archive/manifest.json) records the published
JSON checksum. The source run remains unchanged under ignored `runs/`.

Source result SHA-256:
`fd1e746bcb553a44b63366a4f2afca7f3c44c7dc1986c8ceaa7e075d4b0bb749`.

To rerun, copy the linked config with a new `run_id` and `output_dir` so the
existing run is preserved, then set `STOCKFISH_PATH` and invoke
`poetry run python -m mcchess.eval.stockfish PATH_TO_NEW_CONFIG --keep-awake`.
