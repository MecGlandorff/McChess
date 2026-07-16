# Results

This file records measured results.

Do not put aspirational claims here. Only report results that were actually run.

## Current Status

One reportable supervised baseline run exists:

- `supervised_resnet32x2_lichess2013_01`: a compact ResNet trained on a
  600k-position prefix of the Lichess 2013-01 training shard and evaluated on a
  40k-position held-out test slice.

Two external Stockfish-UCI benchmarks have been recorded:

- `stockfish_mcts200_resnet_b_elo_200games`: ResNet-B MCTS-200 against
  Stockfish 18 `UCI_Elo` handicap levels from 1600 through 2500 under a
  one-second-per-move Stockfish limit.
- `stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games`: ResNet-C epoch 22
  with MCTS-1000 and inference batch size 8 under the same opponent range and
  Stockfish limit.

No archival internal arena, MCTS scaling, search-distillation, self-play, or
matched architecture-ablation result has been promoted to this file yet.

A local working-tree MCTS smoke run is described in
`reports/2026-06-17-mcts-puct-explainer.md`. It should be rerun from committed
code before being treated as an archival result.

## Reporting Standards

Each result should include:

- dataset name and manifest
- model config
- training config
- evaluation config
- random seed
- number of games, if applicable
- MCTS budget, if applicable
- hardware notes, if available
- date run
- checkpoint path or identifier
- metrics path
- results path
- git commit if available
- run status

Do not copy a result into the tables below unless the underlying experiment entry
has enough metadata to reproduce it.

For arena results, also include:

- color policy
- max ply
- draw adjudication rule
- opening protocol
- illegal move count
- search budget or policy-only declaration

## Supervised Learning Results

| Experiment | Model | Train Data | Eval Data | Top-1 | Top-3 | Top-5 | Value MSE |
|---|---|---|---|---:|---:|---:|---:|
| `supervised_resnet32x2_lichess2013_01` | ResNet 32ch x2 (~1.2M params) | Lichess 2013-01, 600k-position train prefix | 40k held-out test positions | 0.262 | 0.492 | 0.610 | 0.858 |

### `supervised_resnet32x2_lichess2013_01`

- status: completed
- question: Can a compact policy/value ResNet learn to predict human moves from
  raw Lichess PGN using only human moves and final game results?
- dataset: Lichess standard rated 2013-01, human moves and final results only
- dataset_manifest: `data/manifests/lichess_2013_01_full_manifest.json`
  (121,332 games, 8,155,187 positions, split by game)
- dataset splits: 7,996,459 train positions, 81,638 validation positions, and
  77,090 test positions
- source_checksum:
  `8963b6a1620a0e9c77e5515a0744ec133e86869487188af047bb0a74400dee37`
- train subset: first 600,000 positions from the train shard, using the
  game-ordered shard prefix
- model_config: `input_planes=18, channels=32, num_blocks=2, policy_size=4672, value_hidden_dim=64`
- train_config: `batch_size=256, epochs=4, optimizer=AdamW, lr=8e-4, weight_decay=1e-3, value_weight=1.0`
- eval_config: held-out test shard, first 40,000 positions, legal-masked top-k
  via `python -m mcchess.eval.supervised`; top-k metrics are computed over legal
  moves only
- seed: 20260601
- git_commit: `f4674bb`
- checkpoint_path: `runs/real_2013_01/checkpoint.pt`
- metrics_path: `runs/real_2013_01/metrics.jsonl`
- results_path: `RESULTS.md`
- hardware: AMD CPU (AMD64 Family 25), PyTorch 2.12.0+cpu, Python 3.11.5;
  approximately 19 minutes wall-clock time
- date_run: 2026-06-04

| Metric | Value |
|---|---:|
| policy top-1, legal-masked | 0.262 |
| policy top-1, raw unmasked | 0.231 |
| policy top-3, legal-masked | 0.492 |
| policy top-5, legal-masked | 0.610 |
| raw argmax is a legal move | 0.888 |
| value MSE, test slice | 0.858 |
| final validation policy CE | 3.113 |
| final validation value MSE | 0.911 |

Notes:

- A uniform policy over 4,672 move indices has cross-entropy of about 8.45. This
  run reached final validation policy CE of 3.113 and legal-masked top-1 of
  0.262 on held-out games.
- The raw unmasked argmax was legal for 88.8% of evaluated positions. This
  suggests the model learned some board-structure regularities, but production
  move selection must still use explicit legal masking.
- This is a deliberately small CPU baseline: 600k of 7,996,459 available
  training positions, 4 epochs, one seed. It is a baseline to improve, not a
  playing strength claim.

## Arena Results

No archival arena evaluation has been promoted to this file yet.

See `reports/2026-06-17-policy-arena-determinism.md` for a deterministic
policy-only diagnostic and `reports/2026-06-17-mcts-puct-explainer.md` for the
first local MCTS-50 smoke result.

| Experiment | Agent | Opponent | Games | Wins | Draws | Losses | Score |
|---|---|---|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## External Stockfish Benchmark Results

These are external evaluation results only. Stockfish moves, evaluations, and
game outcomes are not used as McChess training data, labels, distillation
targets, checkpoint selection targets, or self-play targets.

| Experiment | Agent | Opponent | Included games | W | D | L | Score | Estimate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games` | ResNet-C epoch 22, MCTS-1000 batch 8 | Stockfish 18 `UCI_Elo=1600..2500`, `time=1.0s/move` | 200 | 153 | 33 | 14 | 0.8475 | 2489 |
| `stockfish_mcts200_resnet_b_elo_200games` | ResNet-B MCTS-200 | Stockfish 18 `UCI_Elo=1600..2500`, `time=1.0s/move` | 200 | 95 | 53 | 52 | 0.608 | 2171 |

### `stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games`

- status: completed
- scope: external Stockfish benchmark only
- config:
  `configs/eval/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games.yaml`
- command:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games.yaml --keep-awake
```

- checkpoint:
  `runs/lichess_2026_05_2000plus_resnet_c_epoch30_from_epoch12_cached_batchmetrics/checkpoint_epoch_022.pt`
- agent: `resnet_c_epoch22_mcts_1000_batch8`
- MCTS budget: 1000 simulations per move, `c_puct = 1.5`, inference batch size 8
- opponent: Stockfish 18 through UCI
- Stockfish limit: `time=1.0s/move`
- included opponents: `UCI_Elo=1600` through `2500`, 20 games per level
- excluded sanity games: 2 full-strength Stockfish games
- opening protocol: standard initial position
- color policy: alternating McChess White first over the global game schedule
- max ply: 180
- draw rule: `python_chess_outcome_or_max_ply_draw`
- seed: 0
- git_commit: `a0b1a7a56ba7081e9975282be2b0afde9a4c5d1e`
- runtime: 23,375.3 seconds
- hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB VRAM
- results_path:
  `reports/2026-07-16-stockfish-mcts1000-resnet-c-epoch22-200games.md`
- local artifact path:
  `runs/external_stockfish/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games/result.json`
- date_run: 2026-07-16

| Metric | Value |
|---|---:|
| completed games, all scheduled | 202 / 202 |
| McChess W/D/L, all games | 153 / 33 / 16 |
| McChess score, all games | 0.8391 |
| included `UCI_Elo` handicap games | 200 |
| McChess W/D/L, included games | 153 / 33 / 14 |
| McChess score, included games | 0.8475 |
| rough local Stockfish-UCI estimate | 2489 |
| rough 95% interval | 2415 to 2566 |
| illegal moves | 0 |

Interpretation: the estimate is a local Stockfish `UCI_Elo` handicap benchmark
under the recorded config. It is not an official FIDE rating, Lichess Elo,
CCRL Elo, or a general engine-strength claim.

### `stockfish_mcts200_resnet_b_elo_200games`

- status: completed
- scope: external Stockfish benchmark only
- config: `configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml`
- command:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml
```

- checkpoint:
  `runs/lichess_2026_05_2000plus_resnet_b_epoch20_cached_batchmetrics/checkpoint.pt`
- agent: `resnet_b_mcts_200`
- MCTS budget: 200 simulations per move, `c_puct = 1.5`
- opponent: Stockfish 18 through UCI
- Stockfish limit: `time=1.0s/move`
- included opponents: `UCI_Elo=1600` through `2500`, 20 games per level
- excluded sanity games: 2 full-strength Stockfish games
- opening protocol: standard initial position
- color policy: alternating McChess White first over the global game schedule
- max ply: 180
- draw rule: `python_chess_outcome_or_max_ply_draw`
- seed: 0
- git_commit: `9e261a35711c08d5afd6ba5a683ad0af563717a7`
- results_path:
  `reports/2026-06-19-stockfish-mcts200-resnet-b-200games.md`
- local artifact path:
  `runs/external_stockfish/stockfish_mcts200_resnet_b_elo_200games/result.json`
- date_run: 2026-06-19

| Metric | Value |
|---|---:|
| completed games, all scheduled | 202 / 202 |
| McChess W/D/L, all games | 95 / 53 / 54 |
| McChess score, all games | 0.601 |
| included `UCI_Elo` handicap games | 200 |
| McChess W/D/L, included games | 95 / 53 / 52 |
| McChess score, included games | 0.6075 |
| rough local Stockfish-UCI estimate | 2171 |
| rough 95% interval | 2110 to 2233 |
| illegal moves | 0 |

Interpretation: the estimate is a local Stockfish `UCI_Elo` handicap benchmark
under the recorded config. It is not Lichess Elo, FIDE Elo, CCRL Elo, or a
general engine-strength claim.

## MCTS Scaling Results

No MCTS scaling evaluation has been promoted to this file yet.

The first local MCTS-50 smoke run should be rerun after the MCTS branch is
committed before it is copied into the results table.

| Model | Budget | Games | Score | Nodes/sec | Notes |
|---|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD |

## Architecture Ablation Results

The ResNet row records the supervised baseline above for reference. It is not a
matched architecture ablation because the other architectures have not been run
under the same dataset, seed, and training budget.

| Architecture | Params | History | Attention | Top-1 | Value MSE | Arena Score | Speed |
|---|---:|---|---|---:|---:|---:|---:|
| ResNet | ~1.2M | No | No | 0.262 | 0.858 | TBD | TBD |
| NNUE-style sparse accumulator | TBD | No | No | TBD | TBD | TBD | TBD |
| History ResNet | TBD | Yes | No | TBD | TBD | TBD | TBD |
| ResNet + Square Attention | TBD | No | Yes | TBD | TBD | TBD | TBD |
| LSTM History | TBD | Yes | No | TBD | TBD | TBD | TBD |
| LSTM + Temporal Attention | TBD | Yes | Yes | TBD | TBD | TBD | TBD |
| Temporal Transformer | TBD | Yes | Yes | TBD | TBD | TBD | TBD |

## Failure Analysis

No per-category failure analysis has been run yet.

Future analysis should track examples such as:

- hanging pieces
- missed mate in one
- unsafe captures
- poor king safety
- endgame conversion failures
- repetition or draw-handling mistakes
- tactical horizon problems

The gap between top-1 of 0.262 and top-5 of 0.610 suggests a useful next check:
inspect positions where the human move is in the short list but not ranked first.

## Limitations

- Only one supervised run exists so far.
- The run used 600k of 7,996,459 available training positions for 4 epochs on
  CPU. It is not converged, tuned, or full-data.
- The value head is weak: test MSE is 0.858, compared with about 1.0 for a
  predict-zero baseline.
- The evaluation used a 40k-position slice of the test shard, not the full test
  set.
- No top-k confidence interval or multi-seed variance is reported.
- No arena, MCTS scaling, self-play, search-distillation, or matched
  architecture ablation result has been promoted as archival yet.
