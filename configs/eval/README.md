# Evaluation Configs

Evaluation and arena configs belong here.

Supervised checkpoint top-k/value evaluation uses `mcchess.eval.supervised` with a
YAML config:

```yaml
checkpoint_path: runs/example/checkpoint.pt
data_path: data/processed/example/test.jsonl
output_dir: runs/example/eval_test
dataset_manifest_path: data/manifests/example_manifest.json
split: test
seed: 0
device: auto
batch_size: 256
max_samples: null
num_workers: 0
top_k: [1, 3, 5]
```

The evaluator uses the JSONL shard, not only tensor caches, because legal-masked
top-k metrics require reconstructing the board from FEN.

Arena bot-vs-bot evaluation uses `mcchess.eval.arena` with a YAML config:

```yaml
run_id: arena_smoke_material_vs_random
output_dir: runs/eval/arena_smoke_material_vs_random
seed: 0
num_games: 20
max_ply: 160
opening_fens: []
agent:
  kind: material
opponent:
  kind: random
```

Supported arena bot kinds are `random`, `material`, `negamax`, `policy_only`,
and `mcts`. `negamax` accepts `depth`; `policy_only` requires
`checkpoint_path` and may set `device`; `mcts` requires `checkpoint_path` and
may set `device`, `simulations`, `c_puct`, and `inference_batch_size`. Arena
wins, draws, losses, and score are recorded from the named `agent` perspective
while colors alternate with the agent playing White first.
If `opening_fens` is set, adjacent games reuse the same FEN with colors swapped;
each game record stores `opening_index` and `starting_fen`.

For a fixed-budget search comparison, run:

```powershell
poetry run python -m mcchess.eval.arena configs/eval/arena_resnet_b_policy_vs_mcts_50.yaml
```

That config compares the local ResNet-B policy-only checkpoint against the same
checkpoint with MCTS-50. The result JSON records the MCTS budget at match level,
including the inference batch size.

For a watchable local demo, an arena config may set `print_moves: true` and
`move_delay_seconds`. This only paces policy-only play; it is not a search or
thinking-time budget. For example,
`configs/eval/arena_watch_resnet_a_vs_resnet_b.yaml` runs the local ResNet-A
checkpoint against the local ResNet-B checkpoint with a four-second pause after
each move.

## External Stockfish Benchmark

`configs/eval/stockfish_mcts200_resnet_b_elo.yaml` runs a short local ResNet-B
MCTS-200 benchmark against a local Stockfish UCI binary. For a report-scale run,
use `configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml`, which keeps
the same protocol but runs 20 games at each Stockfish `UCI_Elo` level from 1600
through 2500. These are external evaluation benchmarks only. Stockfish moves,
evaluations, and game outcomes must not be used for training, labels,
distillation, or checkpoint selection targets.

The MCTS simulation budget and inference batch size are explicit in each
config. For epoch-22 ResNet-C, use
`configs/eval/stockfish_mcts200_resnet_c_epoch22_batch8_elo.yaml` for a
22-game MCTS-200 smoke run,
`configs/eval/stockfish_mcts1000_resnet_c_epoch22_batch8_elo.yaml` for a
22-game MCTS-1000 smoke run, and
`configs/eval/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games.yaml`
for the 202-game report run. Each has a separate run ID and output directory.
The smoke result is only a pipeline check; 20 included games are insufficient
for a reportable rating estimate.

Install Stockfish into an ignored local folder such as `.local/stockfish`, set
`STOCKFISH_PATH`, then run:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo.yaml
```

For the 200-game benchmark:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml
```

For the epoch-22 ResNet-C batch-8 smoke benchmark:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_c_epoch22_batch8_elo.yaml --keep-awake
```

To smoke-test the report search budget itself:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts1000_resnet_c_epoch22_batch8_elo.yaml --keep-awake
```

After the smoke run completes, start the batch-8 report benchmark without the
live GUI:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts1000_resnet_c_epoch22_batch8_elo_200games.yaml --keep-awake
```

To watch moves in the terminal:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo.yaml --print-moves --move-delay-seconds 0.15
```

To open live Python windows for the current board and cumulative game table:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo.yaml --show
```

The runner writes `result.json`, `games.csv`, and `report.md` under the
config's `output_dir`. The short config uses
`runs/external_stockfish/stockfish_mcts200_resnet_b_elo/`; the 200-game config
uses `runs/external_stockfish/stockfish_mcts200_resnet_b_elo_200games/`. The
rough Elo estimate uses only completed `UCI_Elo` games and excludes
full-strength Stockfish sanity games. Interpret each opponent label as a
Stockfish `UCI_Elo` handicap setting under the recorded move limit, for example
`time=1.0s/move`; it is not Lichess Elo, FIDE Elo, or a general
engine-strength claim. The Stockfish runner shows game-level terminal progress
by default; use `--no-progress` for plain log output.

On Windows, `--keep-awake` requests system and display availability only for
the lifetime of the evaluator process. The request is released when the command
finishes or fails. It does not modify the saved power plan or the evaluation
protocol.

Stockfish benchmark configs may also set `opening_fens`; adjacent games within
each Stockfish level reuse the same FEN. If setup fails after the config has
loaded, the runner writes a failed `result.json` before re-raising the error.

To check whether the local Stockfish binary's adjacent `UCI_Elo` levels are
ordered sensibly under the same 1.0 second limit, run:

```powershell
poetry run python -m mcchess.eval.stockfish_ladder configs/eval/stockfish_uci_ladder_selfcheck.yaml
```

The ladder diagnostic is a Stockfish-vs-Stockfish self-check. It is not a
calibration to Lichess, FIDE, CCRL, or McChess strength. It also shows
game-level terminal progress by default and supports `--no-progress`.
