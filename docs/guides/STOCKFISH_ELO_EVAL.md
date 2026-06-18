# Stockfish Elo Benchmark

This guide describes how to run McChess MCTS-200 against a local Stockfish UCI
binary from the terminal.

## Scope

This benchmark is allowed only as external evaluation. It plays games against
Stockfish and records McChess results from those games.

Do not use Stockfish moves, evaluations, or game outcomes for:

- training labels
- dataset construction
- value targets
- policy targets
- search distillation
- checkpoint selection targets

The reported number is a local estimate against Stockfish UCI Elo handicap
levels under one config. It is not Lichess Elo, FIDE Elo, or a general claim
about engine strength.

Interpret each opponent level together with the recorded Stockfish search
limit, for example `UCI_Elo=2000` and `time=1.0s/move`. Stockfish's source
describes this handicap range as approximate CCRL Blitz calibration and uses
weakened move selection for limited strength:

```text
https://github.com/official-stockfish/Stockfish/blob/master/src/search.h
https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp
```

## Install Stockfish

Put the Stockfish binary in a repo-local ignored folder:

```powershell
New-Item -ItemType Directory -Force .local\stockfish
```

Download Stockfish from the official site:

```text
https://stockfishchess.org/download/
```

Extract the archive into `.local\stockfish`. The final executable path depends
on the package you downloaded. On Windows it will usually be an `.exe` file.

Find it with:

```powershell
Get-ChildItem .local\stockfish -Recurse -Filter "stockfish*.exe"
```

Set `STOCKFISH_PATH` for the current terminal session:

```powershell
$env:STOCKFISH_PATH = "D:\mcchess\.local\stockfish\path\to\stockfish.exe"
```

Set the path on one line. In PowerShell, a copied path split across a prompt
continuation is treated as a different path.

The `.local/` directory is ignored by git, so the binary should not be
committed.

## Run The Benchmark

From the repo root:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo.yaml
```

That config is a short smoke-scale benchmark. For the 200-game benchmark:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml
```

To watch moves in the terminal:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo.yaml --print-moves --move-delay-seconds 0.15
```

The script shows a game-level progress bar by default. Add `--no-progress` for
plain log output.

To watch the current board and cumulative game table in Python windows:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo.yaml --show
```

The short config runs:

- 2 full-strength Stockfish sanity games, one with each color
- 20 UCI Elo handicap games from 1600 to 2500, two games per level
- McChess ResNet-B with MCTS-200
- Stockfish at 1 second per move
- standard starting position
- max ply draw adjudication at 180 plies

The 200-game config keeps the same protocol but runs 20 games at each
`UCI_Elo` handicap level from 1600 through 2500, for 200 Elo-handicap games
plus the two full-strength sanity games.

The runner writes artifacts to the config's `output_dir`. The short config
writes to:

```text
runs/external_stockfish/stockfish_mcts200_resnet_b_elo/
```

The 200-game config writes to:

```text
runs/external_stockfish/stockfish_mcts200_resnet_b_elo_200games/
```

Expected files:

- `config.yaml`: copied resolved config
- `result.json`: full machine-readable result
- `games.csv`: per-game table
- `report.md`: short report with the game table and rough Elo estimate
- `source_config_path.txt`: original config path

## Interpreting Results

Use `report.md` for the first read. The table is from McChess's perspective and
includes:

- game number
- Stockfish level
- White and Black names
- result
- winner color
- winner name
- McChess score
- whether the game was included in Elo estimation

Full-strength Stockfish sanity games are excluded from Elo estimation. Only
completed games with `UCI_LimitStrength=true` and `UCI_Elo` are included.

If McChess wins every included game or loses every included game, the estimate
is marked as bounded. In that case, expand the tested bracket before reporting a
number as a measured result.

## Check Stockfish UCI_Elo Ordering

To check whether adjacent Stockfish `UCI_Elo` levels are ordered sensibly under
the same local limit:

```powershell
poetry run python -m mcchess.eval.stockfish_ladder configs/eval/stockfish_uci_ladder_selfcheck.yaml
```

This runs paired Stockfish-vs-Stockfish games for adjacent levels from 1600 to
2500. It is a self-consistency check only. It does not calibrate the levels to
Lichess, FIDE, CCRL, or McChess strength. The script shows a game-level
progress bar by default; add `--no-progress` to disable it.

## Promoting A Result

Do not promote old notebook or working-tree runs into `RESULTS.md`.

For a reportable entry, rerun the benchmark from committed code and record:

- config path
- checkpoint path
- Stockfish binary identity
- seed
- git commit
- hardware notes
- number of completed games
- W/D/L and score
- max ply and draw rule
- color policy
- opening protocol
- illegal move count
- output artifact paths

Keep failed or inconclusive runs if they reveal a protocol problem.
