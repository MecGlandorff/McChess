# Playing Local Checkpoints

This guide starts notebook widget boards for playing against local McChess
checkpoints. They are for manual inspection and debugging, not arena evaluation
or playing-strength claims.

The policy-only notebook uses explicit legal move masking through
`python-chess`. It does not use MCTS/search, and it does not use the value head
for move selection. The MCTS notebook uses the same legal move rules with a
fixed simulation budget.

## Prerequisites

Install the project with notebook dependencies:

```powershell
poetry install --with dev,notebook
```

The play notebook scans canonical checkpoint files under `runs/`:

```text
checkpoint_latest.pt
checkpoint.pt
```

It selects the checkpoint with the lowest recorded `val_total_loss`. This is a
validation-loss convenience selector, not a playing-strength claim. If no
candidate records that metric, it falls back to the newest completed or modified
checkpoint. If you want to play a specific run, set `checkpoint_path` directly
in `notebooks/play_policy_bot.ipynb`.

## Register The Kernel

If Jupyter does not show `McChess (.venv)` as a kernel, register it once:

```powershell
poetry run python -m ipykernel install --user --name mcchess --display-name "McChess (.venv)"
```

In the notebook UI, choose:

```text
Kernel -> Change Kernel -> McChess (.venv)
```

This matters because the global Python kernel may not have `python-chess`,
`torch`, or the local `mcchess` package installed.

## Start Jupyter

From the repository root:

```powershell
New-Item -ItemType Directory -Force .local\jupyter\nbclassic-runtime, .local\ipython | Out-Null
$env:JUPYTER_RUNTIME_DIR = "$PWD\.local\jupyter\nbclassic-runtime"
$env:IPYTHONDIR = "$PWD\.local\ipython"
poetry run jupyter nbclassic --no-browser --notebook-dir="$PWD" --port=8888 --ServerApp.token=mcchess
```

Leave that PowerShell window open while playing.

Open:

```text
http://127.0.0.1:8888/notebooks/play_policy_bot.ipynb?token=mcchess
```

For the MCTS play notebook, open:

```text
http://127.0.0.1:8888/notebooks/play_mcts_bot.ipynb?token=mcchess
```

For an external Stockfish reference match with MCTS-200, open:

```text
http://127.0.0.1:8888/notebooks/mcts_200_vs_stockfish.ipynb?token=mcchess
```

## Run The Notebook

In `notebooks/play_policy_bot.ipynb`:

1. Select the `McChess (.venv)` kernel.
2. Run all cells from top to bottom.
3. The checkpoint cell should print the selected checkpoint path and device.
4. The final cell should display the clickable board widget with the starting
   position.

The notebook uses CPU inference by default:

```python
inference_device = "cpu"
```

This avoids competing with CUDA training. After training is finished, you can
change it to `"auto"` or `"cuda"` if you want GPU inference, but the current
policy-only model is small enough for CPU play.

## Play The MCTS Bot

`notebooks/play_mcts_bot.ipynb` loads the local ResNet-B checkpoint and wraps it
in the fixed-budget MCTS bot.

The default MCTS setting is:

```python
MCTS_SIMULATIONS = 300
```

This is the number of tree-search simulations per bot move. It is not a fixed
search depth. The notebook checks that the value stays between 200 and 400.

MCTS play is slower than policy-only play. If CUDA is available, the notebook's
default `INFERENCE_DEVICE = "auto"` may use it for model evaluation.

## Play Stockfish

`notebooks/mcts_200_vs_stockfish.ipynb` and `python -m mcchess.eval.stockfish`
run the local ResNet-B MCTS-200 bot against a local Stockfish UCI binary. The
module runner is the preferred route for reproducible saved artifacts. It
alternates colors and saves JSON records under:

```text
runs/external_stockfish/
```

Install Stockfish and either put `stockfish` on `PATH` or set:

```powershell
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
```

Set the path on one line. In PowerShell, a copied path split across a prompt
continuation is treated as a different path.

Run the short benchmark from the terminal:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo.yaml
```

Run the 200-game benchmark:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml
```

Add `--show` to open one Python window for the current board and one for the
cumulative results table:

```powershell
poetry run python -m mcchess.eval.stockfish configs/eval/stockfish_mcts200_resnet_b_elo.yaml --show
```

The Stockfish `UCI_Elo` labels are handicap settings under the recorded search
limit, not Lichess Elo, FIDE Elo, or a general engine-strength claim.

This is an external reference benchmark. Do not use Stockfish moves,
evaluations, game outcomes, or level settings as training labels.

## How To Move

The board is click-source, click-target:

1. Click one of your pieces.
2. Click the destination square.
3. The bot replies immediately if the game is not over.

Promotions default to queen. Clicking another of your pieces reselects it.
Re-running the board cell starts a fresh game.

To play White:

```python
game = NotebookChessGame(bot, human_color=chess.WHITE)
ClickableChessBoard(game)
```

To play Black (the bot opens immediately):

```python
game = NotebookChessGame(bot, human_color=chess.BLACK)
ClickableChessBoard(game)
```

## Troubleshooting

If `import chess` fails with `ModuleNotFoundError`, the notebook is using the
wrong kernel. Select `McChess (.venv)` and restart the kernel.

If the board cell prints `VBox(children=...)` instead of rendering a clickable
board, the frontend is not rendering ipywidgets. Start Jupyter with
`poetry run jupyter nbclassic` from the Poetry environment, then reopen the
notebook.

If Jupyter reports a permission error for `jupyter_cookie_secret`, IPython
history, or `profile_default`, use the local runtime and `IPYTHONDIR` paths
shown in the start command above.

If board rendering looks stale after a code change, restart the kernel and run
all cells again.

If port `8888` is already in use, either stop the old Jupyter process or change
the command and URL to another port, such as `--port=8889`.

## Limitations

- This is a notebook play helper, not a real game UI.
- Moves are clicked, not dragged.
- `play_policy_bot.ipynb` is policy-only and greedy over legally masked logits.
- The MCTS notebook uses fixed simulations per move, not clock time.
- Manual games are useful for qualitative inspection, but they are not
  reportable evaluation results.

