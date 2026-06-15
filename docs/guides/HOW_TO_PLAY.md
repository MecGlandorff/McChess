# Playing A Policy Checkpoint Locally

This guide starts the notebook widget board for playing against a local
policy-only McChess checkpoint. It is for manual inspection and debugging, not
arena evaluation or a playing-strength claim.

The bot uses explicit legal move masking through `python-chess`. It does not use
MCTS/search, and it does not use the value head for move selection.

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
http://127.0.0.1:8888/notebooks/notebooks/play_policy_bot.ipynb?token=mcchess
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
- The current bot is policy-only and greedy over legally masked logits.
- There is no search/MCTS in this notebook path.
- Manual games are useful for qualitative inspection, but they are not
  reportable evaluation results.

