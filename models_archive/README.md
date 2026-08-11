# Published Model

`resnet_c_epoch_030.pt` is the canonical McChess inference artifact. It contains
the final model state from the completed 30-epoch ResNet-C run, model
configuration, measured training metadata, and provenance. Optimizer state and
other resume-only fields are intentionally excluded.

From a normal clone:

```powershell
poetry install
poetry run mcchess-play
```

The terminal interface defaults to fixed-budget MCTS with 800 simulations per
move. For faster CPU play, select legally masked policy-only moves explicitly:

```powershell
poetry run mcchess-play --mode policy
```

Use `--color black` to let the model move first, or `--checkpoint PATH` to load
a different compatible checkpoint. Run `poetry run mcchess-play --help` for the
complete interface.

The model never owns chess legality. Both policy-only and MCTS play generate
legal moves and legal policy masks through `python-chess`.

## Integrity

On PowerShell:

```powershell
(Get-FileHash models_archive\resnet_c_epoch_030.pt -Algorithm SHA256).Hash.ToLower()
```

The result must match `SHA256SUMS` and `manifest.json`. The manifest records the
source training checkpoint checksum, dataset provenance, selection rule,
training lineage limitation, and evaluation references.

The publication smoke gate loaded the artifact on CPU, checked finite policy
and value tensors with the documented shapes, and selected a legal opening
move. A separate 40,000-position held-out test-prefix evaluation recorded
legal-masked top-1/top-3/top-5 accuracies of 0.50225/0.783075/0.880975 and value
MSE 0.857878. See the
[evaluation report](../reports/2026-08-11-supervised-resnet-c-epoch30-test-40000.md)
for protocol and interpretation limits.

## Scope

The artifact was trained from human Lichess moves and final game results. It
contains no Stockfish or Leela labels, Syzygy tablebase labels, external engine
weights, optimizer state, or training dataset. Stockfish appears only in a
separate post-training development benchmark and was not used to select this
checkpoint.

The artifact is published so the repository is immediately playable. It is not
presented as the strongest possible checkpoint or as evidence of a general Elo
rating.
