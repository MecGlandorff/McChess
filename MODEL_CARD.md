# Model Card: ResNet-C Epoch 30

The published artifact is [models_archive/resnet_c_epoch_030.pt](models_archive/resnet_c_epoch_030.pt).
Its [manifest](models_archive/manifest.json) is the source for checksums,
training configuration, data provenance, selection, and evaluation references.

## Model And Training

- 3,600,612 parameters; 128 channels, 10 residual blocks, BatchNorm, and a
  256-unit value hidden layer.
- Trained on May 2026 Lichess standard rated games with both players rated
  at least 2000. Supervision uses human moves and final game results only.
- Joint policy classification and scalar value regression; final checkpoint
  of the predeclared 30-epoch schedule.
- No Stockfish or Leela labels, Syzygy tablebases, external best moves, or
  pretrained chess-engine weights. External benchmarks did not select the model.
- Inference-only export; optimizer and other resume state are omitted.

## Tensor And Rules Contract

| Tensor | Shape / meaning |
|---|---|
| Input | `float32 [batch, 18, 8, 8]`, single-position encoding |
| Policy logits | `float32 [batch, 4672]` |
| Value | `float32 [batch]` in `[-1, 1]`, from the side-to-move perspective |

The model does not determine legality. Use `python-chess` legal moves and an
explicit legal mask before selection or MCTS expansion. Plane order and move
indices are specified in [DESIGN.md](DESIGN.md) and [INVARIANTS.md](INVARIANTS.md).

## Measured Evaluation

On a 40,000-position held-out test prefix, legal-masked top-1/top-3/top-5
accuracy is 0.502250/0.783075/0.880975. Value MSE is 0.857878 against a
constant-zero baseline of 0.941850. See the [supervised report](reports/2026-08-11-supervised-resnet-c-epoch30-test-40000.md).

With MCTS-1000 and inference batch size 8, the checkpoint scored 150 wins,
37 draws, and 13 losses over 200 included Stockfish 18 handicap games at
one second per opponent move. Two additional full-strength sanity games were
losses. See the [external benchmark](reports/2026-08-12-stockfish-mcts1000-resnet-c-epoch30-200games.md)
for the exact protocol and its uncertainty. This is not a FIDE, Lichess, or
general engine-strength measurement.

## Intended Use And Limitations

Use this artifact for local play and research on supervised chess policies and
neural search. It is not authoritative chess analysis.

- Human-move imitation observes one move per position; alternatives may be good.
- Final game outcomes are noisy position-value targets, especially early in games.
- Input has no temporal history beyond state represented in the current encoding.
- Supervised evaluation covers a test prefix and one training seed. External
  games use one initial position and one schedule seed.
- The source training commit is unavailable, and an epoch-8 ancestor was
  overwritten before consolidation. The manifest records the retained lineage.

Run instructions and integrity checks are in [models_archive/README.md](models_archive/README.md).
Future architectures belong in [ROADMAP.md](ROADMAP.md), not this artifact's contract.
