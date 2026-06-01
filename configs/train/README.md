# Training Configs

Training configs belong here.

Current development configs:

- `tiny_loss_curve.yaml`: small CPU-friendly supervised run over a bounded
  Lichess sample, intended to show epoch-by-epoch loss movement rather than
  chess strength.
- `validation_smoke.yaml`: larger smoke run with a lower learning rate, intended
  to check whether validation loss can move in the right direction before
  longer training.
- `validation_extended.yaml`: more extensive local supervised run over the full
  local 1,000-game sample train/validation shards, intended to run roughly 50x
  longer than `validation_smoke.yaml`.
