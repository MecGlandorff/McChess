# Training Configs

Training configs belong here.

Training configs may set `resume_from_checkpoint` to continue from a saved
checkpoint. In resume mode, `epochs` is the final target epoch: resuming from an
epoch-3 checkpoint with `epochs: 30` trains epochs 4 through 30. New checkpoints
save optimizer state and `global_step`; older checkpoints can still warm-start
model weights, but the optimizer is reset.

For larger BatchNorm runs, `exclude_norm_bias_from_weight_decay: true` keeps
AdamW weight decay off normalization parameters and biases. `warmup_steps`,
`lr_scheduler: cosine`, and `min_learning_rate` provide a simple per-step
warmup/cosine schedule.

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
- `lichess_2013_01_full_smoke.yaml`: bounded smoke run over the full January
  2013 processed dataset path, intended to validate the full-month data before
  a longer run.
- `lichess_2013_01_full_epoch1.yaml`: first uncapped pass over the full January
  2013 train/validation shards for throughput and data-path validation.
- `lichess_2013_01_full_epoch1_cached.yaml`: CUDA throughput variant of the
  full-month run that reads precomputed tensor caches. This is a derived-artifact
  compute path documented in `adr/0001-tensor-cache-for-cuda-training-throughput.md`.
- `lichess_2026_03_05_2000plus_epoch10_cached.yaml`: cached 10-epoch run over
  the recent March-May 2026 Lichess 2000+ filtered dataset.
- `lichess_2026_05_2000plus_epoch10_cached.yaml`: cached 10-epoch run over the
  May 2026 Lichess 2000+ filtered dataset.
- `lichess_2026_05_2000plus_epoch20_cached_batchmetrics.yaml`: cached 20-epoch
  rerun over the May 2026 Lichess 2000+ filtered dataset with batch-level loss
  metrics enabled.
- `lichess_2026_05_2000plus_resnet_b_epoch20_cached_batchmetrics.yaml`: cached
  20-epoch ResNet-B run over the same May 2026 Lichess 2000+ filtered dataset,
  seed, and optimizer settings as the ResNet-A batch-metrics run.
- `lichess_2026_05_2000plus_resnet_c_epoch20_cached_batchmetrics.yaml`: cached
  20-epoch ResNet-C BatchNorm run over the same May 2026 Lichess 2000+
  filtered dataset, seed, batch size, and optimizer settings as the matched
  ResNet-A and ResNet-B batch-metrics runs.
- `lichess_2026_05_2000plus_resnet_c_epoch3_cached_batchmetrics.yaml`: cached
  3-epoch ResNet-C BatchNorm pilot over the same May 2026 Lichess 2000+
  filtered dataset, seed, batch size, and optimizer settings.
- `lichess_2026_05_2000plus_resnet_c_epoch30_resume_cached_batchmetrics.yaml`:
  resume config that starts from the 3-epoch ResNet-C pilot checkpoint and
  trains through epoch 30 with cosine LR scheduling and no weight decay on
  BatchNorm or bias parameters.
- `lichess_2026_05_2000plus_resnet_c_epoch30_from_epoch12_cached_batchmetrics.yaml`:
  exact final consolidated-run config used for epochs 13 through 30 of the
  published artifact lineage. Rerunning it requires the preserved local output
  directory; the lineage limitation is recorded in `models_archive/manifest.json`.
