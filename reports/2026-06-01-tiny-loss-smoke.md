# Tiny Loss Smoke Checks

Date: 2026-06-01

## Purpose

Verify that the current supervised model, dataset loader, and loss function can
optimize on tiny repeated subsets of real human-game data.

These are development smoke checks, not chess-strength results and not
reportable experiments.

## Data

- Source: Lichess standard rated games, January 2013
- Local raw sample: `data/raw/lichess/lichess_db_standard_rated_2013-01.sample1000.pgn`
- Processed train shard:
  `data/processed/lichess_2013_01_sample1000/train.jsonl`
- Dataset manifest:
  `data/manifests/lichess_2013_01_sample1000_manifest.json`
- Training subset: first 256 train positions

The data path uses human moves, game metadata, and final game results only. PGN
comments and engine evaluations are not used as labels.

## Run 1: 256-Position Inline Smoke

- Model: `PolicyValueResNet`
- Channels: 16
- Residual blocks: 1
- Value hidden dimension: 32
- Batch size: 64
- Optimizer: AdamW
- Learning rate: `3e-3`
- Weight decay: `1e-4`
- Steps: 80
- Seed: `20260601`
- Device: CPU
- Runtime for training loop: 6.6 seconds

### Loss Evolution

| Point | Total Loss | Policy Loss | Value Loss |
|---|---:|---:|---:|
| initial average | 9.4901 | 8.4536 | 1.0365 |
| step 1 | 9.5455 | n/a | n/a |
| step 20 | 6.6774 | n/a | n/a |
| step 40 | 5.9485 | n/a | n/a |
| step 60 | 5.8701 | n/a | n/a |
| step 80 | 5.2410 | n/a | n/a |
| final average | 4.9237 | 4.4497 | 0.4741 |

Change from initial average to final average:

| Metric | Delta |
|---|---:|
| total loss | -4.5664 |
| policy loss | -4.0040 |
| value loss | -0.5625 |

Result:

```text
loss_check=passed
```

### Interpretation

The model can reduce supervised policy/value loss on a tiny repeated subset.
This confirms that the current model, loss, tensor loader, and optimizer path
are compatible enough for the next training milestone.

This does not show generalization, validation improvement, chess strength, or
legal move quality during play.

### Limitations

- No checkpoint was saved.
- No config file was saved.
- No validation split was evaluated.
- The subset is tiny and repeated.
- The run used an ad hoc inline Python command because the supervised training
  script does not exist yet.

## Run 2: 10x Epoch Loss Curve

This run used the new supervised training runner:

```bash
poetry run python scripts/train_supervised.py configs/train/tiny_loss_curve.yaml
```

Artifacts:

- Config copy: `runs/tiny_loss_curve/config.yaml`
- Metrics: `runs/tiny_loss_curve/metrics.jsonl`
- Checkpoint: `runs/tiny_loss_curve/checkpoint.pt`
- Status: `runs/tiny_loss_curve/status.json`
- Loss plot: `runs/tiny_loss_curve/loss.svg`

### Setup

- Model: `PolicyValueResNet`
- Channels: 16
- Residual blocks: 1
- Value hidden dimension: 32
- Training subset: first 2,560 train positions
- Validation subset: first 512 validation positions
- Batch size: 64
- Optimizer: AdamW
- Learning rate: `3e-3`
- Weight decay: `1e-4`
- Epochs: 20
- Seed: `20260601`
- Device: MPS
- Total elapsed time: 26.3 seconds

### Epoch Loss Summary

| Epoch | Train Total | Train Policy | Train Value | Val Total | Val Policy | Val Value |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.4671 | 7.5815 | 0.8856 | 7.8008 | 6.8330 | 0.9678 |
| 5 | 6.5217 | 5.9488 | 0.5729 | 7.6896 | 6.6270 | 1.0626 |
| 10 | 2.8961 | 2.5263 | 0.3698 | 11.7024 | 10.5763 | 1.1261 |
| 15 | 0.9961 | 0.7414 | 0.2547 | 20.9061 | 19.6202 | 1.2859 |
| 20 | 0.5022 | 0.3146 | 0.1876 | 28.0797 | 26.7751 | 1.3046 |

Best validation total loss in this run:

| Epoch | Val Total | Val Policy | Val Value |
|---:|---:|---:|---:|
| 5 | 7.6896 | 6.6270 | 1.0626 |

Change from epoch 1 to epoch 20:

| Metric | Delta |
|---|---:|
| train total loss | -7.9649 |
| train policy loss | -7.2669 |
| train value loss | -0.6980 |
| validation total loss | +20.2789 |
| validation policy loss | +19.9421 |
| validation value loss | +0.3368 |

### Interpretation

The training loss drops sharply, which confirms that the model can fit a larger
bounded subset when trained through the real training runner.

The validation loss rises sharply after the first few epochs. This is expected
for a tiny model repeatedly trained on only 2,560 positions from an ordered
sample, and it should be interpreted as overfitting, not chess improvement.

The run is still useful because it verifies:

- epoch-based training works
- metrics are written
- validation is evaluated
- checkpoint saving works
- status metadata is written
- the SVG loss plot is generated

### Limitations

- The sample is small and ordered.
- The validation subset is also small.
- No legal-move play test was run.
- No arena evaluation was run.
- No claim about chess strength is supported by this run.
- The rising validation loss means this configuration should not be treated as a
  useful checkpoint for playing strength.

## Run 3: Validation Smoke Curve

This run used the less aggressive validation-smoke config:

```bash
poetry run python scripts/train_supervised.py configs/train/validation_smoke.yaml
```

Artifacts:

- Config copy: `runs/validation_smoke/config.yaml`
- Metrics: `runs/validation_smoke/metrics.jsonl`
- Checkpoint: `runs/validation_smoke/checkpoint.pt`
- Status: `runs/validation_smoke/status.json`
- Loss plot: `runs/validation_smoke/loss.svg`

### Setup

- Model: `PolicyValueResNet`
- Channels: 16
- Residual blocks: 1
- Value hidden dimension: 32
- Training subset: first 20,000 train positions
- Validation subset: first 2,033 validation positions
- Batch size: 128
- Optimizer: AdamW
- Learning rate: `5e-4`
- Weight decay: `1e-3`
- Epochs: 5
- Seed: `20260601`
- Device: MPS
- Total recorded elapsed time: 28.6 seconds
- Steady-state elapsed time after first-epoch warmup: 15.5 seconds for epochs
  2-5

### Epoch Loss Summary

| Epoch | Train Total | Train Policy | Train Value | Val Total | Val Policy | Val Value |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.9571 | 7.0148 | 0.9422 | 7.2660 | 6.2724 | 0.9936 |
| 2 | 7.0399 | 6.1122 | 0.9277 | 7.0850 | 6.0750 | 1.0100 |
| 3 | 6.8820 | 5.9584 | 0.9236 | 7.0256 | 6.0217 | 1.0039 |
| 4 | 6.7841 | 5.8710 | 0.9131 | 6.9949 | 6.0000 | 0.9949 |
| 5 | 6.6993 | 5.8037 | 0.8956 | 6.9499 | 5.9697 | 0.9802 |

Change from epoch 1 to epoch 5:

| Metric | Delta |
|---|---:|
| train total loss | -1.2577 |
| train policy loss | -1.2112 |
| train value loss | -0.0466 |
| validation total loss | -0.3161 |
| validation policy loss | -0.3028 |
| validation value loss | -0.0134 |

### Interpretation

This is the first smoke run where both train loss and validation loss decrease.
The improvement is modest but directionally useful: the runner can train on a
larger local subset without immediately overfitting like the tiny 2,560-position
configuration.

This still does not prove chess strength. It only validates that a bounded
supervised run can improve held-out supervised loss over a few epochs.

### Limitations

- The data is still only a 1,000-game Lichess sample.
- The validation split has 2,033 positions, which is useful for smoke testing
  but still small.
- The split is by game, but the source sample itself is the first 1,000 games
  from one monthly archive.
- No legal-move policy-only bot or arena evaluation has been run yet.

## Run 4: Extended Validation Curve

This run used the more extensive local config:

```bash
poetry run python scripts/train_supervised.py configs/train/validation_extended.yaml
```

Artifacts:

- Config copy: `runs/validation_extended/config.yaml`
- Metrics: `runs/validation_extended/metrics.jsonl`
- Checkpoint: `runs/validation_extended/checkpoint.pt`
- Status: `runs/validation_extended/status.json`
- Loss plot: `runs/validation_extended/loss.svg`

### Setup

- Model: `PolicyValueResNet`
- Channels: 32
- Residual blocks: 1
- Value hidden dimension: 64
- Training subset: full local train shard, 55,698 positions
- Validation subset: full local validation shard, 2,033 positions
- Batch size: 128
- Optimizer: AdamW
- Learning rate: `3e-4`
- Weight decay: `1e-3`
- Epochs: 25
- Seed: `20260601`
- Device: MPS
- Total recorded elapsed time: 357.7 seconds

### Epoch Loss Summary

| Epoch | Train Total | Train Policy | Train Value | Val Total | Val Policy | Val Value |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.7169 | 6.7679 | 0.9490 | 7.3424 | 6.3820 | 0.9604 |
| 5 | 6.3343 | 5.4828 | 0.8516 | 6.4334 | 5.4931 | 0.9403 |
| 10 | 5.3379 | 4.5827 | 0.7552 | 6.0701 | 5.1060 | 0.9641 |
| 13 | 4.8796 | 4.1685 | 0.7111 | 5.9418 | 4.9917 | 0.9501 |
| 20 | 4.0285 | 3.3995 | 0.6290 | 6.0547 | 5.0532 | 1.0015 |
| 25 | 3.5878 | 3.0100 | 0.5777 | 6.2109 | 5.1881 | 1.0228 |

Best validation total loss in this run:

| Epoch | Val Total | Val Policy | Val Value |
|---:|---:|---:|---:|
| 13 | 5.9418 | 4.9917 | 0.9501 |

Change from epoch 1 to epoch 25:

| Metric | Delta |
|---|---:|
| train total loss | -4.1291 |
| train policy loss | -3.7579 |
| train value loss | -0.3713 |
| validation total loss | -1.1315 |
| validation policy loss | -1.1939 |
| validation value loss | +0.0624 |

Change from epoch 1 to best validation epoch:

| Metric | Delta |
|---|---:|
| validation total loss | -1.4007 |
| validation policy loss | -1.3904 |
| validation value loss | -0.0103 |

### Interpretation

This is the strongest smoke result so far. It uses the full local 1,000-game
processed train shard and shows clear validation improvement through epoch 13.
After that, train loss keeps improving while validation loss drifts upward,
which suggests mild late overfitting or that this small local dataset has been
mostly exhausted.

This run supports the claim that the supervised training runner can improve
held-out supervised policy loss on a real local dataset. It still does not
support claims about playing strength.

### Limitations

- The dataset is still only the first 1,000 games from one Lichess month.
- The model is still small.
- The best checkpoint is not selected separately; `checkpoint.pt` is the final
  epoch 25 model, not the best-validation epoch 13 model.
- No policy-only bot, legal move selection test, or arena evaluation has been
  run yet.

## Next Step

Add best-validation checkpoint saving, then build a policy-only bot that loads a
checkpoint, masks illegal moves, and verifies that the model chooses legal moves
from real positions.
