# Experiments

## Purpose

This file tracks planned, running, and completed experiments.

The project should not only report the best result. It should explain what was tested, why it was tested, and what changed.

## Experiment Naming Convention

Use descriptive names:

- `supervised_resnet64x6_lichess2000`
- `history_resnet64x6_k8`
- `lstm_attention_k8_h128`
- `mcts_distill_resnet64x6_nodes100`

Experiment names should be stable. If a rerun changes data, config, seed, checkpoint, or evaluation protocol, create a new run entry under the same experiment family instead of overwriting the old result.

Each experiment should save:

- config
- git commit if available
- dataset manifest
- random seed
- metrics
- checkpoint path
- evaluation results

## Experiment Registry Schema

Each planned, running, completed, failed, or inconclusive experiment should have:

```yaml
experiment_id:
status: planned | running | completed | failed | aborted | inconclusive
question:
hypothesis:
dataset_manifest:
model_config:
train_config:
eval_config:
seed:
git_commit:
checkpoint_path:
metrics_path:
results_path:
hardware:
started_at:
completed_at:
notes:
```

Use `null` for unavailable fields rather than omitting them.

## Status Discipline

Do not keep only the best run.

Record:

- failed runs that reveal implementation or protocol issues
- negative results that answer a research question
- inconclusive runs where variance or bugs prevent a claim
- follow-up questions created by each result

Move polished result tables to `RESULTS.md` only after the underlying experiment entry has enough metadata to reproduce the result.

## Completed Experiments

### `lichess_2026_05_2000plus_resnet_c_epoch30`

```yaml
experiment_id: lichess_2026_05_2000plus_resnet_c_epoch30
status: completed
question: >
  What supervised policy/value losses does the ResNet-C preset reach after the
  predeclared 30-epoch schedule on the May 2026 Lichess 2000+ dataset?
hypothesis: >
  The larger BatchNorm preset should continue reducing matched validation loss
  through the scheduled run, without implying playing strength.
dataset_manifest: data/manifests/lichess_2026_05_2000plus_manifest.json
model_config: resnet_c preset; exact fields in models_archive/manifest.json
train_config: configs/train/lichess_2026_05_2000plus_resnet_c_epoch30_from_epoch12_cached_batchmetrics.yaml
eval_config: configs/eval/supervised_resnet_c_epoch30_test_40000.yaml
seed: 20260501
git_commit: null
checkpoint_path: models_archive/resnet_c_epoch_030.pt
metrics_path: runs/lichess_2026_05_2000plus_resnet_c_epoch30_from_epoch12_cached_batchmetrics/metrics.jsonl
results_path: reports/2026-08-11-supervised-resnet-c-epoch30-test-40000.md
hardware: NVIDIA RTX 4060 8 GB; 16 GB system RAM
started_at: 2026-08-10T20:28:52.941591+00:00
completed_at: 2026-08-11T09:43:47.602200+00:00
notes: >
  Epoch-30 validation total loss was 2.382666417786772. The published artifact
  is inference-only and was selected because it is the final checkpoint of the
  completed schedule, not because of a Stockfish result. The source run did not
  record a reliable Git commit. On the held-out 40,000-position test prefix,
  legal-masked top-1 was 0.50225 and value MSE was 0.857878. The manifest
  records the consolidated lineage and the overwritten epoch-8 ancestor
  limitation.
```

### `stockfish_mcts200_resnet_b_elo_200games`

```yaml
experiment_id: stockfish_mcts200_resnet_b_elo_200games
status: completed
question: >
  What rough local Stockfish-UCI handicap estimate does the ResNet-B checkpoint
  reach when moves are selected with fixed-budget MCTS-200?
hypothesis: >
  MCTS-200 should produce a stronger local benchmark result than policy-only
  play, but the result should be interpreted only under the exact Stockfish-UCI
  handicap protocol.
dataset_manifest: null
model_config: resnet_b preset from checkpoint metadata
train_config: configs/train/lichess_2026_05_2000plus_resnet_b_epoch20_cached_batchmetrics.yaml
eval_config: configs/eval/stockfish_mcts200_resnet_b_elo_200games.yaml
seed: 0
git_commit: 9e261a35711c08d5afd6ba5a683ad0af563717a7
checkpoint_path: runs/lichess_2026_05_2000plus_resnet_b_epoch20_cached_batchmetrics/checkpoint.pt
metrics_path: runs/external_stockfish/stockfish_mcts200_resnet_b_elo_200games/result.json
results_path: reports/2026-06-19-stockfish-mcts200-resnet-b-200games.md
hardware: null
started_at: 2026-06-18T21:33:59.866225+00:00
completed_at: 2026-06-19T04:50:21.586859+00:00
notes: >
  Completed 202 games: 2 full-strength Stockfish sanity games excluded from the
  estimate and 200 included Stockfish UCI_Elo handicap games from 1600 through
  2500 at time=1.0s/move. McChess scored 95/53/52 over the included games
  for 0.6075. The rough local Stockfish-UCI estimate was 2171 with interval
  2110 to 2233. This is not Lichess Elo, FIDE Elo, CCRL Elo, or a general
  engine-strength claim.
```

## Core Experiment Groups

### Group 1 - Supervised Baseline

Question:

Can a compact policy/value network learn useful chess features from human PGN games?

Models:

- ResNet
- policy-only ResNet
- policy/value ResNet

Metrics:

- policy top-1
- policy top-3
- policy top-5
- value MSE
- value calibration
- training speed
- inference speed

### Group 2 - Representation Ablation

Question:

Does temporal context improve supervised policy/value learning?

Compare:

- single-board encoding
- history-plane encoding
- sequence-of-boards encoding

Metrics:

- policy accuracy
- value MSE
- value calibration
- speed
- parameter count

### Group 3 - Architecture Ablation

Question:

Which architecture gives the best quality under limited compute?

Compare:

- ResNet
- NNUE-style sparse accumulator, if implemented
- History ResNet
- ResNet + square attention
- LSTM history
- LSTM + temporal attention
- Temporal Transformer

Metrics:

- policy top-k
- value calibration
- nodes/sec when used in MCTS
- arena score under fixed MCTS budgets

NNUE-style experiments must use project-defined features and allowed McChess
targets only. Do not use imported Stockfish NNUE weights, engine evaluations,
tablebase labels, or external best-move labels.

### Group 4 - MCTS Scaling

Question:

How does search budget affect playing strength?

Compare:

- policy-only
- MCTS-25
- MCTS-50
- MCTS-100
- MCTS-400

Metrics:

- win/draw/loss
- average game length
- nodes/sec
- score per second

### Group 5 - Search Distillation

Question:

Can MCTS-generated targets improve a compact model without engine labels?

Compare:

- human-supervised policy target
- MCTS visit-count target
- hybrid human + MCTS target

Metrics:

- policy accuracy against human moves
- KL divergence to MCTS policy
- arena strength
- policy-only improvement after distillation

## Experiment Template

```markdown
## Experiment: name_here

### Question

What question does this experiment answer?

### Hypothesis

What do we expect?

### Config

- dataset:
- model:
- training steps:
- seed:
- evaluation:
- git commit:
- hardware:
- checkpoint:
- metrics path:
- results path:

### Results

| Metric | Value |
|---|---:|
| policy top-1 | |
| policy top-3 | |
| value MSE | |
| arena score | |

### Notes

What worked? What failed?

### Follow-up

What should be tested next?
```
