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
