# Value Head Evaluation

## Purpose

Evaluate whether the value head learns useful game-result information beyond
trivial baselines.

This protocol is for supervised value targets generated from human PGNs:

```text
side to move eventually wins: +1
side to move eventually loses: -1
draw: 0
```

It must not use Stockfish, Syzygy, external engine evaluations, or external
best-move labels.

This evaluation does not measure playing strength or Elo. It measures whether
the model value output is useful as a prediction of final game result from the
side-to-move perspective.

## Inputs

A value-head evaluation should record:

- checkpoint path
- model config
- train config, if available
- dataset shard or tensor cache path
- dataset manifest path
- split name: train, val, or test
- sample count
- random seed, if sampling
- max samples, if bounded
- device
- git commit, if available

Use held-out validation or test data for reportable numbers. Training-split
metrics are useful only for debugging overfit behavior.

## Required Baselines

Always report these before interpreting the model:

```text
target_mean = mean(value_target)
constant_zero_mse = mean(value_target ** 2)
constant_mean_mse = mean((value_target - target_mean) ** 2)
```

For a dataset with mostly decisive games, `constant_zero_mse` is approximately
the non-draw target fraction. A value head is not useful merely because its MSE
is below `1.0`; it should be compared with the actual split baseline.

Report absolute and relative improvement:

```text
mse_improvement_vs_zero = constant_zero_mse - model_mse
relative_mse_improvement_vs_zero = mse_improvement_vs_zero / constant_zero_mse
```

## Required Model Metrics

Report:

- MSE
- RMSE
- MAE
- target mean
- prediction mean
- prediction standard deviation
- min, p1, p5, p50, p95, p99, max prediction
- fraction of predictions near zero, for example `abs(value) < 0.05`
- fraction of saturated predictions, for example `abs(value) > 0.95`

The prediction-distribution metrics are important because a value head can have
a modest MSE improvement while still mostly predicting values close to zero.

## Sign Metrics

Report sign accuracy on decisive samples:

```text
decisive = value_target != 0
sign_accuracy = mean(sign(prediction) == sign(value_target))
```

Also report the decisive-sample majority baseline:

```text
majority_sign_baseline = max(count(+1), count(-1)) / count(decisive)
```

If a draw band is used, record the threshold explicitly:

```text
predicted_draw if abs(prediction) < draw_threshold
predicted_win if prediction >= draw_threshold
predicted_loss if prediction <= -draw_threshold
```

Recommended starting thresholds:

- `0.05`
- `0.15`
- `0.25`

Do not tune a threshold on the test split and then report it as a clean test
result.

## Calibration Buckets

Bucket predictions and compare average prediction to average target:

```text
[-1.00, -0.75)
[-0.75, -0.50)
[-0.50, -0.25)
[-0.25,  0.00)
[ 0.00,  0.25)
[ 0.25,  0.50)
[ 0.50,  0.75)
[ 0.75,  1.00]
```

For each bucket, report:

- count
- mean prediction
- mean target
- MSE
- decisive sign accuracy

A useful value head should be at least directionally calibrated: higher
predicted values should generally correspond to better side-to-move outcomes.

## Slice Metrics

Report the required metrics by ply bucket:

```text
0-9
10-19
20-39
40-79
80+
```

Also report by:

- side to move
- game result: `1-0`, `0-1`, `1/2-1/2`
- target class: `-1`, `0`, `+1`

Optional chess-only slices are allowed if they use information from the board
and PGN only:

- material balance bucket from side-to-move perspective
- remaining piece count bucket
- castling-right availability
- en-passant availability

Do not use engine evaluations to define slices.

## Failure Examples

Save a small JSONL file of examples for inspection:

- largest absolute value errors
- confident wrong predictions, such as `prediction > 0.75` with target `-1`
- confident wrong predictions, such as `prediction < -0.75` with target `+1`
- near-zero predictions on decisive late-game positions

Each row should include:

```json
{
  "fen": "",
  "move_uci": "",
  "ply": 0,
  "result": "",
  "target": 0.0,
  "prediction": 0.0,
  "absolute_error": 0.0,
  "split": ""
}
```

These examples are for debugging. Do not infer objective chess evaluation from
them without a legal, documented protocol.

## Output Schema

A value-head evaluation result should be saved as JSON:

```json
{
  "status": "completed",
  "checkpoint_path": "",
  "model_config": {},
  "dataset_path": "",
  "dataset_manifest_path": "",
  "split": "val",
  "sample_count": 0,
  "seed": 0,
  "max_samples": null,
  "device": "cpu",
  "target_counts": {
    "-1": 0,
    "0": 0,
    "1": 0
  },
  "target_mean": 0.0,
  "constant_zero_mse": 0.0,
  "constant_mean_mse": 0.0,
  "model_mse": 0.0,
  "model_rmse": 0.0,
  "model_mae": 0.0,
  "relative_mse_improvement_vs_zero": 0.0,
  "prediction_mean": 0.0,
  "prediction_std": 0.0,
  "sign_accuracy_decisive": 0.0,
  "majority_sign_baseline_decisive": 0.0,
  "calibration_buckets": [],
  "slice_metrics": {},
  "failure_examples_path": null,
  "git_commit": null
}
```

If evaluation fails, keep the result file and set:

```json
{
  "status": "failed",
  "failure_reason": ""
}
```

## Interpretation Guide

Use these labels for internal notes:

- `not_learning`: model MSE is no better than constant baselines.
- `weak`: model beats the zero baseline but by less than about 10% relative MSE
  improvement, or sign accuracy is near the majority-sign baseline.
- `useful`: model beats constant baselines clearly, improves sign accuracy over
  the majority baseline, and calibration buckets are monotonic enough to trust
  for model comparison.
- `overfit`: training value metrics improve while held-out value metrics do not.
- `inconclusive`: sample size is too small, split is not held out, or the
  evaluation failed.

These labels are diagnostic only. They are not playing-strength claims.

## Suggested Improvements

The current value head predicts a position value:

```text
V(s) = expected result from the side-to-move perspective
```

This can already be used to estimate whether a move improves or worsens the
mover's prospects by evaluating the position after the move. For a legal move
`m` from position `s`:

```text
s_after = position after m
mover_value_after = -V(s_after)
mover_delta = mover_value_after - V(s)
```

The sign flip is required because after a legal move it is normally the
opponent's turn, and `V(s_after)` is from the opponent's perspective. Positive
`mover_delta` means the move improved the mover's predicted outcome. Negative
`mover_delta` means the move worsened it.

For White-perspective analysis:

```text
white_value(s) = V(s)   if White is to move
white_value(s) = -V(s)  if Black is to move
white_delta(m) = white_value(s_after) - white_value(s)
```

Positive `white_delta` means the move increased White's predicted outcome.
Negative `white_delta` means it increased Black's predicted outcome.

### Afterstate Move Evaluation

Before changing model architecture, evaluate whether the existing value head can
rank moves through afterstates:

- for each validation position, score every legal move as `-V(s_after)`
- record the rank of the human move by afterstate value
- compare the human move rank from policy logits, afterstate value, and a
  combined policy/value score
- inspect the largest positive and negative value deltas
- save examples where the policy prefers a move but value strongly rejects it

Useful derived metrics:

```text
human_move_value_rank
human_move_value_percentile
best_value_move_delta
policy_top1_value_delta
human_move_value_delta
```

This tests whether value can add decision quality beyond imitation. It still
does not prove playing strength; arena evaluation is required for that.

### Policy/Value Reranking

A simple policy/value reranker can combine human-move plausibility with
afterstate value:

```text
score(m) = log_policy(m) + value_scale * (-V(s_after))
```

The legal move mask remains mandatory. `value_scale` must be tuned only on
validation or development data, not on the final test split.

Reranking should report:

- policy-only top-k metrics
- value-only afterstate top-k metrics
- combined policy/value top-k metrics
- raw argmax legality before masking
- legal-masked selected move legality
- examples where reranking changes the selected move

If reranking improves supervised metrics, it is a candidate for arena testing.
If it hurts supervised metrics but finds plausible tactical improvements, mark
the result as inconclusive until evaluated in games.

### WDL Target

A scalar MSE value head may hide useful structure. A future value head can
predict win/draw/loss probabilities:

```text
P(win), P(draw), P(loss)
```

Derived values:

```text
value = P(win) - P(loss)
expected_score = P(win) + 0.5 * P(draw)
```

This can improve calibration because draws are modeled explicitly instead of
being treated only as scalar `0`. If a WDL head is added, update the model
contract, docs, tests, checkpoint format, and evaluation schema in the same
change.

### Better Value Signal

Final game result is noisy, especially early in the game. A good move in a
lost game still receives a losing target, and a bad move in a won game still
receives a winning target.

Allowed ways to improve the value signal:

- report value metrics by ply bucket before changing training
- upweight later plies where the final result is less noisy
- train a value curriculum on late positions first, then all positions
- balance or separately report decisive and drawn games
- add legal game metadata that is already part of the board state or PGN
- add history encoders in the documented history milestone
- use McChess self-play results when the self-play milestone exists
- use McChess MCTS visit/value targets when the MCTS/distillation milestones
  exist

Disallowed ways:

- Stockfish evaluations
- Syzygy tablebase labels
- Leela evaluations
- external best-move labels
- imported chess-engine value targets

### More Value Capacity

The value task may need more global reasoning than policy imitation. Candidate
model changes to test as controlled ablations:

- larger value hidden dimension
- separate value tower after the shared stem
- deeper or wider residual trunk
- global pooling or attention over squares
- history-board inputs
- side-to-move-preserving temporal models

Any architecture change should include shape tests, parameter count reporting,
and matched-data comparison against the current baseline.

### Direct Move-Value Head

A later architecture may predict action values directly:

```text
Q(s, a): [4672]
```

This would assign a value to each policy move index. It should still use the
explicit legal move mask before selecting moves.

Human PGNs do not provide labels for every legal alternative, so a direct
`Q(s, a)` head should not be trained as if unplayed legal moves are known bad
moves. Better target sources for a direct move-value head are later project
milestones:

- McChess MCTS rollouts or visit/value estimates
- McChess self-play outcomes
- search distillation targets generated by this project

Until those targets exist, afterstate scoring with the scalar value head is the
lowest-risk way to assign move values.

## Current Run Example

In the in-progress `lichess_2026_05_2000plus_epoch10_cached` run, after epoch 5,
the validation target distribution was:

```text
-1: 344,274
 0:  54,287
+1: 348,937
```

The derived held-out baseline was:

```text
constant_zero_mse = 0.9274
```

The logged epoch-5 validation value MSE was:

```text
model_mse = 0.8623
```

That is about a `7.0%` relative MSE improvement over predicting zero for every
position. Under this protocol, that is evidence that the value head is learning
some signal, but the value head should still be treated as weak until sign
accuracy, calibration buckets, and slice metrics are measured.
