# Value Head Evaluation

The scalar value predicts final human-game result from the side to move:
win `+1`, draw `0`, loss `-1`. This evaluates outcome prediction, not objective
position strength. Use held-out data and no engine or tablebase labels.

## Current Evaluator

`poetry run python -m mcchess.eval.supervised CONFIG.yaml` writes `result.json`
under the configured output directory. For the full result format and split
policy, see [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md#supervised-checkpoint-evaluation).
The [epoch-30 report](../reports/2026-08-11-supervised-resnet-c-epoch30-test-40000.md)
contains the current measured result.

Each evaluation records checkpoint/model metadata, dataset and manifest paths,
split, seed, sample limit, device, config, runtime, and available Git commit.
Use validation for selecting weights, thresholds, or models; test only under
a fixed final configuration. Training-split results are debugging evidence.

## Baselines And Metrics

Always compute baselines from the actual evaluated targets `y`:

```text
constant_zero_mse = mean(y ** 2)
constant_mean_mse = mean((y - mean(y)) ** 2)
relative_improvement = (constant_zero_mse - model_mse) / constant_zero_mse
```

If the zero baseline is zero, relative improvement is undefined. Report the
absolute metrics instead. MSE below 1.0 alone does not establish useful learning.

Report MSE, RMSE, MAE, target counts/mean, and prediction mean/std/min/max and
percentiles (p1, p5, p50, p95, p99). Inspect near-zero (`abs(v) < 0.05`) and
saturated (`abs(v) > 0.95`) predictions. On decisive positions, compare sign
accuracy with the majority-sign baseline. Record any draw-band threshold;
suggested diagnostics are 0.05, 0.15, and 0.25, selected outside the test split.

Calibration uses eight intervals with edges
`[-1, -.75, -.50, -.25, 0, .25, .50, .75, 1]`, including 1 in the last bucket.
For each bucket, inspect count, mean prediction/target, MSE, and decisive sign
accuracy. Report ply buckets `0–9`, `10–19`, `20–39`, `40–79`, and `80+`.

Additional diagnostics, when implemented, should split by side to move, game
result, target class, and chess-only features such as material or remaining
piece count. Do not define slices using engine evaluations. Save failure
examples with FEN, played move, ply, result, target, prediction, absolute error,
and split: confident wrong values, large errors, and near-zero predictions on
late decisive positions.

## Interpretation

Treat performance no better than constant baselines as `not_learning`.
Less than roughly 10% relative MSE improvement, or sign accuracy near the
majority baseline, is a `weak` diagnostic. Better baselines, sign accuracy, and
directional calibration together support `useful`; training-only improvements
suggest `overfit`. Inadequate samples or failed evaluation are `inconclusive`.
These labels are diagnostics, not playing-strength claims. Keep failed results
with their failure reason.

## Follow-Up Experiments

The simplest move-value diagnostic evaluates legal afterstates:

```text
mover_value_after = -V(s_after)
mover_delta = -V(s_after) - V(s)
reranking_score(m) = log_policy(m) + value_scale * (-V(s_after))
```

The sign flip is required because the opponent moves next. Handle terminal
positions through chess rules. Compare the played move's rank under policy,
afterstate value, and combined score; inspect disagreements and preserve the
legal mask. Tune `value_scale` on validation only, then test decision quality
in the internal arena. Human-move agreement alone does not establish strength.

Later controlled ablations may change ply weighting, decisive/draw balance,
value-head capacity, history input, or use a WDL head (`V = P(win) - P(loss)`).
WDL or direct action-value outputs change model/checkpoint contracts and require
their own documented milestone. Human PGNs do not label unplayed alternatives;
future search distillation or self-play must supply any such targets using
McChess only. See [ROADMAP.md](../ROADMAP.md).
