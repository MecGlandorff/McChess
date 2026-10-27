# Results

Measured results are indexed here; dated reports retain their configs, artifacts,
and limitations. Publication requirements are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
and the [evaluation protocol](docs/EVALUATION_PROTOCOL.md).

## Published Model

The current inference artifact is [ResNet-C epoch 30](models_archive/README.md),
the final checkpoint of the declared 30-epoch schedule. It was selected before
external benchmarking. Its [manifest](models_archive/manifest.json) records
training lineage, checksums, and evaluation references.

## Supervised Evaluation

Both rows use legal-masked policy metrics on 40,000-position held-out test
prefixes. The models, datasets, and training budgets differ; this is a result
index, not a matched architecture comparison.

| Model / training data | Top-1 | Top-3 | Top-5 | Value MSE | Evidence |
|---|---:|---:|---:|---:|---|
| ResNet-C epoch 30 / May 2026, 2000+ games | 0.502250 | 0.783075 | 0.880975 | 0.857878 | [Report](reports/2026-08-11-supervised-resnet-c-epoch30-test-40000.md), [JSON](reports/data/supervised_resnet_c_epoch30_test_40000.json) |
| ResNet 32ch x2 / January 2013, 600k training positions | 0.262 | 0.492 | 0.610 | 0.858 | [Original CPU baseline](reports/2026-06-04-supervised-resnet32x2-lichess2013-01.md) |

The epoch-30 value MSE improves by 8.9% relative to its actual constant-zero
baseline (0.941850). Decisive-position sign accuracy is 0.610182 versus the
majority-sign baseline of 0.503398. The target is the human game's final result,
not an engine position evaluation. See [value evaluation](docs/valuehead_eval.md).

The earlier full-data ResNet-A and ResNet-B training losses and curves remain
in the [May 2026 training comparison](reports/2026-06-11-supervised-resnet-a-b-training.md).

## External Stockfish Benchmarks

Each run below includes 200 handicap games against Stockfish 18 `UCI_Elo`
settings 1600–2500 at one second per opponent move. Two full-strength sanity
games are excluded from these rows. Runs use the standard initial position,
alternating colors, and a 180-ply draw cap. No engine output or benchmark result
is used for training, checkpoint selection, or hyperparameter tuning.

| Checkpoint / search | W / D / L (included games) | Score | Rough local UCI estimate (95% interval) | Report |
|---|---|---:|---|---|
| ResNet-C epoch 30 / MCTS-1000, batch 8 | 150 / 37 / 13 | 84.25% | 2480 (2408–2557) | [2026-08-12](reports/2026-08-12-stockfish-mcts1000-resnet-c-epoch30-200games.md) |
| ResNet-C epoch 22 / MCTS-1000, batch 8 | 153 / 33 / 14 | 84.75% | 2489 (2415–2566) | [2026-07-16](reports/2026-07-16-stockfish-mcts1000-resnet-c-epoch22-200games.md) |
| ResNet-B / MCTS-200 | 95 / 53 / 52 | 60.75% | 2171 (2110–2233) | [2026-06-19](reports/2026-06-19-stockfish-mcts200-resnet-b-200games.md) |

These estimates describe this Stockfish handicap protocol, not FIDE, Lichess,
CCRL, or general engine strength. The epoch-22 and epoch-30 intervals overlap;
these separate runs do not establish an improvement or regression. The smaller
[22-game epoch-30 diagnostic](reports/2026-08-11-stockfish-mcts1000-resnet-c-epoch30-22games.md)
is retained alongside the completed run.

## Internal Diagnostics And Open Questions

- [Tiny training checks](reports/2026-06-01-tiny-loss-smoke.md) retain early
  overfit and validation observations.
- [Policy-only determinism](reports/2026-06-17-policy-arena-determinism.md) and
  the [MCTS-50 smoke result](reports/2026-06-17-mcts-puct-explainer.md) are local
  diagnostics, not broad strength measurements.
- No archival internal MCTS scaling study, matched history/attention ablation,
  search-distillation experiment, or self-play result has been published.
- Supervised results use test prefixes and single training seeds. They do not
  establish full-test performance or multi-seed variance.

Next experiments and their gates are in [ROADMAP.md](ROADMAP.md). Historical
reports remain indexed in [reports/README.md](reports/README.md).
