# Reports

[RESULTS.md](../RESULTS.md) is the current result index. These dated reports
preserve detailed evidence and limitations, including early development checks.
Publication requirements are in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) and
the [evaluation protocol](../docs/EVALUATION_PROTOCOL.md).

## Published Epoch-30 Model

- [Completed 202-game external benchmark](2026-08-12-stockfish-mcts1000-resnet-c-epoch30-200games.md):
  200 included handicap games plus two full-strength sanity games; [JSON](data/stockfish_mcts1000_resnet_c_epoch30_batch8_elo_200games.json).
- [Held-out supervised evaluation](2026-08-11-supervised-resnet-c-epoch30-test-40000.md):
  40,000-position test prefix; [JSON](data/supervised_resnet_c_epoch30_test_40000.json).
- [Earlier 22-game diagnostic](2026-08-11-stockfish-mcts1000-resnet-c-epoch30-22games.md).

## Earlier Results And Diagnostics

- [Epoch-22 MCTS-1000 external benchmark](2026-07-16-stockfish-mcts1000-resnet-c-epoch22-200games.md).
- [ResNet-B MCTS-200 external benchmark](2026-06-19-stockfish-mcts200-resnet-b-200games.md).
- [MCTS/PUCT explanation and local smoke result](2026-06-17-mcts-puct-explainer.md).
- [Policy-only determinism check](2026-06-17-policy-arena-determinism.md).
- [Earlier full-data ResNet-A/B training](2026-06-11-supervised-resnet-a-b-training.md).
- [Early supervised CPU baseline](2026-06-04-supervised-resnet32x2-lichess2013-01.md).
- [Tiny training and validation checks](2026-06-01-tiny-loss-smoke.md).

Development diagnostics do not establish broad playing strength. Historical
reports describe the code and artifacts used at the time; current workflows
are documented in the repository README and guides.
