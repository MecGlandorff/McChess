# ResNet-C Epoch 30: Held-Out Supervised Evaluation

This evaluation applies the bundled epoch-30 model to a deterministic
40,000-position prefix of the held-out test split. It measures agreement with
human moves and final game outcomes; it does not measure Elo or engine strength.

## Protocol

- Run ID: `supervised_resnet_c_epoch30_test_40000`
- Config: `configs/eval/supervised_resnet_c_epoch30_test_40000.yaml`
- Result: `reports/data/supervised_resnet_c_epoch30_test_40000.json`
- Artifact: `models_archive/resnet_c_epoch_030.pt`
- Dataset: May 2026 Lichess standard rated games, both players rated 2000+
- Split: held-out test, deterministic first 40,000 of 734,559 positions
- Device: CPU
- Batch size: 512
- Seed: 20260501
- Started: 2026-08-11 21:02:40 UTC
- Completed: 2026-08-11 21:06:58 UTC
- Elapsed: 257.46 seconds with one math thread to isolate a concurrent benchmark
- Recorded base Git commit: `d5f96740d2519ccb738b2815bb7cab792b07c462`

## Results

| Metric | Value |
|---|---:|
| Legal-masked policy cross-entropy | 1.559681 |
| Legal-masked top-1 accuracy | 0.502250 |
| Legal-masked top-3 accuracy | 0.783075 |
| Legal-masked top-5 accuracy | 0.880975 |
| Raw argmax legal fraction | 0.999350 |
| Value MSE | 0.857878 |
| Constant-zero value MSE | 0.941850 |
| Relative MSE improvement vs zero | 0.089157 |
| Decisive-position sign accuracy | 0.610182 |
| Decisive majority-sign baseline | 0.503398 |

The 40,000 targets contain 18,709 losses, 2,326 draws, and 18,965 wins from the
side-to-move perspective. Explicit legal masking remains mandatory even though
the raw policy argmax happened to be legal for 99.935% of this sample.

## Interpretation Limits

- This is a prefix evaluation, not the full test split.
- Policy accuracy measures agreement with one observed human move; other legal
  moves may also be strong.
- The value target is the final game result, not an engine position score.
- The run occurred from the feature working tree before its commit; the result
  envelope therefore records the branch's base commit.
- Stockfish, Syzygy, and external engine labels were not used in this evaluation
  or in checkpoint selection.
