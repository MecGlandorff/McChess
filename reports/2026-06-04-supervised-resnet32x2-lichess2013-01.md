# Early Supervised CPU Baseline

This preserves the original recorded result and its limitations; the experiment
was not rerun during documentation cleanup.

- status: completed
- question: Can a compact policy/value ResNet learn to predict human moves from
  raw Lichess PGN using only human moves and final game results?
- dataset: Lichess standard rated 2013-01, human moves and final results only
- dataset_manifest: `data/manifests/lichess_2013_01_full_manifest.json`
  (121,332 games, 8,155,187 positions, split by game)
- dataset splits: 7,996,459 train positions, 81,638 validation positions, and
  77,090 test positions
- source_checksum:
  `8963b6a1620a0e9c77e5515a0744ec133e86869487188af047bb0a74400dee37`
- train subset: first 600,000 positions from the train shard, using the
  game-ordered shard prefix
- model_config: `input_planes=18, channels=32, num_blocks=2, policy_size=4672, value_hidden_dim=64`
- train_config: `batch_size=256, epochs=4, optimizer=AdamW, lr=8e-4, weight_decay=1e-3, value_weight=1.0`
- eval_config: held-out test shard, first 40,000 positions, legal-masked top-k
  via `python -m mcchess.eval.supervised`; top-k metrics are computed over legal
  moves only
- seed: 20260601
- git_commit: `f4674bb`
- checkpoint_path: `runs/real_2013_01/checkpoint.pt`
- metrics_path: `runs/real_2013_01/metrics.jsonl`
- original_results_path: `RESULTS.md` (record moved here during documentation cleanup)
- hardware: AMD CPU (AMD64 Family 25), PyTorch 2.12.0+cpu, Python 3.11.5;
  approximately 19 minutes wall-clock time
- date_run: 2026-06-04

| Metric | Value |
|---|---:|
| policy top-1, legal-masked | 0.262 |
| policy top-1, raw unmasked | 0.231 |
| policy top-3, legal-masked | 0.492 |
| policy top-5, legal-masked | 0.610 |
| raw argmax is a legal move | 0.888 |
| value MSE, test slice | 0.858 |
| final validation policy CE | 3.113 |
| final validation value MSE | 0.911 |

Notes:

- A uniform policy over 4,672 move indices has cross-entropy of about 8.45. This
  run reached final validation policy CE of 3.113 and legal-masked top-1 of
  0.262 on held-out games.
- The raw unmasked argmax was legal for 88.8% of evaluated positions. This
  suggests the model learned some board-structure regularities, but production
  move selection must still use explicit legal masking.
- This is a deliberately small CPU baseline: 600k of 7,996,459 available
  training positions, 4 epochs, one seed. It is a baseline to improve, not a
  playing strength claim.
