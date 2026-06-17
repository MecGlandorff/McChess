# Policy-Only Arena Determinism Check

Date: 2026-06-17

## Purpose

Check what happens when the local ResNet-A and ResNet-B May 2026 supervised
checkpoints play policy-only games from the standard initial position.

This is a development diagnostic, not a playing-strength result. The run is
useful because it shows a limitation in the current arena setup: deterministic
policy-only bots from one starting position do not produce a broad comparison.

## Setup

- Config: `configs/eval/arena_resnet_a_vs_resnet_b_50.yaml`
- Agent: `resnet_a_policy_only`
- Opponent: `resnet_b_policy_only`
- Agent checkpoint:
  `runs/lichess_2026_05_2000plus_epoch20_cached_batchmetrics/checkpoint.pt`
- Opponent checkpoint:
  `runs/lichess_2026_05_2000plus_resnet_b_epoch20_cached_batchmetrics/checkpoint.pt`
- Games: 50
- Seed: 0
- Max ply: 160
- Move delay: 0.0 seconds
- Printed moves: false
- Color policy: alternating agent colors, agent White first
- Opening protocol: standard initial position
- Move selection: legal-masked policy argmax
- Search: none
- MCTS budget: none
- Result path: `runs/eval/arena_resnet_a_vs_resnet_b_50.json`
- Git commit recorded by the run:
  `0d0edba5a4bd66f0c040b2600c3651077e1028fd`

## Result

| Metric | Value |
|---|---:|
| games completed | 50 |
| ResNet-A wins | 25 |
| draws | 0 |
| ResNet-A losses | 25 |
| ResNet-A score | 0.500 |
| illegal moves | 0 |
| elapsed seconds | 13.0 |

Additional run details:

- All 50 games ended by checkmate.
- Every game result was `0-1`.
- ResNet-A played White 25 times and Black 25 times.
- The run contained 2 unique move sequences.
- Game lengths were 42 and 54 plies.

## Interpretation

The 25/50 score does not show that ResNet-A and ResNet-B have equal playing
strength.

Both bots are deterministic policy-only players. With the standard initial
position and no opening variation, the 50-game match repeats exactly two games:
one with ResNet-A as White and one with ResNet-A as Black. In this run, Black
won both deterministic lines. Alternating colors therefore produces 25 wins and
25 losses from the ResNet-A perspective.

This result is compatible with ResNet-B having better supervised validation
loss. Validation loss measures average human-move prediction over many
positions. This arena run tests only two deterministic trajectories and has no
search to recover from a bad top-1 policy move.

## Conclusion

This run is a useful legality and plumbing check:

- both checkpoints loaded
- legal move masking worked
- no illegal moves were selected
- games completed without arena errors

It is not a useful comparative strength result. The next arena evaluation should
use a fixed opening suite or otherwise vary starting positions. Each opening
should be played with both colors before recording the score as evidence about
relative policy-only play.

Report this run only as a deterministic policy-only check: ResNet-A scored
25/50 against ResNet-B under `arena_resnet_a_vs_resnet_b_50.yaml`.
