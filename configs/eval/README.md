# Evaluation Configs

Evaluation and arena configs belong here.

Supervised checkpoint top-k/value evaluation uses `scripts/eval_top1.py` with a
YAML config:

```yaml
checkpoint_path: runs/example/checkpoint.pt
data_path: data/processed/example/test.jsonl
output_path: runs/example/eval_test.json
dataset_manifest_path: data/manifests/example_manifest.json
split: test
seed: 0
device: auto
batch_size: 256
max_samples: null
num_workers: 0
top_k: [1, 3, 5]
```

The evaluator uses the JSONL shard, not only tensor caches, because legal-masked
top-k metrics require reconstructing the board from FEN.

Arena bot-vs-bot evaluation uses `scripts/run_arena.py` with a YAML config:

```yaml
run_id: arena_smoke_material_vs_random
output_path: runs/eval/arena_smoke_material_vs_random.json
seed: 0
num_games: 20
max_ply: 160
agent:
  kind: material
opponent:
  kind: random
```

Supported arena bot kinds are `random`, `material`, `negamax`, `policy_only`,
and `mcts`. `negamax` accepts `depth`; `policy_only` requires
`checkpoint_path` and may set `device`; `mcts` requires `checkpoint_path` and
may set `device`, `simulations`, and `c_puct`. Arena wins, draws, losses, and
score are recorded from the named `agent` perspective while colors alternate
with the agent playing White first.

For a fixed-budget search comparison, run:

```powershell
poetry run python scripts/run_arena.py configs/eval/arena_resnet_b_policy_vs_mcts_50.yaml
```

That config compares the local ResNet-B policy-only checkpoint against the same
checkpoint with deterministic MCTS-50. The result JSON records the MCTS budget
at match level.

For a watchable local demo, an arena config may set `print_moves: true` and
`move_delay_seconds`. This only paces policy-only play; it is not a search or
thinking-time budget. For example,
`configs/eval/arena_watch_resnet_a_vs_resnet_b.yaml` runs the local ResNet-A
checkpoint against the local ResNet-B checkpoint with a four-second pause after
each move.
