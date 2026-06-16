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

Supported arena bot kinds are `random`, `material`, `negamax`, and
`policy_only`. `negamax` accepts `depth`; `policy_only` requires
`checkpoint_path` and may set `device`. Arena wins, draws, losses, and score
are recorded from the named `agent` perspective while colors alternate with the
agent playing White first.
