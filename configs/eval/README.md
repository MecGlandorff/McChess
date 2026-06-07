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
