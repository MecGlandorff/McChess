# Configs

YAML configs make data preparation, training, and evaluation reproducible.

| Directory | Contents |
|---|---|
| [data](data/README.md) | PGN sources, filters, splits, and output paths |
| [train](train/README.md) | Supervised training and resume settings |
| [eval](eval/README.md) | Held-out metrics, arena games, and external benchmarks |
| [model](model/README.md) | Selecting built-in architecture presets |

Self-play remains a future milestone. Preserve historical configs referenced
by reports; changed data, checkpoints, or budgets need a distinct run ID and
output directory.
