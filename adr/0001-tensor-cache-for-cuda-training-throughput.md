# ADR 0001: Tensor Cache For CUDA Training Throughput

## Status

Accepted

## Date

2026-06-04

## Context

The full January 2013 supervised training run was changed from CPU fallback to
CUDA, but observed GPU utilization stayed low during the normal JSONL training
path. The run was confirmed to use CUDA, but the GPU was mostly waiting while
the CPU prepared each batch:

```text
JSONL row -> FEN string -> python-chess Board -> encode_board -> tensor -> GPU
```

This is expected for the current `SupervisedChessDataset`, which intentionally
keeps JSONL/FEN shards portable and decoupled from encoder revisions. That path
is correct and simple, but it makes the training loop CPU-bound for large CUDA
runs.

Increasing batch size and enabling pinned memory can help transfer overhead,
but they do not remove the per-sample FEN parsing and board encoding bottleneck.
DataLoader workers are risky on Windows for the normal JSONL dataset because
the current dataset materializes the full sample list in memory, and worker
processes may duplicate that large Python object graph.

## Decision

Keep JSONL/FEN shards as the source of truth, and add an optional tensor cache
as a derived compute artifact for throughput-sensitive CUDA runs.

The tensor cache stores:

```text
boards.npy          uint8   [num_samples, 18, 8, 8]
policy_indices.npy  int64   [num_samples]
values.npy          float32 [num_samples]
manifest.json
```

Boards are stored as `uint8` because all current 18 board planes are binary.
Training casts board batches to `float32` on the selected device before model
inference.

The normal full-month config remains the simple JSONL/FEN path. A separate
cached config, `configs/train/lichess_2013_01_full_epoch1_cached.yaml`, uses
cache directories and a distinct output directory so cached throughput runs are
not confused with the normal data-path run.

## Consequences

- CUDA training can avoid `python-chess` and `encode_board` in the hot loop once
  caches are built.
- Cache building is still CPU-bound and may show near-zero GPU utilization.
- The train cache for this dataset is expected to require roughly 9-10 GB of
  disk space.
- Tensor caches are invalidated by any board-encoding shape, plane order, or
  plane semantics change.
- Cached runs must be documented as using a derived artifact and should record
  the cache paths in the copied training config and `status.json`.
- Cache builders write arrays through temporary files and write `manifest.json`
  last, so interrupted rebuilds do not leave a stale manifest pointing at
  partially written arrays.
- Cache readers validate schema version, file presence, array shapes, and dtypes
  before returning cached samples.
- Larger cached batch sizes improve throughput but change optimizer update
  count per epoch, so they are distinct training configs, not silent
  replacements for smaller-batch runs.

## Operational Commands

Build caches:

```powershell
poetry run python scripts\build_tensor_cache.py data\processed\lichess_2013_01_full\train.jsonl data\tensor_cache\lichess_2013_01_full\train
poetry run python scripts\build_tensor_cache.py data\processed\lichess_2013_01_full\val.jsonl data\tensor_cache\lichess_2013_01_full\val
```

Run cached training:

```powershell
poetry run python scripts\train_supervised.py configs\train\lichess_2013_01_full_epoch1_cached.yaml
```

Run normal JSONL/FEN training:

```powershell
poetry run python scripts\train_supervised.py configs\train\lichess_2013_01_full_epoch1.yaml
```

## Alternatives Considered

- Only increase batch size: improves samples per batch but leaves CPU FEN
  parsing and encoding as the dominant bottleneck.
- Only use pinned memory and non-blocking CUDA transfers: reduces transfer
  overhead but does not address data preparation.
- Enable DataLoader workers for the normal JSONL dataset: may help throughput,
  but on Windows it can duplicate the full in-memory sample list.
- Rewrite FEN encoding without `python-chess`: could reduce CPU time, but it
  adds chess-rule risk around en-passant semantics and duplicates rule logic
  that is intentionally delegated to `python-chess`.

## Related Files

- `scripts/build_tensor_cache.py`
- `src/mcchess/data/tensor_cache.py`
- `scripts/train_supervised.py`
- `configs/train/lichess_2013_01_full_epoch1.yaml`
- `configs/train/lichess_2013_01_full_epoch1_cached.yaml`
- `docs/DATASET_PROTOCOL.md`
