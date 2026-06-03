"""Data loading, PGN parsing, and dataset-building utilities."""

from mcchess.data.dataset_builder import build_dataset
from mcchess.data.pgn_reader import iter_samples, new_counters
from mcchess.data.torch_dataset import (
    SupervisedChessDataset,
    SupervisedTensorSample,
    iter_jsonl_samples,
    read_jsonl_samples,
)
from mcchess.data.tensor_cache import (
    SupervisedTensorCacheDataset,
    build_supervised_tensor_cache,
    count_jsonl_samples,
)

__all__ = [
    "SupervisedChessDataset",
    "SupervisedTensorCacheDataset",
    "SupervisedTensorSample",
    "build_supervised_tensor_cache",
    "build_dataset",
    "count_jsonl_samples",
    "iter_samples",
    "iter_jsonl_samples",
    "new_counters",
    "read_jsonl_samples",
]
