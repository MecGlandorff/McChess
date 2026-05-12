"""Data loading, PGN parsing, and dataset-building utilities."""

from mcchess.data.dataset_builder import BuildConfig, SCHEMA_VERSION, build_dataset
from mcchess.data.pgn_reader import (
    PgnSample,
    ReaderCounters,
    iter_pgn_games,
    iter_samples,
)

__all__ = [
    "BuildConfig",
    "PgnSample",
    "ReaderCounters",
    "SCHEMA_VERSION",
    "build_dataset",
    "iter_pgn_games",
    "iter_samples",
]
