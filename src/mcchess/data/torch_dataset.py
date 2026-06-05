"""PyTorch dataset for supervised JSONL chess samples."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import chess
import torch
from torch.utils.data import Dataset

from mcchess.board import BOARD_TENSOR_SHAPE, encode_board
from mcchess.data.dataset_builder import DatasetSample


class SupervisedTensorSample(TypedDict):
    board: torch.Tensor
    policy_index: torch.Tensor
    value: torch.Tensor


def read_jsonl_samples(path: str | Path) -> list[DatasetSample]:
    """Read supervised dataset-builder samples from one JSONL shard."""

    return list(iter_jsonl_samples(path))


def iter_jsonl_samples(path: str | Path, *, start_index: int = 0) -> Iterator[DatasetSample]:
    """Stream supervised dataset-builder samples from one JSONL shard.

    `start_index` skips that many non-empty rows without decoding them. It is
    intended for resuming derived cache builds after the source shard identity
    has already been checked.
    """

    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    shard_path = Path(path)
    sample_index = 0
    with shard_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            if sample_index < start_index:
                sample_index += 1
                continue
            sample = json.loads(line)
            _validate_sample(sample, shard_path, line_number)
            yield sample
            sample_index += 1


class SupervisedChessDataset(Dataset):
    """Dataset that turns JSONL/FEN samples into model-ready tensors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.samples = read_jsonl_samples(self.path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SupervisedTensorSample:
        sample = self.samples[index]
        board = chess.Board(sample["fen"])
        encoded = encode_board(board)
        if encoded.shape != BOARD_TENSOR_SHAPE:
            raise ValueError(f"encoded board has unexpected shape {encoded.shape}")
        return {
            "board": torch.from_numpy(encoded),
            "policy_index": torch.tensor(sample["policy_index"], dtype=torch.long),
            "value": torch.tensor(sample["value"], dtype=torch.float32),
        }


def _validate_sample(sample: object, path: Path, line_number: int) -> None:
    if not isinstance(sample, dict):
        raise ValueError(f"{path}:{line_number} must contain a JSON object")

    required = {
        "game_id",
        "ply",
        "fen",
        "move_uci",
        "policy_index",
        "value",
        "result",
        "split",
    }
    missing = required.difference(sample)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"{path}:{line_number} missing fields: {missing_fields}")
