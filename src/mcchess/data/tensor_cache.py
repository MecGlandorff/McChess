"""Tensor cache for supervised JSONL chess shards.

The JSONL/FEN dataset remains the source of truth. This module writes a local
cache of encoded board tensors and targets to avoid parsing FEN through
`python-chess` inside the training hot path.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

import chess
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from mcchess.board import BOARD_TENSOR_SHAPE, encode_board
from mcchess.data.dataset_builder import DatasetSample
from mcchess.data.torch_dataset import SupervisedTensorSample, iter_jsonl_samples

CACHE_SCHEMA_VERSION = 1
BOARDS_FILENAME = "boards.npy"
POLICY_FILENAME = "policy_indices.npy"
VALUE_FILENAME = "values.npy"
MANIFEST_FILENAME = "manifest.json"
TMP_SUFFIX = ".tmp"


class TensorCacheManifest(TypedDict):
    schema_version: int
    source_path: str
    num_samples: int
    board_shape: list[int]
    board_dtype: str
    policy_shape: list[int]
    policy_dtype: str
    value_shape: list[int]
    value_dtype: str
    created_at: str


def count_jsonl_samples(path: str | Path) -> int:
    """Count non-empty JSONL rows without materializing the shard."""

    shard_path = Path(path)
    count = 0
    with shard_path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def build_supervised_tensor_cache(
    shard_path: str | Path,
    output_dir: str | Path,
    *,
    show_progress: bool = True,
) -> Path:
    """Build a tensor cache from one supervised JSONL shard.

    Boards are stored as `uint8` because every currently documented plane is
    binary. Training casts them to `float32` after moving the batch to the
    selected device.
    """

    shard_path = Path(shard_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest_path.unlink()

    num_samples = count_jsonl_samples(shard_path)
    board_shape = (num_samples, *BOARD_TENSOR_SHAPE)
    boards_path = output_dir / BOARDS_FILENAME
    policy_path = output_dir / POLICY_FILENAME
    values_path = output_dir / VALUE_FILENAME
    tmp_boards_path = _tmp_path(boards_path)
    tmp_policy_path = _tmp_path(policy_path)
    tmp_values_path = _tmp_path(values_path)
    for path in (tmp_boards_path, tmp_policy_path, tmp_values_path):
        if path.exists():
            path.unlink()

    boards = np.lib.format.open_memmap(
        tmp_boards_path,
        mode="w+",
        dtype=np.uint8,
        shape=board_shape,
    )
    policy_indices = np.lib.format.open_memmap(
        tmp_policy_path,
        mode="w+",
        dtype=np.int64,
        shape=(num_samples,),
    )
    values = np.lib.format.open_memmap(
        tmp_values_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples,),
    )

    samples = iter_jsonl_samples(shard_path)
    if show_progress:
        samples = tqdm(
            samples,
            total=num_samples,
            desc=f"caching {shard_path.name}",
            dynamic_ncols=True,
            unit="sample",
        )

    written = 0
    for index, sample in enumerate(samples):
        _write_cached_sample(index, sample, boards, policy_indices, values)
        written = index + 1

    if written != num_samples:
        raise ValueError(f"expected {num_samples} samples but cached {written}")
    boards.flush()
    policy_indices.flush()
    values.flush()
    del boards
    del policy_indices
    del values

    os.replace(tmp_boards_path, boards_path)
    os.replace(tmp_policy_path, policy_path)
    os.replace(tmp_values_path, values_path)

    manifest: TensorCacheManifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_path": str(shard_path),
        "num_samples": num_samples,
        "board_shape": list(board_shape),
        "board_dtype": "uint8",
        "policy_shape": [num_samples],
        "policy_dtype": "int64",
        "value_shape": [num_samples],
        "value_dtype": "float32",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + TMP_SUFFIX)


def _write_cached_sample(
    index: int,
    sample: DatasetSample,
    boards: np.ndarray,
    policy_indices: np.ndarray,
    values: np.ndarray,
) -> None:
    board = chess.Board(sample["fen"])
    encoded = encode_board(board)
    if encoded.shape != BOARD_TENSOR_SHAPE:
        raise ValueError(f"encoded board has unexpected shape {encoded.shape}")
    boards[index] = encoded.astype(np.uint8)
    policy_indices[index] = sample["policy_index"]
    values[index] = sample["value"]


class SupervisedTensorCacheDataset(Dataset):
    """Dataset backed by a precomputed supervised tensor cache."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = self._read_manifest()
        self._validate_cache_files()
        self._boards: np.ndarray | None = None
        self._policy_indices: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.manifest["num_samples"])

    def __getitem__(self, index: int) -> SupervisedTensorSample:
        boards, policy_indices, values = self._arrays()
        board = np.array(boards[index], copy=True)
        return {
            "board": torch.from_numpy(board),
            "policy_index": torch.tensor(policy_indices[index], dtype=torch.long),
            "value": torch.tensor(values[index], dtype=torch.float32),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_boards"] = None
        state["_policy_indices"] = None
        state["_values"] = None
        return state

    def _read_manifest(self) -> TensorCacheManifest:
        manifest_path = self.cache_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing tensor cache manifest: {manifest_path}")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{manifest_path} must contain a JSON object")
        manifest = cast(TensorCacheManifest, raw)
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"{manifest_path} has unsupported schema_version "
                f"{manifest.get('schema_version')}"
            )
        expected_shape = [int(manifest["num_samples"]), *BOARD_TENSOR_SHAPE]
        if manifest.get("board_shape") != expected_shape:
            raise ValueError(f"{manifest_path} has unexpected board_shape")
        if manifest.get("board_dtype") != "uint8":
            raise ValueError(f"{manifest_path} has unexpected board_dtype")
        if manifest.get("policy_shape") != [int(manifest["num_samples"])]:
            raise ValueError(f"{manifest_path} has unexpected policy_shape")
        if manifest.get("policy_dtype") != "int64":
            raise ValueError(f"{manifest_path} has unexpected policy_dtype")
        if manifest.get("value_shape") != [int(manifest["num_samples"])]:
            raise ValueError(f"{manifest_path} has unexpected value_shape")
        if manifest.get("value_dtype") != "float32":
            raise ValueError(f"{manifest_path} has unexpected value_dtype")
        return manifest

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._boards is None:
            boards, policy_indices, values = self._load_arrays()
            self._validate_arrays(boards, policy_indices, values)
            self._boards = boards
            self._policy_indices = policy_indices
            self._values = values
        assert self._policy_indices is not None
        assert self._values is not None
        return self._boards, self._policy_indices, self._values

    def _validate_cache_files(self) -> None:
        boards, policy_indices, values = self._load_arrays()
        self._validate_arrays(boards, policy_indices, values)

    def _load_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.load(self.cache_dir / BOARDS_FILENAME, mmap_mode="r"),
            np.load(self.cache_dir / POLICY_FILENAME, mmap_mode="r"),
            np.load(self.cache_dir / VALUE_FILENAME, mmap_mode="r"),
        )

    def _validate_arrays(
        self,
        boards: np.ndarray,
        policy_indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        num_samples = int(self.manifest["num_samples"])
        expected_board_shape = (num_samples, *BOARD_TENSOR_SHAPE)
        if boards.shape != expected_board_shape:
            raise ValueError(
                f"{self.cache_dir / BOARDS_FILENAME} has shape {boards.shape}, "
                f"expected {expected_board_shape}"
            )
        if boards.dtype != np.uint8:
            raise ValueError(f"{self.cache_dir / BOARDS_FILENAME} has dtype {boards.dtype}")
        if policy_indices.shape != (num_samples,):
            raise ValueError(
                f"{self.cache_dir / POLICY_FILENAME} has shape {policy_indices.shape}, "
                f"expected {(num_samples,)}"
            )
        if policy_indices.dtype != np.int64:
            raise ValueError(f"{self.cache_dir / POLICY_FILENAME} has dtype {policy_indices.dtype}")
        if values.shape != (num_samples,):
            raise ValueError(
                f"{self.cache_dir / VALUE_FILENAME} has shape {values.shape}, "
                f"expected {(num_samples,)}"
            )
        if values.dtype != np.float32:
            raise ValueError(f"{self.cache_dir / VALUE_FILENAME} has dtype {values.dtype}")
