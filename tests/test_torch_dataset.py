import json

import chess
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mcchess.board import BOARD_TENSOR_SHAPE, move_to_index
from mcchess.data import (
    SupervisedChessDataset,
    SupervisedTensorCacheDataset,
    build_supervised_tensor_cache,
    iter_jsonl_samples,
    read_jsonl_samples,
)
from mcchess.data.tensor_cache import MANIFEST_FILENAME, PROGRESS_FILENAME, TMP_SUFFIX


def write_sample_shard(
    path,
    moves: tuple[str, ...] = ("e2e4", "e7e5"),
) -> list[dict]:
    board = chess.Board()
    rows = []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        rows.append(
            {
                "game_id": "g000000",
                "ply": len(rows),
                "fen": board.fen(),
                "move_uci": uci,
                "policy_index": move_to_index(board, move),
                "value": 1.0 if board.turn == chess.WHITE else -1.0,
                "result": "1-0",
                "split": "train",
            }
        )
        board.push(move)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return rows


def test_read_jsonl_samples_returns_dataset_rows(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    rows = write_sample_shard(shard)

    samples = read_jsonl_samples(shard)

    assert samples == rows


def test_iter_jsonl_samples_streams_dataset_rows(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    rows = write_sample_shard(shard)

    samples = list(iter_jsonl_samples(shard))

    assert samples == rows


def test_iter_jsonl_samples_can_skip_existing_rows(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    rows = write_sample_shard(shard, ("e2e4", "e7e5", "g1f3"))

    samples = list(iter_jsonl_samples(shard, start_index=1))

    assert samples == rows[1:]


def test_supervised_chess_dataset_returns_model_ready_tensors(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    rows = write_sample_shard(shard)
    dataset = SupervisedChessDataset(shard)

    item = dataset[0]

    assert len(dataset) == 2
    assert item["board"].shape == torch.Size(BOARD_TENSOR_SHAPE)
    assert item["board"].dtype == torch.float32
    assert item["policy_index"].shape == torch.Size([])
    assert item["policy_index"].dtype == torch.long
    assert item["policy_index"].item() == rows[0]["policy_index"]
    assert item["value"].shape == torch.Size([])
    assert item["value"].dtype == torch.float32
    assert item["value"].item() == rows[0]["value"]


def test_supervised_chess_dataset_batches_with_default_dataloader(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    write_sample_shard(shard)
    loader = DataLoader(SupervisedChessDataset(shard), batch_size=2)

    batch = next(iter(loader))

    assert batch["board"].shape == (2, *BOARD_TENSOR_SHAPE)
    assert batch["policy_index"].shape == (2,)
    assert batch["value"].shape == (2,)


def test_supervised_tensor_cache_dataset_returns_cached_tensors(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    rows = write_sample_shard(shard)
    cache_dir = tmp_path / "cache"

    manifest_path = build_supervised_tensor_cache(shard, cache_dir, show_progress=False)
    dataset = SupervisedTensorCacheDataset(cache_dir)
    item = dataset[0]

    assert manifest_path == cache_dir / "manifest.json"
    assert len(dataset) == 2
    assert item["board"].shape == torch.Size(BOARD_TENSOR_SHAPE)
    assert item["board"].dtype == torch.uint8
    assert item["policy_index"].shape == torch.Size([])
    assert item["policy_index"].dtype == torch.long
    assert item["policy_index"].item() == rows[0]["policy_index"]
    assert item["value"].shape == torch.Size([])
    assert item["value"].dtype == torch.float32
    assert item["value"].item() == rows[0]["value"]


def test_supervised_tensor_cache_batches_with_default_dataloader(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    write_sample_shard(shard)
    cache_dir = tmp_path / "cache"
    build_supervised_tensor_cache(shard, cache_dir, show_progress=False)
    loader = DataLoader(SupervisedTensorCacheDataset(cache_dir), batch_size=2)

    batch = next(iter(loader))

    assert batch["board"].shape == (2, *BOARD_TENSOR_SHAPE)
    assert batch["board"].dtype == torch.uint8
    assert batch["policy_index"].shape == (2,)
    assert batch["value"].shape == (2,)


def test_build_supervised_tensor_cache_replaces_stale_manifest_last(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    write_sample_shard(shard)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale_manifest = cache_dir / MANIFEST_FILENAME
    stale_manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")

    build_supervised_tensor_cache(shard, cache_dir, show_progress=False)

    manifest = json.loads(stale_manifest.read_text(encoding="utf-8"))
    assert manifest["num_samples"] == 2
    assert not list(cache_dir.glob(f"*{TMP_SUFFIX}"))
    assert not (cache_dir / PROGRESS_FILENAME).exists()


def test_build_supervised_tensor_cache_resumes_interrupted_build(tmp_path, monkeypatch) -> None:
    import mcchess.data.tensor_cache as tensor_cache

    shard = tmp_path / "train.jsonl"
    rows = write_sample_shard(shard, ("e2e4", "e7e5", "g1f3", "b8c6"))
    cache_dir = tmp_path / "cache"
    original_iter_jsonl_samples = tensor_cache.iter_jsonl_samples

    def interrupted_samples(path, *, start_index: int = 0):
        for sample_index, sample in enumerate(
            original_iter_jsonl_samples(path, start_index=start_index),
            start=start_index,
        ):
            if sample_index == 2:
                raise KeyboardInterrupt
            yield sample

    with monkeypatch.context() as patch:
        patch.setattr(tensor_cache, "iter_jsonl_samples", interrupted_samples)
        with pytest.raises(KeyboardInterrupt):
            tensor_cache.build_supervised_tensor_cache(
                shard,
                cache_dir,
                show_progress=False,
                progress_interval=10,
            )

    progress_path = cache_dir / PROGRESS_FILENAME
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completed_samples"] == 2
    assert not (cache_dir / MANIFEST_FILENAME).exists()
    assert (cache_dir / "boards.npy.tmp").exists()

    seen_start_indices = []

    def recording_samples(path, *, start_index: int = 0):
        seen_start_indices.append(start_index)
        yield from original_iter_jsonl_samples(path, start_index=start_index)

    with monkeypatch.context() as patch:
        patch.setattr(tensor_cache, "iter_jsonl_samples", recording_samples)
        manifest_path = tensor_cache.build_supervised_tensor_cache(
            shard,
            cache_dir,
            show_progress=False,
            progress_interval=10,
        )

    dataset = SupervisedTensorCacheDataset(cache_dir)

    assert seen_start_indices == [2]
    assert manifest_path == cache_dir / MANIFEST_FILENAME
    assert len(dataset) == len(rows)
    assert [dataset[index]["policy_index"].item() for index in range(len(dataset))] == [
        row["policy_index"] for row in rows
    ]
    assert not progress_path.exists()
    assert not list(cache_dir.glob(f"*{TMP_SUFFIX}"))


def test_supervised_tensor_cache_rejects_wrong_array_dtype(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    write_sample_shard(shard)
    cache_dir = tmp_path / "cache"
    build_supervised_tensor_cache(shard, cache_dir, show_progress=False)
    np.save(cache_dir / "values.npy", np.zeros(2, dtype=np.float64))

    with pytest.raises(ValueError, match="values.npy has dtype float64"):
        SupervisedTensorCacheDataset(cache_dir)


def test_supervised_tensor_cache_rejects_missing_array_file(tmp_path) -> None:
    shard = tmp_path / "train.jsonl"
    write_sample_shard(shard)
    cache_dir = tmp_path / "cache"
    build_supervised_tensor_cache(shard, cache_dir, show_progress=False)
    (cache_dir / "policy_indices.npy").unlink()

    with pytest.raises(FileNotFoundError):
        SupervisedTensorCacheDataset(cache_dir)


def test_read_jsonl_samples_rejects_missing_required_fields(tmp_path) -> None:
    shard = tmp_path / "bad.jsonl"
    shard.write_text(json.dumps({"fen": chess.STARTING_FEN}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        read_jsonl_samples(shard)
