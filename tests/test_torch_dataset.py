import json

import chess
import pytest
import torch
from torch.utils.data import DataLoader

from mcchess.board import BOARD_TENSOR_SHAPE, move_to_index
from mcchess.data import SupervisedChessDataset, read_jsonl_samples


def write_sample_shard(path) -> list[dict]:
    board = chess.Board()
    rows = []
    for split, uci in (("train", "e2e4"), ("train", "e7e5")):
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
                "split": split,
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


def test_read_jsonl_samples_rejects_missing_required_fields(tmp_path) -> None:
    shard = tmp_path / "bad.jsonl"
    shard.write_text(json.dumps({"fen": chess.STARTING_FEN}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        read_jsonl_samples(shard)
