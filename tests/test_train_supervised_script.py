from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import chess
import torch
import yaml

from mcchess.board import move_to_index


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_supervised.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("train_supervised", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_shard(path: Path, count: int) -> None:
    board = chess.Board()
    rows = []
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    for ply in range(count):
        move = chess.Move.from_uci(moves[ply % len(moves)])
        rows.append(
            {
                "game_id": "g000000",
                "ply": ply,
                "fen": board.fen(),
                "move_uci": move.uci(),
                "policy_index": move_to_index(board, move),
                "value": 1.0 if board.turn == chess.WHITE else -1.0,
                "result": "1-0",
                "split": "train",
            }
        )
        board.push(move)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_supervised_training_script_writes_artifacts(tmp_path: Path) -> None:
    script = load_script_module()
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    write_shard(train_path, count=4)
    write_shard(val_path, count=2)

    config = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "output_dir": str(output_dir),
        "seed": 7,
        "device": "cpu",
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "value_weight": 1.0,
        "num_workers": 0,
        "log_every_steps": 0,
        "model": {
            "channels": 4,
            "num_blocks": 1,
            "value_hidden_dim": 8,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result_dir = script.run_training(config_path)

    assert result_dir == output_dir
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "loss.svg").exists()
    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["train_samples"] == 4
    assert status["val_samples"] == 2

    metrics = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1
    assert metrics[0]["train_total_loss"] > 0
    assert metrics[0]["val_total_loss"] > 0
    assert "<svg" in (output_dir / "loss.svg").read_text(encoding="utf-8")

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    assert checkpoint["model_config"]["channels"] == 4
    assert checkpoint["train_config"]["seed"] == 7
