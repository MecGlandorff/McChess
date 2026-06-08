from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import chess
import pytest
import torch
import yaml

from mcchess.board import move_to_index
from mcchess.model import RESNET_B


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


def write_train_config(
    path: Path,
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    epochs: int,
) -> None:
    config = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "output_dir": str(output_dir),
        "seed": 7,
        "device": "cpu",
        "batch_size": 2,
        "epochs": epochs,
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
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_make_loader_can_pin_memory() -> None:
    script = load_script_module()
    dataset = torch.utils.data.TensorDataset(torch.zeros(2, 1))

    loader = script.make_loader(
        dataset,
        batch_size=1,
        shuffle=False,
        seed=7,
        num_workers=0,
        pin_memory=True,
    )

    assert loader.pin_memory is True


def test_load_config_accepts_model_preset(tmp_path: Path) -> None:
    script = load_script_module()
    config_path = tmp_path / "config.yaml"
    config = {
        "train_path": "train.jsonl",
        "output_dir": "run",
        "model_preset": "resnet_b",
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = script.load_config(config_path)

    assert loaded.model_preset == "resnet_b"
    assert loaded.model == RESNET_B.config


def test_load_config_rejects_ambiguous_model_and_preset(tmp_path: Path) -> None:
    script = load_script_module()
    config_path = tmp_path / "config.yaml"
    config = {
        "train_path": "train.jsonl",
        "output_dir": "run",
        "model_preset": "resnet_b",
        "model": {"channels": 4, "num_blocks": 1, "value_hidden_dim": 8},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="either model or model_preset"):
        script.load_config(config_path)


def test_make_dataset_uses_tensor_cache_when_configured(tmp_path: Path) -> None:
    script = load_script_module()
    shard = tmp_path / "train.jsonl"
    cache_dir = tmp_path / "cache"
    write_shard(shard, count=2)

    from mcchess.data import build_supervised_tensor_cache

    build_supervised_tensor_cache(shard, cache_dir, show_progress=False)

    dataset = script.make_dataset(shard, cache_dir)

    assert dataset.__class__.__name__ == "SupervisedTensorCacheDataset"


def test_supervised_training_script_writes_artifacts(tmp_path: Path) -> None:
    script = load_script_module()
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    write_shard(train_path, count=4)
    write_shard(val_path, count=2)
    write_train_config(
        config_path,
        train_path=train_path,
        val_path=val_path,
        output_dir=output_dir,
        epochs=1,
    )

    result_dir = script.run_training(config_path)

    assert result_dir == output_dir
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "batch_metrics.jsonl").exists()
    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "checkpoint_latest.pt").exists()
    assert (output_dir / "checkpoint_epoch_001.pt").exists()
    assert (output_dir / "loss.svg").exists()
    assert (output_dir / "batch_loss.svg").exists()
    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["train_samples"] == 4
    assert status["val_samples"] == 2
    assert status["last_completed_epoch"] == 1
    assert status["batch_metrics_path"] == str(output_dir / "batch_metrics.jsonl")
    assert status["batch_loss_plot_path"] == str(output_dir / "batch_loss.svg")
    assert status["latest_checkpoint_path"] == str(output_dir / "checkpoint_latest.pt")
    assert status["latest_epoch_checkpoint_path"] == str(output_dir / "checkpoint_epoch_001.pt")

    metrics = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1
    assert metrics[0]["train_total_loss"] > 0
    assert metrics[0]["val_total_loss"] > 0
    assert "<svg" in (output_dir / "loss.svg").read_text(encoding="utf-8")

    batch_metrics = [
        json.loads(line)
        for line in (output_dir / "batch_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(batch_metrics) == 2
    assert batch_metrics[0]["epoch"] == 1
    assert batch_metrics[0]["step"] == 1
    assert batch_metrics[-1]["global_step"] == 2
    assert batch_metrics[-1]["epoch_samples_seen"] == 4
    assert batch_metrics[-1]["running_train_total_loss"] > 0
    assert "<svg" in (output_dir / "batch_loss.svg").read_text(encoding="utf-8")

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    assert checkpoint["model_config"]["channels"] == 4
    assert checkpoint["train_config"]["seed"] == 7
    assert checkpoint["epoch"] == 1
    assert checkpoint["metrics"]["epoch"] == 1
    assert checkpoint["completed_at"] is not None

    latest_checkpoint = torch.load(output_dir / "checkpoint_latest.pt", map_location="cpu")
    assert latest_checkpoint["epoch"] == 1
    assert latest_checkpoint["completed_at"] == checkpoint["completed_at"]


def test_supervised_training_script_keeps_epoch_checkpoint_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script_module()
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    write_shard(train_path, count=4)
    write_shard(val_path, count=2)
    write_train_config(
        config_path,
        train_path=train_path,
        val_path=val_path,
        output_dir=output_dir,
        epochs=2,
    )

    original_train_one_epoch = script.train_one_epoch
    call_count = 0

    def fail_on_second_epoch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated training interruption")
        return original_train_one_epoch(*args, **kwargs)

    monkeypatch.setattr(script, "train_one_epoch", fail_on_second_epoch)

    with pytest.raises(RuntimeError, match="simulated training interruption"):
        script.run_training(config_path)

    assert not (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "checkpoint_latest.pt").exists()
    assert (output_dir / "checkpoint_epoch_001.pt").exists()
    assert not (output_dir / "checkpoint_epoch_002.pt").exists()
    assert (output_dir / "loss.svg").exists()
    assert (output_dir / "batch_loss.svg").exists()

    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["last_completed_epoch"] == 1
    assert status["latest_epoch_checkpoint_path"] == str(output_dir / "checkpoint_epoch_001.pt")

    metrics = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1

    batch_metrics = [
        json.loads(line)
        for line in (output_dir / "batch_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(batch_metrics) == 2
    assert batch_metrics[-1]["global_step"] == 2

    latest_checkpoint = torch.load(output_dir / "checkpoint_latest.pt", map_location="cpu")
    assert latest_checkpoint["epoch"] == 1
    assert latest_checkpoint["completed_at"] is None
