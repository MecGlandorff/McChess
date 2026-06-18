from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import chess
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from mcchess.board import POLICY_SIZE, move_to_index
from mcchess.model import (
    CheckpointMetadata,
    LoadedPolicyValueCheckpoint,
    ResNetConfig,
)
from mcchess.eval import supervised as supervised_module


def write_eval_shard(path: Path) -> list[dict[str, Any]]:
    board = chess.Board()
    rows = []
    for ply, (uci, value) in enumerate((("e2e4", 1.0), ("e7e5", -1.0))):
        move = chess.Move.from_uci(uci)
        rows.append(
            {
                "game_id": "g000000",
                "ply": ply,
                "fen": board.fen(),
                "move_uci": uci,
                "policy_index": move_to_index(board, move),
                "value": value,
                "result": "1-0",
                "split": "test",
            }
        )
        board.push(move)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return rows


def write_eval_config(
    path: Path,
    *,
    checkpoint_path: Path,
    data_path: Path,
    output_dir: Path,
) -> None:
    config = {
        "checkpoint_path": str(checkpoint_path),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "dataset_manifest_path": "data/manifests/example_manifest.json",
        "split": "test",
        "seed": 11,
        "device": "cpu",
        "batch_size": 2,
        "max_samples": None,
        "num_workers": 0,
        "top_k": [1, 3],
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


class FakePolicyValueModel:
    def __init__(self, policy_indices: list[int], values: list[float]) -> None:
        self.policy_indices = policy_indices
        self.values = values
        self.offset = 0

    def eval(self) -> "FakePolicyValueModel":
        return self

    def __call__(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(board.shape[0])
        logits = torch.full((batch_size, POLICY_SIZE), -20.0, device=board.device)
        values = torch.empty((batch_size,), dtype=torch.float32, device=board.device)
        for row in range(batch_size):
            source_index = self.offset + row
            logits[row, self.policy_indices[source_index]] = 20.0
            values[row] = self.values[source_index]
        self.offset += batch_size
        return logits, values


def fake_loaded_checkpoint(model: object) -> LoadedPolicyValueCheckpoint:
    return LoadedPolicyValueCheckpoint(
        model=cast(Any, model),
        model_config=ResNetConfig(channels=4, num_blocks=1, value_hidden_dim=8),
        metadata=CheckpointMetadata(
            path=Path("checkpoint.pt"),
            epoch=2,
            saved_at="2026-06-07T00:00:00+00:00",
            completed_at="2026-06-07T00:01:00+00:00",
            metrics={"val_total_loss": 1.25},
            train_config={"seed": 11},
        ),
        device=torch.device("cpu"),
    )


def test_supervised_eval_module_writes_reproducible_json(tmp_path: Path, monkeypatch) -> None:
    shard_path = tmp_path / "test.jsonl"
    rows = write_eval_shard(shard_path)
    config_path = tmp_path / "eval.yaml"
    output_dir = tmp_path / "eval_out"
    checkpoint_path = tmp_path / "checkpoint.pt"
    policy_indices = [int(row["policy_index"]) for row in rows]
    fake_model = FakePolicyValueModel(policy_indices, values=[0.8, -0.2])
    monkeypatch.setattr(
        supervised_module,
        "load_policy_value_checkpoint",
        lambda path, device: fake_loaded_checkpoint(fake_model),
    )
    monkeypatch.setattr(supervised_module, "current_git_commit", lambda: "abc123")
    write_eval_config(
        config_path,
        checkpoint_path=checkpoint_path,
        data_path=shard_path,
        output_dir=output_dir,
    )

    result_path = supervised_module.run_evaluation(config_path)

    assert result_path == output_dir / "result.json"
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "source_config_path.txt").read_text(encoding="utf-8") == (
        str(config_path) + "\n"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 2
    assert result["run"]["status"] == "completed"
    assert result["run"]["git_commit"] == "abc123"
    assert result["samples"]["count"] == 2
    assert result["participants"]["checkpoint"]["epoch"] == 2
    assert result["metrics"]["raw_top_k_accuracy"]["1"] == 1.0
    assert result["metrics"]["legal_masked_top_k_accuracy"]["1"] == 1.0
    assert result["metrics"]["raw_argmax_legal_fraction"] == 1.0
    assert result["metrics"]["target_counts"] == {"-1": 1, "0": 0, "1": 1}
    assert result["metrics"]["constant_zero_mse"] == 1.0
    assert result["metrics"]["model_mse"] == pytest.approx(0.34)
    assert result["metrics"]["sign_accuracy_decisive"] == 1.0


def test_eval_top1_script_rejects_illegal_policy_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    shard_path = tmp_path / "test.jsonl"
    rows = write_eval_shard(shard_path)
    rows[0]["policy_index"] = 0
    shard_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = supervised_module.SupervisedEvalConfig(
        checkpoint_path=str(tmp_path / "checkpoint.pt"),
        data_path=str(shard_path),
        output_dir=str(tmp_path / "eval_out"),
        batch_size=2,
        device="cpu",
    )
    fake_model = FakePolicyValueModel([0, int(rows[1]["policy_index"])], values=[0.0, 0.0])
    monkeypatch.setattr(
        supervised_module,
        "load_policy_value_checkpoint",
        lambda path, device: fake_loaded_checkpoint(fake_model),
    )

    with pytest.raises(ValueError, match="policy target is illegal"):
        supervised_module.evaluate_checkpoint(config)
