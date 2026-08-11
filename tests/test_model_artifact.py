from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chess
import pytest
import torch

from mcchess.board import BOARD_PLANE_COUNT, POLICY_SIZE
from mcchess.bots import PolicyOnlyBot
from mcchess.model import (
    PolicyValueResNet,
    ResNetConfig,
    export_inference_artifact,
    load_policy_value_checkpoint,
    sha256_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = REPOSITORY_ROOT / "models_archive"
ARCHIVED_MODEL_PATH = ARCHIVE_DIR / "resnet_c_epoch_030.pt"
ARCHIVE_MANIFEST_PATH = ARCHIVE_DIR / "manifest.json"


def _write_training_checkpoint(path: Path, *, train_path: str = "data/train.jsonl") -> None:
    torch.manual_seed(7)
    config = ResNetConfig(channels=4, num_blocks=1, value_hidden_dim=8)
    model = PolicyValueResNet(config)
    optimizer = torch.optim.AdamW(model.parameters())
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": 42,
            "model_config": config.__dict__,
            "train_config": {
                "train_path": train_path,
                "output_dir": "runs/example",
                "seed": 7,
            },
            "epoch": 3,
            "metrics": {"val_total_loss": 1.25},
            "saved_at": "2026-08-11T00:00:00+00:00",
            "completed_at": "2026-08-11T00:01:00+00:00",
        },
        path,
    )


def test_exported_artifact_omits_resume_state_and_preserves_outputs(tmp_path: Path) -> None:
    source_path = tmp_path / "training.pt"
    output_path = tmp_path / "inference.pt"
    _write_training_checkpoint(source_path)

    exported = export_inference_artifact(
        source_path,
        output_path,
        artifact_id="test_epoch_003",
        exported_at="2026-08-11T01:00:00+00:00",
    )

    raw = torch.load(output_path, map_location="cpu", weights_only=True)
    assert "optimizer_state_dict" not in raw
    assert "global_step" not in raw
    assert raw["artifact_schema_version"] == 1
    assert raw["artifact_id"] == "test_epoch_003"
    assert raw["provenance"] == {
        "source_checkpoint_sha256": sha256_file(source_path),
        "exported_at": "2026-08-11T01:00:00+00:00",
    }
    assert exported.output_sha256 == sha256_file(output_path)
    assert exported.output_size_bytes == output_path.stat().st_size

    source = load_policy_value_checkpoint(source_path, device="cpu")
    artifact = load_policy_value_checkpoint(output_path, device="cpu")
    inputs = torch.randn(2, BOARD_PLANE_COUNT, 8, 8)
    with torch.inference_mode():
        source_outputs = source.model(inputs)
        artifact_outputs = artifact.model(inputs)
    torch.testing.assert_close(artifact_outputs[0], source_outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(artifact_outputs[1], source_outputs[1], rtol=0, atol=0)


def test_export_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    source_path = tmp_path / "training.pt"
    output_path = tmp_path / "inference.pt"
    _write_training_checkpoint(source_path)
    output_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_inference_artifact(source_path, output_path, artifact_id="test_epoch_003")

    assert output_path.read_bytes() == b"existing"


def test_export_is_byte_reproducible_for_fixed_metadata(tmp_path: Path) -> None:
    source_path = tmp_path / "training.pt"
    _write_training_checkpoint(source_path)

    first = export_inference_artifact(
        source_path,
        tmp_path / "first.pt",
        artifact_id="test_epoch_003",
        exported_at="2026-08-11T01:00:00+00:00",
    )
    second = export_inference_artifact(
        source_path,
        tmp_path / "second.pt",
        artifact_id="test_epoch_003",
        exported_at="2026-08-11T01:00:00+00:00",
    )

    assert first.output_sha256 == second.output_sha256
    assert first.output_path.read_bytes() == second.output_path.read_bytes()


def test_export_rejects_machine_absolute_training_paths(tmp_path: Path) -> None:
    source_path = tmp_path / "training.pt"
    _write_training_checkpoint(source_path, train_path=str(tmp_path / "train.jsonl"))

    with pytest.raises(ValueError, match="machine-absolute path"):
        export_inference_artifact(
            source_path,
            tmp_path / "inference.pt",
            artifact_id="test_epoch_003",
        )


def test_archived_model_matches_manifest_and_returns_a_legal_move() -> None:
    manifest = json.loads(ARCHIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact_metadata = manifest["artifact"]

    assert artifact_metadata["filename"] == ARCHIVED_MODEL_PATH.name
    assert artifact_metadata["sha256"] == sha256_file(ARCHIVED_MODEL_PATH)
    assert artifact_metadata["size_bytes"] == ARCHIVED_MODEL_PATH.stat().st_size
    assert (ARCHIVE_DIR / "SHA256SUMS").read_text(encoding="utf-8").strip() == (
        f"{artifact_metadata['sha256']}  {ARCHIVED_MODEL_PATH.name}"
    )

    loaded = load_policy_value_checkpoint(ARCHIVED_MODEL_PATH, device="cpu")
    assert asdict(loaded.model_config) == manifest["model"]["config"]
    raw = torch.load(ARCHIVED_MODEL_PATH, map_location="cpu", weights_only=True)
    assert raw["artifact_id"] == artifact_metadata["id"]
    assert raw["artifact_schema_version"] == artifact_metadata["artifact_schema_version"]
    assert raw["provenance"]["source_checkpoint_sha256"] == (
        manifest["source"]["checkpoint_sha256"]
    )
    supervised_evaluation = manifest["evaluation"]["supervised"]
    result_path = REPOSITORY_ROOT / supervised_evaluation["result_path"]
    assert supervised_evaluation["result_sha256"] == sha256_file(result_path)
    inputs = torch.zeros(1, BOARD_PLANE_COUNT, 8, 8)
    with torch.inference_mode():
        policy_logits, value = loaded.model(inputs)
    assert policy_logits.shape == (1, POLICY_SIZE)
    assert value.shape == (1,)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()

    board = chess.Board()
    move = PolicyOnlyBot(loaded).choose_move(board)
    assert move in board.legal_moves
