"""Evaluate supervised checkpoint policy top-k and value metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import chess
import numpy as np
import torch
import torch.nn.functional as F
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset

from mcchess.board import BOARD_TENSOR_SHAPE, POLICY_SIZE, encode_board, legal_policy_mask
from mcchess.data import iter_jsonl_samples
from mcchess.data.dataset_builder import DatasetSample
from mcchess.eval.common import git_commit as current_git_commit
from mcchess.eval.common import load_yaml_mapping, write_json_atomic, write_text_atomic
from mcchess.eval.schema import result_envelope
from mcchess.model import load_policy_value_checkpoint


@dataclass(frozen=True)
class SupervisedEvalConfig:
    checkpoint_path: str
    data_path: str
    output_dir: str
    run_id: str = "supervised_eval"
    dataset_manifest_path: str | None = None
    split: str = "test"
    seed: int = 0
    device: str = "auto"
    batch_size: int = 128
    max_samples: int | None = None
    num_workers: int = 0
    top_k: tuple[int, ...] = (1, 3, 5)
    value_near_zero_threshold: float = 0.05
    value_saturation_threshold: float = 0.95

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        top_k = tuple(int(k) for k in self.top_k)
        if not top_k:
            raise ValueError("top_k must not be empty")
        if any(k <= 0 or k > POLICY_SIZE for k in top_k):
            raise ValueError(f"top_k values must be between 1 and {POLICY_SIZE}")
        object.__setattr__(self, "top_k", tuple(sorted(set(top_k))))

        if self.value_near_zero_threshold < 0:
            raise ValueError("value_near_zero_threshold must be non-negative")
        if self.value_saturation_threshold < 0:
            raise ValueError("value_saturation_threshold must be non-negative")


class EvalTensorSample(TypedDict):
    board: torch.Tensor
    policy_index: torch.Tensor
    value: torch.Tensor
    ply: torch.Tensor
    fen: str


class SupervisedEvalDataset(Dataset):
    """JSONL-backed evaluation dataset that preserves FEN for legal masks."""

    def __init__(self, path: str | Path, *, max_samples: int | None = None) -> None:
        self.path = Path(path)
        self.samples = _read_eval_samples(self.path, max_samples=max_samples)
        if not self.samples:
            raise ValueError(f"{self.path} contains no evaluation samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EvalTensorSample:
        sample = self.samples[index]
        board = chess.Board(sample["fen"])
        encoded = encode_board(board)
        if encoded.shape != BOARD_TENSOR_SHAPE:
            raise ValueError(f"encoded board has unexpected shape {encoded.shape}")
        return {
            "board": torch.from_numpy(encoded),
            "policy_index": torch.tensor(sample["policy_index"], dtype=torch.long),
            "value": torch.tensor(sample["value"], dtype=torch.float32),
            "ply": torch.tensor(sample["ply"], dtype=torch.long),
            "fen": sample["fen"],
        }


def _read_eval_samples(path: Path, *, max_samples: int | None) -> list[DatasetSample]:
    samples = []
    for sample in iter_jsonl_samples(path):
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def load_config(path: str | Path) -> SupervisedEvalConfig:
    raw = load_yaml_mapping(path)
    return SupervisedEvalConfig(**raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate_checkpoint(
    config: SupervisedEvalConfig,
    *,
    git_commit: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    set_seed(config.seed)
    loaded = load_policy_value_checkpoint(config.checkpoint_path, device=config.device)
    device = loaded.device
    model = loaded.model
    model.eval()

    dataset = SupervisedEvalDataset(config.data_path, max_samples=config.max_samples)
    loader = make_loader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    start_time = time.perf_counter()
    accum = _new_accumulators(config.top_k)
    predictions: list[float] = []
    targets: list[float] = []
    plies: list[int] = []
    non_blocking = device.type == "cuda"

    for batch in loader:
        board = batch["board"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        policy_index = batch["policy_index"].to(device, non_blocking=non_blocking)
        value_target = batch["value"].to(device, non_blocking=non_blocking)
        batch_plies = batch["ply"].tolist()
        fens = cast(list[str], batch["fen"])

        policy_logits, value = model(board)
        legal_masks = _legal_masks_for_fens(fens, device)
        _validate_legal_targets(legal_masks, policy_index, fens)
        legal_logits = policy_logits.masked_fill(
            legal_masks <= 0.0,
            torch.finfo(policy_logits.dtype).min,
        )

        _accumulate_policy_metrics(
            accum,
            policy_logits,
            legal_logits,
            legal_masks,
            policy_index,
            config.top_k,
        )
        _accumulate_value_metrics(accum, value, value_target)

        predictions.extend(float(v) for v in value.detach().cpu().tolist())
        targets.extend(float(v) for v in value_target.detach().cpu().tolist())
        plies.extend(int(p) for p in batch_plies)

    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    sample_count = int(accum["sample_count"])
    if sample_count == 0:
        raise ValueError("evaluation produced no samples")

    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    ply_array = np.asarray(plies, dtype=np.int64)
    value_metrics = _value_summary(
        prediction_array,
        target_array,
        ply_array,
        near_zero_threshold=config.value_near_zero_threshold,
        saturation_threshold=config.value_saturation_threshold,
    )

    metrics = {
        "policy_cross_entropy": float(accum["policy_ce_sum"]) / sample_count,
        "legal_masked_policy_cross_entropy": float(accum["legal_policy_ce_sum"]) / sample_count,
        "raw_argmax_legal_fraction": float(accum["raw_argmax_legal_count"]) / sample_count,
        "raw_top_k_accuracy": {
            str(k): int(accum["raw_topk_correct"][k]) / sample_count for k in config.top_k
        },
        "legal_masked_top_k_accuracy": {
            str(k): int(accum["legal_topk_correct"][k]) / sample_count for k in config.top_k
        },
        **value_metrics,
    }
    return result_envelope(
        run_id=config.run_id,
        run_type="supervised",
        status="completed",
        seed=config.seed,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=time.perf_counter() - start_time,
        git_commit=git_commit,
        config_path=config_path,
        config=asdict(config),
        protocol={
            "split": config.split,
            "device": str(device),
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "top_k": list(config.top_k),
            "sample_mode": "prefix",
            "value_near_zero_threshold": config.value_near_zero_threshold,
            "value_saturation_threshold": config.value_saturation_threshold,
        },
        participants={
            "checkpoint": {
                "path": str(config.checkpoint_path),
                "epoch": loaded.metadata.epoch,
                "saved_at": loaded.metadata.saved_at,
                "completed_at": loaded.metadata.completed_at,
                "metrics": loaded.metadata.metrics,
            }
        },
        samples={
            "dataset_path": str(config.data_path),
            "dataset_manifest_path": config.dataset_manifest_path,
            "split": config.split,
            "max_samples": config.max_samples,
            "count": sample_count,
        },
        summary={
            "sample_count": sample_count,
            "policy_top1": metrics["raw_top_k_accuracy"].get("1"),
            "legal_masked_policy_top1": metrics["legal_masked_top_k_accuracy"].get("1"),
            "raw_argmax_legal_fraction": metrics["raw_argmax_legal_fraction"],
            "model_mse": metrics["model_mse"],
            "sign_accuracy_decisive": metrics["sign_accuracy_decisive"],
        },
        metrics=metrics,
    )


def _new_accumulators(top_k: tuple[int, ...]) -> dict[str, Any]:
    return {
        "sample_count": 0,
        "policy_ce_sum": 0.0,
        "legal_policy_ce_sum": 0.0,
        "value_sq_error_sum": 0.0,
        "value_abs_error_sum": 0.0,
        "raw_argmax_legal_count": 0,
        "raw_topk_correct": {k: 0 for k in top_k},
        "legal_topk_correct": {k: 0 for k in top_k},
    }


def _legal_masks_for_fens(fens: list[str], device: torch.device) -> torch.Tensor:
    masks = np.stack([legal_policy_mask(chess.Board(fen)) for fen in fens])
    return torch.from_numpy(masks).to(device=device)


def _validate_legal_targets(
    legal_masks: torch.Tensor,
    policy_index: torch.Tensor,
    fens: list[str],
) -> None:
    target_legal = legal_masks.gather(1, policy_index.unsqueeze(1)).squeeze(1) > 0.0
    if torch.all(target_legal):
        return
    bad_index = int((~target_legal).nonzero(as_tuple=False)[0].item())
    raise ValueError(f"policy target is illegal for FEN: {fens[bad_index]}")


def _accumulate_policy_metrics(
    accum: dict[str, Any],
    policy_logits: torch.Tensor,
    legal_logits: torch.Tensor,
    legal_masks: torch.Tensor,
    policy_index: torch.Tensor,
    top_k: tuple[int, ...],
) -> None:
    batch_size = int(policy_logits.shape[0])
    accum["sample_count"] += batch_size
    accum["policy_ce_sum"] += float(
        F.cross_entropy(policy_logits, policy_index, reduction="sum").item()
    )
    accum["legal_policy_ce_sum"] += float(
        F.cross_entropy(legal_logits, policy_index, reduction="sum").item()
    )

    max_k = max(top_k)
    raw_topk = torch.topk(policy_logits, k=max_k, dim=1).indices
    legal_topk = torch.topk(legal_logits, k=max_k, dim=1).indices
    target = policy_index.unsqueeze(1)

    raw_argmax = raw_topk[:, 0].unsqueeze(1)
    accum["raw_argmax_legal_count"] += int(legal_masks.gather(1, raw_argmax).sum().item())
    for k in top_k:
        accum["raw_topk_correct"][k] += int((raw_topk[:, :k] == target).any(dim=1).sum().item())
        accum["legal_topk_correct"][k] += int(
            (legal_topk[:, :k] == target).any(dim=1).sum().item()
        )


def _accumulate_value_metrics(
    accum: dict[str, Any],
    value: torch.Tensor,
    value_target: torch.Tensor,
) -> None:
    error = value - value_target
    accum["value_sq_error_sum"] += float(torch.sum(error * error).item())
    accum["value_abs_error_sum"] += float(torch.sum(torch.abs(error)).item())


def _value_summary(
    predictions: np.ndarray,
    targets: np.ndarray,
    plies: np.ndarray,
    *,
    near_zero_threshold: float,
    saturation_threshold: float,
) -> dict[str, Any]:
    errors = predictions - targets
    model_mse = float(np.mean(errors * errors))
    model_mae = float(np.mean(np.abs(errors)))
    target_mean = float(np.mean(targets))
    constant_zero_mse = float(np.mean(targets * targets))
    constant_mean_mse = float(np.mean((targets - target_mean) ** 2))
    relative_improvement = (
        (constant_zero_mse - model_mse) / constant_zero_mse
        if constant_zero_mse > 0.0
        else None
    )

    return {
        "target_counts": _target_counts(targets),
        "target_mean": target_mean,
        "constant_zero_mse": constant_zero_mse,
        "constant_mean_mse": constant_mean_mse,
        "model_mse": model_mse,
        "model_rmse": float(np.sqrt(model_mse)),
        "model_mae": model_mae,
        "relative_mse_improvement_vs_zero": relative_improvement,
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
        "prediction_min": float(np.min(predictions)),
        "prediction_p1": float(np.percentile(predictions, 1)),
        "prediction_p5": float(np.percentile(predictions, 5)),
        "prediction_p50": float(np.percentile(predictions, 50)),
        "prediction_p95": float(np.percentile(predictions, 95)),
        "prediction_p99": float(np.percentile(predictions, 99)),
        "prediction_max": float(np.max(predictions)),
        "prediction_near_zero_fraction": float(
            np.mean(np.abs(predictions) < near_zero_threshold)
        ),
        "prediction_saturated_fraction": float(
            np.mean(np.abs(predictions) > saturation_threshold)
        ),
        "sign_accuracy_decisive": _sign_accuracy_decisive(predictions, targets),
        "majority_sign_baseline_decisive": _majority_sign_baseline_decisive(targets),
        "calibration_buckets": _calibration_buckets(predictions, targets),
        "ply_buckets": _ply_buckets(predictions, targets, plies),
    }


def _target_counts(targets: np.ndarray) -> dict[str, int]:
    return {
        "-1": int(np.sum(targets == -1.0)),
        "0": int(np.sum(targets == 0.0)),
        "1": int(np.sum(targets == 1.0)),
    }


def _sign_accuracy_decisive(predictions: np.ndarray, targets: np.ndarray) -> float | None:
    decisive = targets != 0.0
    if not np.any(decisive):
        return None
    return float(np.mean(np.sign(predictions[decisive]) == np.sign(targets[decisive])))


def _majority_sign_baseline_decisive(targets: np.ndarray) -> float | None:
    decisive = targets != 0.0
    decisive_count = int(np.sum(decisive))
    if decisive_count == 0:
        return None
    wins = int(np.sum(targets[decisive] > 0.0))
    losses = int(np.sum(targets[decisive] < 0.0))
    return max(wins, losses) / decisive_count


def _calibration_buckets(predictions: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    edges = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
    buckets = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        if index == len(edges) - 2:
            mask = (predictions >= lower) & (predictions <= upper)
        else:
            mask = (predictions >= lower) & (predictions < upper)
        buckets.append(_bucket_summary(f"{lower:.2f}:{upper:.2f}", predictions, targets, mask))
    return buckets


def _ply_buckets(
    predictions: np.ndarray,
    targets: np.ndarray,
    plies: np.ndarray,
) -> list[dict[str, Any]]:
    ranges = (
        ("0-9", 0, 9),
        ("10-19", 10, 19),
        ("20-39", 20, 39),
        ("40-79", 40, 79),
        ("80+", 80, None),
    )
    buckets = []
    for name, lower, upper in ranges:
        mask = plies >= lower if upper is None else (plies >= lower) & (plies <= upper)
        buckets.append(_bucket_summary(name, predictions, targets, mask))
    return buckets


def _bucket_summary(
    name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {
            "bucket": name,
            "count": 0,
            "prediction_mean": None,
            "target_mean": None,
            "mse": None,
            "sign_accuracy_decisive": None,
        }

    bucket_predictions = predictions[mask]
    bucket_targets = targets[mask]
    errors = bucket_predictions - bucket_targets
    return {
        "bucket": name,
        "count": count,
        "prediction_mean": float(np.mean(bucket_predictions)),
        "target_mean": float(np.mean(bucket_targets)),
        "mse": float(np.mean(errors * errors)),
        "sign_accuracy_decisive": _sign_accuracy_decisive(bucket_predictions, bucket_targets),
    }


def run_evaluation(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    result = evaluate_checkpoint(
        config,
        git_commit=current_git_commit(),
        config_path=str(config_path),
    )
    result_path = write_artifacts(config, result)
    print(f"saved supervised evaluation to {result_path}")
    return result_path


def write_artifacts(config: SupervisedEvalConfig, result: dict[str, Any]) -> Path:
    """Write supervised evaluation artifacts and return the result path."""

    output_dir = Path(config.output_dir)
    config_copy = asdict(config)
    config_copy["top_k"] = list(config.top_k)
    write_text_atomic(
        output_dir / "config.yaml",
        yaml.safe_dump(config_copy, sort_keys=False),
    )
    result_path = output_dir / "result.json"
    write_json_atomic(result_path, result)
    config_path = result.get("run", {}).get("config_path")
    if isinstance(config_path, str):
        write_text_atomic(output_dir / "source_config_path.txt", config_path + "\n")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a supervised McChess checkpoint.")
    parser.add_argument("config", type=Path, help="YAML evaluation config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_evaluation(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
