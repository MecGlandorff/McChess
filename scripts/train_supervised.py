#!/usr/bin/env python3
"""Train the supervised policy/value baseline from JSONL dataset shards."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import shutil
import time
from collections.abc import Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from mcchess.data import SupervisedChessDataset, SupervisedTensorCacheDataset
from mcchess.model import PolicyValueResNet, ResNetConfig, policy_value_loss


@dataclass(frozen=True)
class SupervisedTrainConfig:
    train_path: str
    output_dir: str
    val_path: str | None = None
    train_cache_dir: str | None = None
    val_cache_dir: str | None = None
    dataset_manifest_path: str | None = None
    seed: int = 0
    device: str = "auto"
    batch_size: int = 64
    epochs: int = 1
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    num_workers: int = 0
    log_every_steps: int = 20
    model: ResNetConfig | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive when set")
        if self.max_val_samples is not None and self.max_val_samples <= 0:
            raise ValueError("max_val_samples must be positive when set")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.value_weight < 0:
            raise ValueError("value_weight must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.log_every_steps < 0:
            raise ValueError("log_every_steps must be non-negative")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_policy_loss: float
    train_value_loss: float
    val_total_loss: float | None
    val_policy_loss: float | None
    val_value_loss: float | None
    elapsed_seconds: float


def load_config(path: str | Path) -> SupervisedTrainConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    model_raw = raw.pop("model", None)
    model_config = ResNetConfig(**model_raw) if isinstance(model_raw, dict) else ResNetConfig()
    return SupervisedTrainConfig(**raw, model=model_config)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    dataset_size = len(cast(Sized, dataset))
    if max_samples is None or max_samples >= dataset_size:
        return dataset
    return Subset(dataset, range(max_samples))


def make_dataset(path: str | Path, cache_dir: str | Path | None = None) -> Dataset:
    """Create the supervised dataset, preferring a precomputed cache when set."""

    if cache_dir is not None:
        return SupervisedTensorCacheDataset(cache_dir)
    return SupervisedChessDataset(path)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


def train_one_epoch(
    model: PolicyValueResNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_weight: float,
    *,
    epoch: int,
    log_every_steps: int,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    total_samples = 0
    progress = tqdm(
        loader,
        desc=f"epoch {epoch:03d} train",
        dynamic_ncols=True,
        unit="batch",
    )

    non_blocking = device.type == "cuda"
    for step, batch in enumerate(progress, start=1):
        board = batch["board"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        policy_index = batch["policy_index"].to(device, non_blocking=non_blocking)
        value_target = batch["value"].to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        policy_logits, value = model(board)
        losses = policy_value_loss(
            policy_logits,
            value,
            policy_index,
            value_target,
            value_weight=value_weight,
        )
        losses.total.backward()
        optimizer.step()

        batch_size = int(board.shape[0])
        total_samples += batch_size
        total_loss += losses.total.item() * batch_size
        policy_loss += losses.policy.item() * batch_size
        value_loss += losses.value.item() * batch_size

        if log_every_steps == 0 or step % log_every_steps == 0 or step == len(loader):
            progress.set_postfix(
                total=f"{losses.total.item():.4f}",
                policy=f"{losses.policy.item():.4f}",
                value=f"{losses.value.item():.4f}",
            )

    return (
        total_loss / total_samples,
        policy_loss / total_samples,
        value_loss / total_samples,
    )


@torch.no_grad()
def evaluate(
    model: PolicyValueResNet,
    loader: DataLoader,
    device: torch.device,
    value_weight: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    total_samples = 0
    progress = tqdm(
        loader,
        desc="validation",
        dynamic_ncols=True,
        unit="batch",
    )

    non_blocking = device.type == "cuda"
    for batch in progress:
        board = batch["board"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        policy_index = batch["policy_index"].to(device, non_blocking=non_blocking)
        value_target = batch["value"].to(device, non_blocking=non_blocking)
        policy_logits, value = model(board)
        losses = policy_value_loss(
            policy_logits,
            value,
            policy_index,
            value_target,
            value_weight=value_weight,
        )

        batch_size = int(board.shape[0])
        total_samples += batch_size
        total_loss += losses.total.item() * batch_size
        policy_loss += losses.policy.item() * batch_size
        value_loss += losses.value.item() * batch_size
        progress.set_postfix(
            total=f"{losses.total.item():.4f}",
            policy=f"{losses.policy.item():.4f}",
            value=f"{losses.value.item():.4f}",
        )

    return (
        total_loss / total_samples,
        policy_loss / total_samples,
        value_loss / total_samples,
    )


def json_ready_config(config: SupervisedTrainConfig, device: torch.device) -> dict[str, Any]:
    data = asdict(config)
    data["model"] = asdict(config.model or ResNetConfig())
    data["resolved_device"] = str(device)
    return data


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def write_loss_svg(metrics: list[EpochMetrics], output_path: Path) -> None:
    """Write a small dependency-free SVG line chart for train/validation loss."""

    width = 820
    height = 460
    margin_left = 70
    margin_right = 28
    margin_top = 30
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    train_values = [metric.train_total_loss for metric in metrics]
    val_values = [metric.val_total_loss for metric in metrics if metric.val_total_loss is not None]
    all_values = train_values + val_values
    min_loss = min(all_values)
    max_loss = max(all_values)
    if min_loss == max_loss:
        min_loss -= 0.5
        max_loss += 0.5
    loss_padding = (max_loss - min_loss) * 0.08
    min_loss -= loss_padding
    max_loss += loss_padding

    def x_for(index: int) -> float:
        if len(metrics) == 1:
            return margin_left + plot_width / 2
        return margin_left + (index / (len(metrics) - 1)) * plot_width

    def y_for(loss: float) -> float:
        fraction = (loss - min_loss) / (max_loss - min_loss)
        return margin_top + (1.0 - fraction) * plot_height

    def points(values: list[float]) -> str:
        return " ".join(f"{x_for(index):.1f},{y_for(value):.1f}" for index, value in enumerate(values))

    y_ticks = 5
    grid_lines = []
    for tick in range(y_ticks + 1):
        fraction = tick / y_ticks
        loss = max_loss - fraction * (max_loss - min_loss)
        y = margin_top + fraction * plot_height
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" '
            f'y2="{y:.1f}" stroke="#e5e7eb" />'
        )
        grid_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#374151">{loss:.2f}</text>'
        )

    x_labels = []
    for index, metric in enumerate(metrics):
        if index == 0 or index == len(metrics) - 1 or metric.epoch % 5 == 0:
            x = x_for(index)
            x_labels.append(
                f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" '
                f'font-size="12" fill="#374151">{metric.epoch}</text>'
            )

    val_polyline = ""
    if val_values:
        val_polyline = (
            f'<polyline points="{points(val_values)}" fill="none" stroke="#dc2626" '
            f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{width / 2:.1f}" y="22" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#111827">Supervised Tiny Loss Curve</text>
  <g font-family="Arial, sans-serif">
    {"".join(grid_lines)}
    <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" />
    <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" />
    {"".join(x_labels)}
    <text x="{width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="13" fill="#111827">Epoch</text>
    <text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-size="13" fill="#111827">Total loss</text>
    <polyline points="{points(train_values)}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
    {val_polyline}
    <rect x="{width - 205}" y="42" width="160" height="52" fill="#ffffff" stroke="#d1d5db" />
    <line x1="{width - 190}" y1="62" x2="{width - 158}" y2="62" stroke="#2563eb" stroke-width="3" />
    <text x="{width - 148}" y="66" font-size="12" fill="#111827">train total</text>
    <line x1="{width - 190}" y1="82" x2="{width - 158}" y2="82" stroke="#dc2626" stroke-width="3" />
    <text x="{width - 148}" y="86" font-size="12" fill="#111827">val total</text>
  </g>
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")


def run_training(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    device = resolve_device(config.device)
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "config.yaml")

    train_dataset = limited_dataset(
        make_dataset(config.train_path, config.train_cache_dir),
        config.max_train_samples,
    )
    val_dataset = (
        limited_dataset(
            make_dataset(config.val_path, config.val_cache_dir),
            config.max_val_samples,
        )
        if config.val_path
        else None
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        make_loader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_dataset is not None
        else None
    )

    model_config = config.model or ResNetConfig()
    model = PolicyValueResNet(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    status_path = output_dir / "status.json"
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    train_sample_count = len(cast(Sized, train_dataset))
    val_sample_count = len(cast(Sized, val_dataset)) if val_dataset is not None else 0
    status: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "config_path": str(config_path),
        "metrics_path": str(output_dir / "metrics.jsonl"),
        "checkpoint_path": str(output_dir / "checkpoint.pt"),
        "loss_plot_path": str(output_dir / "loss.svg"),
        "dataset_manifest_path": config.dataset_manifest_path,
        "train_cache_dir": config.train_cache_dir,
        "val_cache_dir": config.val_cache_dir,
        "seed": config.seed,
        "device": str(device),
        "pin_memory": device.type == "cuda",
        "train_samples": train_sample_count,
        "val_samples": val_sample_count,
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"training on {train_sample_count} samples"
        + (f", validating on {val_sample_count} samples" if val_dataset is not None else "")
    )
    print(f"device={device} epochs={config.epochs} batch_size={config.batch_size}")

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    run_start = time.perf_counter()
    epoch_metrics: list[EpochMetrics] = []

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()
            train_total, train_policy, train_value = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                config.value_weight,
                epoch=epoch,
                log_every_steps=config.log_every_steps,
            )
            val_total: float | None = None
            val_policy: float | None = None
            val_value: float | None = None
            if val_loader is not None:
                val_total, val_policy, val_value = evaluate(
                    model,
                    val_loader,
                    device,
                    config.value_weight,
                )

            metrics = EpochMetrics(
                epoch=epoch,
                train_total_loss=train_total,
                train_policy_loss=train_policy,
                train_value_loss=train_value,
                val_total_loss=val_total,
                val_policy_loss=val_policy,
                val_value_loss=val_value,
                elapsed_seconds=time.perf_counter() - epoch_start,
            )
            epoch_metrics.append(metrics)
            write_jsonl(metrics_path, asdict(metrics))

            val_text = (
                f" val_total={val_total:.4f} val_policy={val_policy:.4f} val_value={val_value:.4f}"
                if val_total is not None and val_policy is not None and val_value is not None
                else ""
            )
            print(
                f"epoch {epoch:03d}/{config.epochs:03d} "
                f"train_total={train_total:.4f} "
                f"train_policy={train_policy:.4f} train_value={train_value:.4f}"
                f"{val_text} elapsed={metrics.elapsed_seconds:.1f}s",
                flush=True,
            )

        write_loss_svg(epoch_metrics, output_dir / "loss.svg")
        completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": json_ready_config(config, device),
            "completed_at": completed_at,
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")
        status["status"] = "completed"
        status["completed_at"] = completed_at
        status["elapsed_seconds"] = time.perf_counter() - run_start
        status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved checkpoint to {output_dir / 'checkpoint.pt'}")
        print(f"saved loss plot to {output_dir / 'loss.svg'}")
        return output_dir
    except Exception:
        status["status"] = "failed"
        status["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        status["elapsed_seconds"] = time.perf_counter() - run_start
        status_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the supervised McChess baseline.")
    parser.add_argument("config", type=Path, help="YAML training config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_training(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
