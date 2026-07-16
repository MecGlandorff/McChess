"""Program: McChess GPU Protect
Author: Mec Glandorff
Version: 0.1-proto

Description:
    Small NVIDIA GPU power and clock-limit helper for local McChess training
    runs.

    Provides:
      - protected power-limit toggle via nvidia-smi
      - protected graphics clock-limit toggle via nvidia-smi
      - default restore path after training
      - status inspection for current/default/draw power
      - dry-run mode for checking planned changes safely

Usage:
    poetry run python gpu_protect --on
    poetry run python gpu_protect --off
    poetry run python gpu_protect --status

Notes:
    - This supports NVIDIA GPUs only.
    - Power limits are set in watts, not percent.
    - --on computes watts from the default power limit.
    - --off restores the default power limit reported by nvidia-smi.
    - --limit-mode clock locks the graphics clock instead of setting watts.
    - Windows may require Administrator PowerShell for -pl changes.
    - This does not make training use CUDA; training must print device=cuda.
    - Keep failure modes obvious and never hide nvidia-smi errors.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass


DEFAULT_PERCENT = 75.0
_POWER_VALUE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str]], CommandResult]


@dataclass(frozen=True)
class PowerLimitChange:
    gpu: int
    default_watts: float
    target_watts: int
    percent: float
    enabled: bool


@dataclass(frozen=True)
class ClockLimitChange:
    gpu: int
    target_clock_mhz: int
    reference_clock_mhz: int | None
    percent: float | None
    enabled: bool


LimitChange = PowerLimitChange | ClockLimitChange


class GpuProtectError(RuntimeError):
    """Raised when the GPU power-limit command cannot be completed."""


def run_command(command: Sequence[str]) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError as exc:
        raise GpuProtectError(
            "nvidia-smi was not found on PATH. This helper supports NVIDIA GPUs only."
        ) from exc

    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_power_watts(output: str) -> float:
    match = _POWER_VALUE_RE.search(output)
    if match is None:
        raise GpuProtectError(f"could not parse a watt value from nvidia-smi output: {output!r}")
    return float(match.group(0))


def parse_clock_mhz_values(output: str) -> list[int]:
    values = sorted({int(float(match.group(0))) for match in _POWER_VALUE_RE.finditer(output)})
    if not values:
        raise GpuProtectError(f"could not parse clock values from nvidia-smi output: {output!r}")
    return values


def checked_nvidia_smi(args: Sequence[str], runner: Runner) -> str:
    result = runner(["nvidia-smi", *args])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise GpuProtectError(f"nvidia-smi failed: {detail}")
    return result.stdout


def query_default_power_limit(gpu: int, runner: Runner = run_command) -> float:
    output = checked_nvidia_smi(
        [
            "-i",
            str(gpu),
            "--query-gpu=power.default_limit",
            "--format=csv,noheader,nounits",
        ],
        runner,
    )
    return parse_power_watts(output)


def query_status(gpu: int, runner: Runner = run_command) -> str:
    return checked_nvidia_smi(
        [
            "-i",
            str(gpu),
            (
                "--query-gpu=index,name,temperature.gpu,power.limit,"
                "power.default_limit,power.draw,clocks.current.graphics"
            ),
            "--format=csv",
        ],
        runner,
    )


def query_supported_graphics_clocks(gpu: int, runner: Runner = run_command) -> list[int]:
    output = checked_nvidia_smi(
        [
            "-i",
            str(gpu),
            "--query-supported-clocks=gr",
            "--format=csv,noheader,nounits",
        ],
        runner,
    )
    return parse_clock_mhz_values(output)


def set_power_limit(gpu: int, watts: int, runner: Runner = run_command) -> None:
    checked_nvidia_smi(["-i", str(gpu), "-pl", str(watts)], runner)


def set_graphics_clock_limit(gpu: int, clock_mhz: int, runner: Runner = run_command) -> None:
    checked_nvidia_smi(["-i", str(gpu), "-lgc", str(clock_mhz)], runner)


def reset_graphics_clock_limit(gpu: int, runner: Runner = run_command) -> None:
    checked_nvidia_smi(["-i", str(gpu), "-rgc"], runner)


def protected_watts(default_watts: float, percent: float) -> int:
    if percent <= 0.0 or percent > 100.0:
        raise GpuProtectError("--percent must be greater than 0 and at most 100")
    return max(1, math.floor(default_watts * percent / 100.0))


def protected_clock_mhz(supported_clocks: Sequence[int], percent: float) -> int:
    if percent <= 0.0 or percent > 100.0:
        raise GpuProtectError("--percent must be greater than 0 and at most 100")
    if not supported_clocks:
        raise GpuProtectError("no supported graphics clocks were reported by nvidia-smi")

    unique_clocks = sorted(set(supported_clocks))
    max_clock = unique_clocks[-1]
    target_clock = math.floor(max_clock * percent / 100.0)
    for clock in reversed(unique_clocks):
        if clock <= target_clock:
            return clock
    return unique_clocks[0]


def enable_limit(
    *,
    gpu: int = 0,
    percent: float = DEFAULT_PERCENT,
    runner: Runner = run_command,
    dry_run: bool = False,
) -> PowerLimitChange:
    default_watts = query_default_power_limit(gpu, runner)
    target_watts = protected_watts(default_watts, percent)
    if not dry_run:
        set_power_limit(gpu, target_watts, runner)
    return PowerLimitChange(
        gpu=gpu,
        default_watts=default_watts,
        target_watts=target_watts,
        percent=percent,
        enabled=True,
    )


def enable_clock_limit(
    *,
    gpu: int = 0,
    percent: float = DEFAULT_PERCENT,
    gpu_clock_mhz: int | None = None,
    runner: Runner = run_command,
    dry_run: bool = False,
) -> ClockLimitChange:
    reference_clock_mhz = None
    if gpu_clock_mhz is None:
        supported_clocks = query_supported_graphics_clocks(gpu, runner)
        reference_clock_mhz = max(supported_clocks)
        target_clock_mhz = protected_clock_mhz(supported_clocks, percent)
        effective_percent = percent
    else:
        if gpu_clock_mhz <= 0:
            raise GpuProtectError("--gpu-clock-mhz must be greater than 0")
        target_clock_mhz = gpu_clock_mhz
        effective_percent = None

    if not dry_run:
        set_graphics_clock_limit(gpu, target_clock_mhz, runner)
    return ClockLimitChange(
        gpu=gpu,
        target_clock_mhz=target_clock_mhz,
        reference_clock_mhz=reference_clock_mhz,
        percent=effective_percent,
        enabled=True,
    )


def disable_limit(
    *,
    gpu: int = 0,
    runner: Runner = run_command,
    dry_run: bool = False,
) -> PowerLimitChange:
    default_watts = query_default_power_limit(gpu, runner)
    target_watts = math.floor(default_watts)
    if not dry_run:
        set_power_limit(gpu, target_watts, runner)
    return PowerLimitChange(
        gpu=gpu,
        default_watts=default_watts,
        target_watts=target_watts,
        percent=100.0,
        enabled=False,
    )


def disable_clock_limit(
    *,
    gpu: int = 0,
    runner: Runner = run_command,
    dry_run: bool = False,
) -> ClockLimitChange:
    if not dry_run:
        reset_graphics_clock_limit(gpu, runner)
    return ClockLimitChange(
        gpu=gpu,
        target_clock_mhz=0,
        reference_clock_mhz=None,
        percent=None,
        enabled=False,
    )


def format_change(change: LimitChange, dry_run: bool) -> str:
    action = "would set" if dry_run else "set"
    mode = "protected" if change.enabled else "default"
    if isinstance(change, ClockLimitChange):
        if change.enabled:
            details = []
            if change.reference_clock_mhz is not None:
                details.append(f"max supported {change.reference_clock_mhz} MHz")
            if change.percent is not None:
                details.append(f"{change.percent:.1f}% of max supported")
            suffix = "" if not details else f" ({', '.join(details)})"
            return (
                f"GPU {change.gpu}: {action} protected graphics clock limit to "
                f"{change.target_clock_mhz} MHz{suffix}."
            )
        reset_action = "would reset" if dry_run else "reset"
        return f"GPU {change.gpu}: {reset_action} graphics clock limit to default."

    return (
        f"GPU {change.gpu}: {action} {mode} power limit to "
        f"{change.target_watts} W "
        f"(default {change.default_watts:.1f} W, {change.percent:.1f}%)."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toggle an NVIDIA GPU power limit for local McChess runs."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--on", action="store_true", help="Set the protected power limit.")
    mode.add_argument("--off", action="store_true", help="Restore the default power limit.")
    mode.add_argument("--status", action="store_true", help="Print current NVIDIA power status.")
    parser.add_argument("--gpu", type=int, default=0, help="NVIDIA GPU index. Default: 0.")
    parser.add_argument(
        "--percent",
        type=float,
        default=DEFAULT_PERCENT,
        help=(
            "Protected limit as a percent of default power, or max supported "
            "graphics clock in clock mode. Default: 75."
        ),
    )
    parser.add_argument(
        "--limit-mode",
        choices=("power", "clock"),
        default="power",
        help="Use a watt power limit or a graphics clock limit. Default: power.",
    )
    parser.add_argument(
        "--gpu-clock-mhz",
        type=int,
        default=None,
        help="Explicit graphics clock limit for --limit-mode clock.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned change only.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.status:
            print(query_status(args.gpu), end="")
            return 0
        change: LimitChange
        if args.on:
            if args.limit_mode == "clock":
                change = enable_clock_limit(
                    gpu=args.gpu,
                    percent=args.percent,
                    gpu_clock_mhz=args.gpu_clock_mhz,
                    dry_run=args.dry_run,
                )
            else:
                change = enable_limit(
                    gpu=args.gpu,
                    percent=args.percent,
                    dry_run=args.dry_run,
                )
        elif args.limit_mode == "clock":
            change = disable_clock_limit(gpu=args.gpu, dry_run=args.dry_run)
        else:
            change = disable_limit(gpu=args.gpu, dry_run=args.dry_run)
    except GpuProtectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_change(change, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
