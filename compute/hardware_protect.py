"""Unified CPU and GPU protection for local McChess workloads."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from typing import Literal

from compute import cpu_protect as cpu
from compute import gpu_protect as gpu


STATE_VERSION = 1
DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / ".local" / "hardware_protect.json"
Component = Literal["all", "cpu", "gpu"]
GpuLimitMode = Literal["power", "clock"]


@dataclass(frozen=True)
class GpuProtectionConfig:
    gpu: int
    limit_mode: GpuLimitMode
    percent: float
    clock_mhz: int | None


@dataclass(frozen=True)
class HardwareProtectionState:
    cpu_restore: cpu.CpuPowerState | None
    cpu_max_frequency_mhz: int | None
    gpu_config: GpuProtectionConfig | None


@dataclass(frozen=True)
class HardwareProtectionChange:
    cpu_state: cpu.CpuPowerState | None
    gpu_change: gpu.LimitChange | None
    enabled: bool
    cpu_scheme_is_active: bool = True


class HardwareProtectError(RuntimeError):
    """Raised when unified hardware protection cannot be completed safely."""


def _uses_cpu(component: Component) -> bool:
    return component in {"all", "cpu"}


def _uses_gpu(component: Component) -> bool:
    return component in {"all", "gpu"}


def _enable_gpu(
    config: GpuProtectionConfig,
    *,
    runner: gpu.Runner,
    dry_run: bool,
) -> gpu.LimitChange:
    if config.limit_mode == "clock":
        return gpu.enable_clock_limit(
            gpu=config.gpu,
            percent=config.percent,
            gpu_clock_mhz=config.clock_mhz,
            runner=runner,
            dry_run=dry_run,
        )
    return gpu.enable_limit(
        gpu=config.gpu,
        percent=config.percent,
        runner=runner,
        dry_run=dry_run,
    )


def _disable_gpu(
    config: GpuProtectionConfig,
    *,
    runner: gpu.Runner,
    dry_run: bool,
) -> gpu.LimitChange:
    if config.limit_mode == "clock":
        return gpu.disable_clock_limit(
            gpu=config.gpu,
            runner=runner,
            dry_run=dry_run,
        )
    return gpu.disable_limit(
        gpu=config.gpu,
        runner=runner,
        dry_run=dry_run,
    )


def _gpu_config_to_payload(config: GpuProtectionConfig) -> dict[str, object]:
    return {
        "gpu": config.gpu,
        "limit_mode": config.limit_mode,
        "percent": config.percent,
        "clock_mhz": config.clock_mhz,
    }


def _gpu_config_from_payload(payload: object) -> GpuProtectionConfig:
    if not isinstance(payload, dict):
        raise HardwareProtectError("saved hardware state has invalid gpu configuration")
    gpu_index = payload.get("gpu")
    limit_mode = payload.get("limit_mode")
    percent = payload.get("percent")
    clock_mhz = payload.get("clock_mhz")
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        raise HardwareProtectError("saved hardware state has an invalid GPU index")
    if limit_mode not in {"power", "clock"}:
        raise HardwareProtectError("saved hardware state has an invalid GPU limit mode")
    if (
        not isinstance(percent, (int, float))
        or isinstance(percent, bool)
        or not 0.0 < float(percent) <= 100.0
    ):
        raise HardwareProtectError("saved hardware state has an invalid GPU percent")
    if clock_mhz is not None and (
        not isinstance(clock_mhz, int) or isinstance(clock_mhz, bool) or clock_mhz <= 0
    ):
        raise HardwareProtectError("saved hardware state has an invalid GPU clock")
    if limit_mode == "power" and clock_mhz is not None:
        raise HardwareProtectError("saved power-limit state cannot contain a GPU clock")
    return GpuProtectionConfig(
        gpu=gpu_index,
        limit_mode=limit_mode,
        percent=float(percent),
        clock_mhz=clock_mhz,
    )


def hardware_state_to_payload(state: HardwareProtectionState) -> dict[str, object]:
    cpu_payload: dict[str, object] | None = None
    if state.cpu_restore is not None:
        cpu_payload = {
            "restore": cpu.cpu_state_to_payload(state.cpu_restore),
            "max_frequency_mhz": state.cpu_max_frequency_mhz,
        }
    return {
        "version": STATE_VERSION,
        "cpu": cpu_payload,
        "gpu": (
            _gpu_config_to_payload(state.gpu_config)
            if state.gpu_config is not None
            else None
        ),
    }


def hardware_state_from_payload(payload: object) -> HardwareProtectionState:
    if not isinstance(payload, dict):
        raise HardwareProtectError("saved hardware protection state must be a JSON object")
    if payload.get("version") != STATE_VERSION:
        raise HardwareProtectError("saved hardware protection state has an unsupported version")

    cpu_payload = payload.get("cpu")
    if cpu_payload is None:
        cpu_restore = None
        cpu_max_frequency_mhz = None
    else:
        if not isinstance(cpu_payload, dict):
            raise HardwareProtectError("saved hardware state has invalid CPU configuration")
        cpu_max_frequency_mhz = cpu_payload.get("max_frequency_mhz")
        if (
            not isinstance(cpu_max_frequency_mhz, int)
            or isinstance(cpu_max_frequency_mhz, bool)
        ):
            raise HardwareProtectError("saved hardware state has an invalid CPU frequency")
        try:
            cpu_restore = cpu.cpu_state_from_payload(cpu_payload.get("restore"))
            cpu.protected_cpu_power_state(cpu_restore, cpu_max_frequency_mhz)
        except cpu.CpuProtectError as exc:
            raise HardwareProtectError(str(exc)) from exc

    gpu_payload = payload.get("gpu")
    gpu_config = None if gpu_payload is None else _gpu_config_from_payload(gpu_payload)
    if cpu_restore is None and gpu_config is None:
        raise HardwareProtectError("saved hardware state does not contain CPU or GPU protection")
    return HardwareProtectionState(
        cpu_restore=cpu_restore,
        cpu_max_frequency_mhz=cpu_max_frequency_mhz,
        gpu_config=gpu_config,
    )


def save_hardware_state(path: Path, state: HardwareProtectionState) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(hardware_state_to_payload(state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError as exc:
        raise HardwareProtectError(f"could not save restore state to {path}: {exc}") from exc


def load_hardware_state(path: Path = DEFAULT_STATE_PATH) -> HardwareProtectionState:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HardwareProtectError(
            f"no saved hardware restore state at {path}; refusing to guess prior settings"
        ) from exc
    except OSError as exc:
        raise HardwareProtectError(f"could not read restore state from {path}: {exc}") from exc
    try:
        return hardware_state_from_payload(json.loads(text))
    except json.JSONDecodeError as exc:
        raise HardwareProtectError(f"saved restore state at {path} is not valid JSON") from exc


def remove_hardware_state(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:
        raise HardwareProtectError(
            f"hardware settings were restored but state file {path} could not be removed: {exc}"
        ) from exc


def enable_protection(
    *,
    component: Component = "all",
    cpu_max_frequency_mhz: int = cpu.DEFAULT_MAX_FREQUENCY_MHZ,
    gpu_index: int = 0,
    gpu_percent: float = gpu.DEFAULT_PERCENT,
    gpu_limit_mode: GpuLimitMode = "power",
    gpu_clock_mhz: int | None = None,
    cpu_runner: cpu.Runner = cpu.run_command,
    gpu_runner: gpu.Runner = gpu.run_command,
    state_path: Path = DEFAULT_STATE_PATH,
    dry_run: bool = False,
) -> HardwareProtectionChange:
    if component not in {"all", "cpu", "gpu"}:
        raise HardwareProtectError(f"unsupported hardware component: {component!r}")
    if state_path.exists():
        raise HardwareProtectError(
            f"hardware protection already has restore state at {state_path}; run --off first"
        )
    if _uses_gpu(component) and gpu_limit_mode == "power" and gpu_clock_mhz is not None:
        raise HardwareProtectError("--gpu-clock-mhz requires --gpu-limit-mode clock")

    cpu_restore = cpu.query_cpu_power_state(cpu_runner) if _uses_cpu(component) else None
    cpu_target = (
        cpu.protected_cpu_power_state(cpu_restore, cpu_max_frequency_mhz)
        if cpu_restore is not None
        else None
    )
    if _uses_gpu(component):
        gpu_config = _gpu_config_from_payload(
            _gpu_config_to_payload(
                GpuProtectionConfig(
                    gpu=gpu_index,
                    limit_mode=gpu_limit_mode,
                    percent=gpu_percent,
                    clock_mhz=gpu_clock_mhz,
                )
            )
        )
    else:
        gpu_config = None

    try:
        gpu_change = (
            _enable_gpu(gpu_config, runner=gpu_runner, dry_run=True)
            if gpu_config is not None
            else None
        )
    except gpu.GpuProtectError as exc:
        raise HardwareProtectError(str(exc)) from exc

    restore_state = HardwareProtectionState(
        cpu_restore=cpu_restore,
        cpu_max_frequency_mhz=(cpu_max_frequency_mhz if cpu_restore is not None else None),
        gpu_config=gpu_config,
    )
    if dry_run:
        return HardwareProtectionChange(
            cpu_state=cpu_target,
            gpu_change=gpu_change,
            enabled=True,
        )

    save_hardware_state(state_path, restore_state)
    try:
        if gpu_config is not None:
            gpu_change = _enable_gpu(gpu_config, runner=gpu_runner, dry_run=False)
        if cpu_target is not None:
            cpu.apply_cpu_power_state(cpu_target, activate=True, runner=cpu_runner)
            cpu.verify_cpu_power_state(cpu_target, cpu_runner)
    except (cpu.CpuProtectError, gpu.GpuProtectError) as exc:
        raise HardwareProtectError(
            f"hardware protection was only partially applied: {exc}. "
            f"Restore state remains at {state_path}; run hardware_protect --off"
        ) from exc
    return HardwareProtectionChange(
        cpu_state=cpu_target,
        gpu_change=gpu_change,
        enabled=True,
    )


def disable_protection(
    *,
    cpu_runner: cpu.Runner = cpu.run_command,
    gpu_runner: gpu.Runner = gpu.run_command,
    state_path: Path = DEFAULT_STATE_PATH,
    dry_run: bool = False,
) -> HardwareProtectionChange:
    saved = load_hardware_state(state_path)
    errors: list[str] = []
    gpu_change: gpu.LimitChange | None = None
    cpu_scheme_is_active = True

    if saved.gpu_config is not None:
        try:
            gpu_change = _disable_gpu(
                saved.gpu_config,
                runner=gpu_runner,
                dry_run=dry_run,
            )
        except gpu.GpuProtectError as exc:
            errors.append(f"GPU restore failed: {exc}")

    if saved.cpu_restore is not None:
        try:
            active_scheme_guid, _ = cpu.query_active_scheme(cpu_runner)
            cpu_scheme_is_active = active_scheme_guid == saved.cpu_restore.scheme_guid
            if not dry_run:
                cpu.apply_cpu_power_state(
                    saved.cpu_restore,
                    activate=cpu_scheme_is_active,
                    runner=cpu_runner,
                )
                cpu.verify_cpu_power_state(saved.cpu_restore, cpu_runner)
        except cpu.CpuProtectError as exc:
            errors.append(f"CPU restore failed: {exc}")

    if errors:
        raise HardwareProtectError(
            "; ".join(errors) + f". Restore state remains at {state_path}"
        )
    if not dry_run:
        remove_hardware_state(state_path)
    return HardwareProtectionChange(
        cpu_state=saved.cpu_restore,
        gpu_change=gpu_change,
        enabled=False,
        cpu_scheme_is_active=cpu_scheme_is_active,
    )


def format_change(change: HardwareProtectionChange, dry_run: bool) -> str:
    if change.enabled:
        heading = "Hardware protection dry run:" if dry_run else "Hardware protection enabled:"
    else:
        heading = "Hardware restore dry run:" if dry_run else "Hardware protection disabled:"
    lines = [heading]
    if change.cpu_state is not None:
        state = change.cpu_state
        if change.enabled:
            verb = "would cap" if dry_run else "capped"
            lines.append(
                f"CPU: {verb} AC/DC at {state.max_frequency_mhz.ac} MHz with boost disabled"
            )
        else:
            verb = "would restore" if dry_run else "restored"
            lines.append(
                f"CPU: {verb} AC={state.max_frequency_mhz.ac} MHz, "
                f"DC={state.max_frequency_mhz.dc} MHz, "
                f"boost={state.boost_mode.ac}/{state.boost_mode.dc}"
            )
            if not change.cpu_scheme_is_active:
                lines.append("CPU: restored the saved plan without making it active")
    if change.gpu_change is not None:
        lines.append(f"GPU: {gpu.format_change(change.gpu_change, dry_run)}")
    return "\n".join(lines)


def format_status(
    *,
    component: Component = "all",
    gpu_index: int = 0,
    cpu_runner: cpu.Runner = cpu.run_command,
    gpu_runner: gpu.Runner = gpu.run_command,
    state_path: Path = DEFAULT_STATE_PATH,
) -> str:
    lines = [
        f"Hardware restore state: {state_path if state_path.exists() else 'none'}",
    ]
    if _uses_cpu(component):
        lines.append("CPU:")
        lines.append(indent(cpu.format_cpu_status(cpu.query_cpu_power_state(cpu_runner)), "  "))
    if _uses_gpu(component):
        try:
            gpu_status = gpu.query_status(gpu_index, gpu_runner).strip()
        except gpu.GpuProtectError as exc:
            raise HardwareProtectError(str(exc)) from exc
        lines.append("GPU:")
        lines.append(indent(gpu_status, "  "))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply and restore McChess CPU/GPU hardware limits together."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--on", action="store_true", help="Apply selected hardware limits.")
    mode.add_argument("--off", action="store_true", help="Restore all saved hardware settings.")
    mode.add_argument("--status", action="store_true", help="Print current hardware settings.")
    parser.add_argument(
        "--component",
        choices=("all", "cpu", "gpu"),
        default="all",
        help="Protect or inspect all hardware, CPU only, or GPU only. Default: all.",
    )
    parser.add_argument(
        "--cpu-max-frequency-mhz",
        type=int,
        default=cpu.DEFAULT_MAX_FREQUENCY_MHZ,
        help=f"Maximum AC/DC CPU frequency. Default: {cpu.DEFAULT_MAX_FREQUENCY_MHZ}.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="NVIDIA GPU index. Default: 0.")
    parser.add_argument(
        "--gpu-percent",
        type=float,
        default=gpu.DEFAULT_PERCENT,
        help="GPU power/default-clock percentage. Default: 75.",
    )
    parser.add_argument(
        "--gpu-limit-mode",
        choices=("power", "clock"),
        default="power",
        help="Use an NVIDIA watt or graphics-clock limit. Default: power.",
    )
    parser.add_argument(
        "--gpu-clock-mhz",
        type=int,
        default=None,
        help="Explicit GPU graphics clock for clock mode.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.off and args.component != "all":
        print("error: --off restores every saved component; omit --component", file=sys.stderr)
        return 1
    try:
        if args.status:
            print(format_status(component=args.component, gpu_index=args.gpu))
            return 0
        if args.on:
            change = enable_protection(
                component=args.component,
                cpu_max_frequency_mhz=args.cpu_max_frequency_mhz,
                gpu_index=args.gpu,
                gpu_percent=args.gpu_percent,
                gpu_limit_mode=args.gpu_limit_mode,
                gpu_clock_mhz=args.gpu_clock_mhz,
                dry_run=args.dry_run,
            )
        else:
            change = disable_protection(dry_run=args.dry_run)
    except (HardwareProtectError, cpu.CpuProtectError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_change(change, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
