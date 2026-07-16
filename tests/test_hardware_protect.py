from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from compute import cpu_protect as cpu
from compute import gpu_protect as gpu
from compute.hardware_protect import (
    GpuProtectionConfig,
    HardwareProtectError,
    HardwareProtectionState,
    disable_protection,
    enable_protection,
    format_change,
    format_status,
    hardware_state_from_payload,
    hardware_state_to_payload,
    load_hardware_state,
)


SCHEME_GUID = "64a64f24-65b9-4b56-befd-5ec1eaced9b3"


class FakeCpuRunner:
    def __init__(self) -> None:
        self.active_scheme_guid = SCHEME_GUID
        self.settings = {
            cpu.MAX_FREQUENCY_GUID: cpu.PowerSettingValues(ac=0, dc=4000),
            cpu.BOOST_MODE_GUID: cpu.PowerSettingValues(ac=2, dc=2),
        }
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> cpu.CommandResult:
        saved = list(command)
        self.commands.append(saved)
        action = saved[1].lower()
        if action == "/getactivescheme":
            return cpu.CommandResult(
                0,
                f"Power Scheme GUID: {self.active_scheme_guid}  (Silent)\n",
                "",
            )
        if action == "/qh":
            values = self.settings[saved[-1].lower()]
            return cpu.CommandResult(
                0,
                (
                    f"Current AC Power Setting Index: 0x{values.ac:08x}\n"
                    f"Current DC Power Setting Index: 0x{values.dc:08x}\n"
                ),
                "",
            )
        if action in {"/setacvalueindex", "/setdcvalueindex"}:
            setting_guid = saved[4].lower()
            value = int(saved[5])
            current = self.settings[setting_guid]
            if action == "/setacvalueindex":
                self.settings[setting_guid] = cpu.PowerSettingValues(value, current.dc)
            else:
                self.settings[setting_guid] = cpu.PowerSettingValues(current.ac, value)
            return cpu.CommandResult(0, "", "")
        if action == "/setactive":
            self.active_scheme_guid = saved[2].lower()
            return cpu.CommandResult(0, "", "")
        raise AssertionError(f"unexpected CPU command: {saved}")


class FakeGpuRunner:
    def __init__(self) -> None:
        self.default_watts = 100.0
        self.power_limit_watts = 100
        self.clock_limit_mhz: int | None = None
        self.fail_clock_set = False
        self.fail_clock_reset = False
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> gpu.CommandResult:
        saved = list(command)
        self.commands.append(saved)
        if "--query-gpu=power.default_limit" in saved:
            return gpu.CommandResult(0, f"{self.default_watts:.2f}\n", "")
        if "--query-supported-clocks=gr" in saved:
            return gpu.CommandResult(0, "615\n930\n1500\n3105\n", "")
        if any(value.startswith("--query-gpu=index,name,temperature.gpu") for value in saved):
            return gpu.CommandResult(
                0,
                "index, name, temperature.gpu\n0, Fake GPU, 55\n",
                "",
            )
        if "-pl" in saved:
            self.power_limit_watts = int(saved[-1])
            return gpu.CommandResult(0, "", "")
        if "-lgc" in saved:
            if self.fail_clock_set:
                return gpu.CommandResult(1, "", "clock permission denied")
            self.clock_limit_mhz = int(saved[-1])
            return gpu.CommandResult(0, "", "")
        if "-rgc" in saved:
            if self.fail_clock_reset:
                return gpu.CommandResult(1, "", "clock reset permission denied")
            self.clock_limit_mhz = None
            return gpu.CommandResult(0, "", "")
        raise AssertionError(f"unexpected GPU command: {saved}")


def original_cpu_state() -> cpu.CpuPowerState:
    return cpu.CpuPowerState(
        scheme_guid=SCHEME_GUID,
        scheme_name="Silent",
        max_frequency_mhz=cpu.PowerSettingValues(ac=0, dc=4000),
        boost_mode=cpu.PowerSettingValues(ac=2, dc=2),
    )


def test_hardware_state_round_trip() -> None:
    state = HardwareProtectionState(
        cpu_restore=original_cpu_state(),
        cpu_max_frequency_mhz=3200,
        gpu_config=GpuProtectionConfig(
            gpu=0,
            limit_mode="clock",
            percent=75.0,
            clock_mhz=1500,
        ),
    )

    assert hardware_state_from_payload(hardware_state_to_payload(state)) == state


def test_enable_and_disable_all_components_with_one_restore_state(tmp_path: Path) -> None:
    cpu_runner = FakeCpuRunner()
    gpu_runner = FakeGpuRunner()
    state_path = tmp_path / ".local" / "hardware_protect.json"

    enabled = enable_protection(
        component="all",
        cpu_max_frequency_mhz=3200,
        gpu_limit_mode="clock",
        gpu_clock_mhz=1500,
        cpu_runner=cpu_runner,
        gpu_runner=gpu_runner,
        state_path=state_path,
    )

    saved = load_hardware_state(state_path)
    assert saved.cpu_restore == original_cpu_state()
    assert saved.gpu_config is not None
    assert saved.gpu_config.limit_mode == "clock"
    assert cpu_runner.settings[cpu.MAX_FREQUENCY_GUID] == cpu.PowerSettingValues(3200, 3200)
    assert cpu_runner.settings[cpu.BOOST_MODE_GUID] == cpu.PowerSettingValues(0, 0)
    assert gpu_runner.clock_limit_mhz == 1500
    assert "Hardware protection enabled" in format_change(enabled, dry_run=False)

    disabled = disable_protection(
        cpu_runner=cpu_runner,
        gpu_runner=gpu_runner,
        state_path=state_path,
    )

    assert cpu_runner.settings[cpu.MAX_FREQUENCY_GUID] == cpu.PowerSettingValues(0, 4000)
    assert cpu_runner.settings[cpu.BOOST_MODE_GUID] == cpu.PowerSettingValues(2, 2)
    assert gpu_runner.clock_limit_mhz is None
    assert not state_path.exists()
    assert "Hardware protection disabled" in format_change(disabled, dry_run=False)


def test_enable_dry_run_does_not_mutate_or_save(tmp_path: Path) -> None:
    cpu_runner = FakeCpuRunner()
    gpu_runner = FakeGpuRunner()
    state_path = tmp_path / "hardware_protect.json"

    change = enable_protection(
        component="all",
        gpu_limit_mode="clock",
        gpu_clock_mhz=1500,
        cpu_runner=cpu_runner,
        gpu_runner=gpu_runner,
        state_path=state_path,
        dry_run=True,
    )

    assert not state_path.exists()
    assert cpu_runner.settings[cpu.MAX_FREQUENCY_GUID] == cpu.PowerSettingValues(0, 4000)
    assert gpu_runner.clock_limit_mhz is None
    assert not any(command[1].startswith("/set") for command in cpu_runner.commands)
    assert "Hardware protection dry run" in format_change(change, dry_run=True)


def test_cpu_only_does_not_call_nvidia_smi(tmp_path: Path) -> None:
    cpu_runner = FakeCpuRunner()
    gpu_runner = FakeGpuRunner()
    state_path = tmp_path / "hardware_protect.json"

    enable_protection(
        component="cpu",
        cpu_runner=cpu_runner,
        gpu_runner=gpu_runner,
        state_path=state_path,
    )

    assert gpu_runner.commands == []
    assert load_hardware_state(state_path).gpu_config is None


def test_gpu_failure_leaves_restore_state_for_cpu_rollback(tmp_path: Path) -> None:
    cpu_runner = FakeCpuRunner()
    gpu_runner = FakeGpuRunner()
    gpu_runner.fail_clock_set = True
    state_path = tmp_path / "hardware_protect.json"

    with pytest.raises(HardwareProtectError, match="partially applied"):
        enable_protection(
            component="all",
            gpu_limit_mode="clock",
            gpu_clock_mhz=1500,
            cpu_runner=cpu_runner,
            gpu_runner=gpu_runner,
            state_path=state_path,
        )

    assert state_path.exists()
    assert cpu_runner.settings[cpu.MAX_FREQUENCY_GUID] == cpu.PowerSettingValues(0, 4000)

    gpu_runner.fail_clock_set = False
    disable_protection(
        cpu_runner=cpu_runner,
        gpu_runner=gpu_runner,
        state_path=state_path,
    )
    assert cpu_runner.settings[cpu.MAX_FREQUENCY_GUID] == cpu.PowerSettingValues(0, 4000)


def test_restore_attempts_cpu_when_gpu_reset_fails(tmp_path: Path) -> None:
    cpu_runner = FakeCpuRunner()
    gpu_runner = FakeGpuRunner()
    state_path = tmp_path / "hardware_protect.json"
    enable_protection(
        component="all",
        gpu_limit_mode="clock",
        gpu_clock_mhz=1500,
        cpu_runner=cpu_runner,
        gpu_runner=gpu_runner,
        state_path=state_path,
    )
    gpu_runner.fail_clock_reset = True

    with pytest.raises(HardwareProtectError, match="GPU restore failed"):
        disable_protection(
            cpu_runner=cpu_runner,
            gpu_runner=gpu_runner,
            state_path=state_path,
        )

    assert cpu_runner.settings[cpu.MAX_FREQUENCY_GUID] == cpu.PowerSettingValues(0, 4000)
    assert cpu_runner.settings[cpu.BOOST_MODE_GUID] == cpu.PowerSettingValues(2, 2)
    assert state_path.exists()

    gpu_runner.fail_clock_reset = False
    disable_protection(
        cpu_runner=cpu_runner,
        gpu_runner=gpu_runner,
        state_path=state_path,
    )
    assert not state_path.exists()


def test_power_mode_rejects_explicit_gpu_clock(tmp_path: Path) -> None:
    with pytest.raises(HardwareProtectError, match="requires"):
        enable_protection(
            component="all",
            gpu_limit_mode="power",
            gpu_clock_mhz=1500,
            cpu_runner=FakeCpuRunner(),
            gpu_runner=FakeGpuRunner(),
            state_path=tmp_path / "state.json",
        )


def test_cpu_only_ignores_gpu_specific_options(tmp_path: Path) -> None:
    enable_protection(
        component="cpu",
        gpu_limit_mode="power",
        gpu_clock_mhz=1500,
        cpu_runner=FakeCpuRunner(),
        gpu_runner=FakeGpuRunner(),
        state_path=tmp_path / "state.json",
    )


def test_gpu_index_must_be_non_negative(tmp_path: Path) -> None:
    with pytest.raises(HardwareProtectError, match="GPU index"):
        enable_protection(
            component="gpu",
            gpu_index=-1,
            gpu_limit_mode="clock",
            gpu_clock_mhz=1500,
            cpu_runner=FakeCpuRunner(),
            gpu_runner=FakeGpuRunner(),
            state_path=tmp_path / "state.json",
        )


def test_enable_refuses_to_overwrite_existing_restore_state(tmp_path: Path) -> None:
    state_path = tmp_path / "hardware_protect.json"
    state_path.write_text("{}", encoding="utf-8")

    with pytest.raises(HardwareProtectError, match="run --off first"):
        enable_protection(
            component="cpu",
            cpu_runner=FakeCpuRunner(),
            gpu_runner=FakeGpuRunner(),
            state_path=state_path,
        )


def test_disable_refuses_to_guess_without_restore_state(tmp_path: Path) -> None:
    with pytest.raises(HardwareProtectError, match="refusing to guess"):
        disable_protection(
            cpu_runner=FakeCpuRunner(),
            gpu_runner=FakeGpuRunner(),
            state_path=tmp_path / "missing.json",
        )


def test_status_combines_cpu_and_gpu_output(tmp_path: Path) -> None:
    output = format_status(
        component="all",
        cpu_runner=FakeCpuRunner(),
        gpu_runner=FakeGpuRunner(),
        state_path=tmp_path / "state.json",
    )

    assert "Hardware restore state: none" in output
    assert "CPU:" in output
    assert "AC unlimited (0 MHz), DC 4000 MHz" in output
    assert "GPU:" in output
    assert "Fake GPU" in output
