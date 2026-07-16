from __future__ import annotations

from collections.abc import Sequence

import pytest

from compute.gpu_protect import (
    CommandResult,
    GpuProtectError,
    disable_limit,
    disable_clock_limit,
    enable_clock_limit,
    enable_limit,
    format_change,
    parse_clock_mhz_values,
    parse_power_watts,
    protected_clock_mhz,
    protected_watts,
)


class FakeRunner:
    def __init__(
        self,
        default_watts: str = "200.00\n",
        supported_clocks: str = "3105\n930\n615\n",
    ) -> None:
        self.default_watts = default_watts
        self.supported_clocks = supported_clocks
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> CommandResult:
        saved = list(command)
        self.commands.append(saved)
        if "--query-gpu=power.default_limit" in saved:
            return CommandResult(returncode=0, stdout=self.default_watts, stderr="")
        if "--query-supported-clocks=gr" in saved:
            return CommandResult(returncode=0, stdout=self.supported_clocks, stderr="")
        return CommandResult(returncode=0, stdout="", stderr="")


def test_parse_power_watts_accepts_plain_or_unit_output() -> None:
    assert parse_power_watts("250.00\n") == 250.0
    assert parse_power_watts("250.00 W\n") == 250.0


def test_parse_power_watts_rejects_missing_number() -> None:
    with pytest.raises(GpuProtectError):
        parse_power_watts("not available")


def test_parse_clock_mhz_values_accepts_units_and_deduplicates() -> None:
    assert parse_clock_mhz_values("930 MHz\n615\n930.00 MHz\n") == [615, 930]


def test_parse_clock_mhz_values_rejects_missing_number() -> None:
    with pytest.raises(GpuProtectError):
        parse_clock_mhz_values("not available")


def test_protected_watts_uses_floor_percent() -> None:
    assert protected_watts(201.0, 75.0) == 150


def test_protected_clock_mhz_uses_supported_floor_percent() -> None:
    assert protected_clock_mhz([615, 930, 3105], 30.0) == 930


def test_protected_clock_mhz_uses_lowest_supported_clock_as_floor() -> None:
    assert protected_clock_mhz([615, 930, 3105], 5.0) == 615


def test_enable_limit_sets_percent_of_default() -> None:
    runner = FakeRunner(default_watts="200.00\n")

    change = enable_limit(gpu=1, percent=75.0, runner=runner)

    assert change.target_watts == 150
    assert runner.commands == [
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-gpu=power.default_limit",
            "--format=csv,noheader,nounits",
        ],
        ["nvidia-smi", "-i", "1", "-pl", "150"],
    ]


def test_disable_limit_restores_default() -> None:
    runner = FakeRunner(default_watts="200.00\n")

    change = disable_limit(gpu=0, runner=runner)

    assert change.target_watts == 200
    assert runner.commands[-1] == ["nvidia-smi", "-i", "0", "-pl", "200"]


def test_enable_clock_limit_sets_percent_of_max_supported_clock() -> None:
    runner = FakeRunner(supported_clocks="3105\n930\n615\n")

    change = enable_clock_limit(gpu=1, percent=30.0, runner=runner)

    assert change.target_clock_mhz == 930
    assert change.reference_clock_mhz == 3105
    assert runner.commands == [
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-supported-clocks=gr",
            "--format=csv,noheader,nounits",
        ],
        ["nvidia-smi", "-i", "1", "-lgc", "930"],
    ]


def test_enable_clock_limit_accepts_explicit_clock() -> None:
    runner = FakeRunner()

    change = enable_clock_limit(gpu=0, gpu_clock_mhz=900, runner=runner)

    assert change.target_clock_mhz == 900
    assert change.reference_clock_mhz is None
    assert runner.commands == [["nvidia-smi", "-i", "0", "-lgc", "900"]]


def test_disable_clock_limit_resets_graphics_clock() -> None:
    runner = FakeRunner()

    change = disable_clock_limit(gpu=0, runner=runner)

    assert change.enabled is False
    assert runner.commands == [["nvidia-smi", "-i", "0", "-rgc"]]


def test_dry_run_does_not_set_power_limit() -> None:
    runner = FakeRunner(default_watts="200.00\n")

    change = enable_limit(gpu=0, percent=75.0, runner=runner, dry_run=True)

    assert change.target_watts == 150
    assert len(runner.commands) == 1


def test_clock_dry_run_does_not_set_graphics_clock() -> None:
    runner = FakeRunner()

    change = enable_clock_limit(gpu=0, percent=30.0, runner=runner, dry_run=True)

    assert change.target_clock_mhz == 930
    assert len(runner.commands) == 1


def test_format_explicit_clock_limit_has_no_empty_details() -> None:
    change = enable_clock_limit(
        gpu=0,
        gpu_clock_mhz=900,
        runner=FakeRunner(),
        dry_run=True,
    )

    assert format_change(change, dry_run=True) == (
        "GPU 0: would set protected graphics clock limit to 900 MHz."
    )
