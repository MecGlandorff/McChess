from __future__ import annotations

from collections.abc import Sequence

import pytest

from compute.gpu_protect import (
    CommandResult,
    GpuProtectError,
    disable_limit,
    enable_limit,
    parse_power_watts,
    protected_watts,
)


class FakeRunner:
    def __init__(self, default_watts: str = "200.00\n") -> None:
        self.default_watts = default_watts
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> CommandResult:
        saved = list(command)
        self.commands.append(saved)
        if "--query-gpu=power.default_limit" in saved:
            return CommandResult(returncode=0, stdout=self.default_watts, stderr="")
        return CommandResult(returncode=0, stdout="", stderr="")


def test_parse_power_watts_accepts_plain_or_unit_output() -> None:
    assert parse_power_watts("250.00\n") == 250.0
    assert parse_power_watts("250.00 W\n") == 250.0


def test_parse_power_watts_rejects_missing_number() -> None:
    with pytest.raises(GpuProtectError):
        parse_power_watts("not available")


def test_protected_watts_uses_floor_percent() -> None:
    assert protected_watts(201.0, 75.0) == 150


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


def test_dry_run_does_not_set_power_limit() -> None:
    runner = FakeRunner(default_watts="200.00\n")

    change = enable_limit(gpu=0, percent=75.0, runner=runner, dry_run=True)

    assert change.target_watts == 150
    assert len(runner.commands) == 1
