from __future__ import annotations

from collections.abc import Sequence

import pytest

from compute.cpu_protect import (
    BOOST_MODE_GUID,
    MAX_FREQUENCY_GUID,
    SUB_PROCESSOR_GUID,
    CommandResult,
    CpuPowerState,
    CpuProtectError,
    PowerSettingValues,
    apply_cpu_power_state,
    cpu_state_from_payload,
    cpu_state_to_payload,
    parse_active_scheme,
    parse_power_setting_values,
    protected_cpu_power_state,
    query_cpu_power_state,
    verify_cpu_power_state,
)


SCHEME_GUID = "64a64f24-65b9-4b56-befd-5ec1eaced9b3"


class FakePowercfgRunner:
    def __init__(self) -> None:
        self.active_scheme_guid = SCHEME_GUID
        self.settings = {
            MAX_FREQUENCY_GUID: PowerSettingValues(ac=0, dc=4000),
            BOOST_MODE_GUID: PowerSettingValues(ac=2, dc=2),
        }
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> CommandResult:
        saved = list(command)
        self.commands.append(saved)
        action = saved[1].lower()
        if action == "/getactivescheme":
            return CommandResult(
                0,
                f"Power Scheme GUID: {self.active_scheme_guid}  (Silent)\n",
                "",
            )
        if action == "/qh":
            values = self.settings[saved[-1].lower()]
            return CommandResult(
                0,
                (
                    "Minimum Possible Setting: 0x00000000\n"
                    "Maximum Possible Setting: 0xffffffff\n"
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
                self.settings[setting_guid] = PowerSettingValues(value, current.dc)
            else:
                self.settings[setting_guid] = PowerSettingValues(current.ac, value)
            return CommandResult(0, "", "")
        if action == "/setactive":
            self.active_scheme_guid = saved[2].lower()
            return CommandResult(0, "", "")
        raise AssertionError(f"unexpected command: {saved}")


def original_state() -> CpuPowerState:
    return CpuPowerState(
        scheme_guid=SCHEME_GUID,
        scheme_name="Silent",
        max_frequency_mhz=PowerSettingValues(ac=0, dc=4000),
        boost_mode=PowerSettingValues(ac=2, dc=2),
    )


def test_parse_active_scheme_accepts_powercfg_output() -> None:
    assert parse_active_scheme(
        f"Power Scheme GUID: {SCHEME_GUID.upper()}  (Silent)\n"
    ) == (SCHEME_GUID, "Silent")


def test_parse_power_setting_values_accepts_named_indices() -> None:
    output = (
        "Current AC Power Setting Index: 0x00000c80\n"
        "Current DC Power Setting Index: 0x00000fa0\n"
    )
    assert parse_power_setting_values(output) == PowerSettingValues(ac=3200, dc=4000)


def test_parse_power_setting_values_supports_localized_labels() -> None:
    output = (
        "minimum: 0x00000000\n"
        "maximum: 0xffffffff\n"
        "courant secteur: 0x00000c80\n"
        "courant batterie: 0x00000fa0\n"
    )
    assert parse_power_setting_values(output) == PowerSettingValues(ac=3200, dc=4000)


def test_parse_power_setting_values_rejects_missing_values() -> None:
    with pytest.raises(CpuProtectError):
        parse_power_setting_values("not available")


def test_protected_cpu_state_caps_both_sources_and_disables_boost() -> None:
    target = protected_cpu_power_state(original_state(), 3200)

    assert target.max_frequency_mhz == PowerSettingValues(ac=3200, dc=3200)
    assert target.boost_mode == PowerSettingValues(ac=0, dc=0)


@pytest.mark.parametrize("frequency", [0, 99, 64_001])
def test_protected_cpu_state_rejects_invalid_frequency(frequency: int) -> None:
    with pytest.raises(CpuProtectError):
        protected_cpu_power_state(original_state(), frequency)


def test_apply_and_verify_cpu_state() -> None:
    runner = FakePowercfgRunner()
    target = protected_cpu_power_state(original_state(), 3200)

    apply_cpu_power_state(target, activate=True, runner=runner)
    verify_cpu_power_state(target, runner)

    assert runner.settings[MAX_FREQUENCY_GUID] == PowerSettingValues(ac=3200, dc=3200)
    assert runner.settings[BOOST_MODE_GUID] == PowerSettingValues(ac=0, dc=0)
    assert [command for command in runner.commands if command[1].startswith("/set")] == [
        [
            "powercfg",
            "/setacvalueindex",
            SCHEME_GUID,
            SUB_PROCESSOR_GUID,
            MAX_FREQUENCY_GUID,
            "3200",
        ],
        [
            "powercfg",
            "/setdcvalueindex",
            SCHEME_GUID,
            SUB_PROCESSOR_GUID,
            MAX_FREQUENCY_GUID,
            "3200",
        ],
        [
            "powercfg",
            "/setacvalueindex",
            SCHEME_GUID,
            SUB_PROCESSOR_GUID,
            BOOST_MODE_GUID,
            "0",
        ],
        [
            "powercfg",
            "/setdcvalueindex",
            SCHEME_GUID,
            SUB_PROCESSOR_GUID,
            BOOST_MODE_GUID,
            "0",
        ],
        ["powercfg", "/setactive", SCHEME_GUID],
    ]


def test_query_cpu_power_state_reads_active_plan() -> None:
    assert query_cpu_power_state(FakePowercfgRunner()) == original_state()


def test_cpu_state_payload_round_trip() -> None:
    assert cpu_state_from_payload(cpu_state_to_payload(original_state())) == original_state()


def test_cpu_state_payload_rejects_boolean_value() -> None:
    payload = cpu_state_to_payload(original_state())
    frequencies = payload["max_frequency_mhz"]
    assert isinstance(frequencies, dict)
    frequencies["ac"] = True

    with pytest.raises(CpuProtectError):
        cpu_state_from_payload(payload)
