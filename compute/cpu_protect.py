"""Windows CPU frequency and boost controls used by hardware protection."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass


DEFAULT_MAX_FREQUENCY_MHZ = 3200
MIN_MAX_FREQUENCY_MHZ = 100
MAX_MAX_FREQUENCY_MHZ = 64_000
SUB_PROCESSOR_GUID = "54533251-82be-4824-96c1-47b60b740d00"
MAX_FREQUENCY_GUID = "75b0ae3f-bce0-45a7-8c89-c9611c25e100"
BOOST_MODE_GUID = "be337238-0d82-4146-a960-4f3749d470c7"
BOOST_DISABLED = 0

_GUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_AC_VALUE_RE = re.compile(
    r"Current AC Power Setting Index:\s*0x([0-9a-f]+)",
    re.IGNORECASE,
)
_DC_VALUE_RE = re.compile(
    r"Current DC Power Setting Index:\s*0x([0-9a-f]+)",
    re.IGNORECASE,
)
_HEX_VALUE_RE = re.compile(r"0x([0-9a-f]+)", re.IGNORECASE)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str]], CommandResult]


@dataclass(frozen=True)
class PowerSettingValues:
    ac: int
    dc: int


@dataclass(frozen=True)
class CpuPowerState:
    scheme_guid: str
    scheme_name: str | None
    max_frequency_mhz: PowerSettingValues
    boost_mode: PowerSettingValues


class CpuProtectError(RuntimeError):
    """Raised when Windows CPU power settings cannot be read or changed."""


def run_command(command: Sequence[str]) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError as exc:
        raise CpuProtectError(
            "powercfg was not found on PATH. CPU protection supports Windows only."
        ) from exc
    except OSError as exc:
        raise CpuProtectError(f"could not run powercfg: {exc}") from exc

    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def checked_powercfg(args: Sequence[str], runner: Runner = run_command) -> str:
    result = runner(["powercfg", *args])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise CpuProtectError(f"powercfg failed: {detail}")
    return result.stdout


def parse_active_scheme(output: str) -> tuple[str, str | None]:
    match = _GUID_RE.search(output)
    if match is None:
        raise CpuProtectError(f"could not parse the active power-plan GUID: {output!r}")
    scheme_guid = match.group(0).lower()
    name_match = re.search(r"\(([^()]*)\)", output[match.end() :])
    scheme_name = name_match.group(1).strip() if name_match is not None else None
    return scheme_guid, scheme_name or None


def parse_power_setting_values(output: str) -> PowerSettingValues:
    ac_match = _AC_VALUE_RE.search(output)
    dc_match = _DC_VALUE_RE.search(output)
    if ac_match is not None and dc_match is not None:
        return PowerSettingValues(
            ac=int(ac_match.group(1), 16),
            dc=int(dc_match.group(1), 16),
        )

    # Labels are localized. In a single-setting /qh response, the final two
    # hexadecimal values are the AC and DC setting indices.
    values = _HEX_VALUE_RE.findall(output)
    if len(values) < 2:
        raise CpuProtectError(f"could not parse AC/DC power-setting values: {output!r}")
    return PowerSettingValues(ac=int(values[-2], 16), dc=int(values[-1], 16))


def query_active_scheme(runner: Runner = run_command) -> tuple[str, str | None]:
    return parse_active_scheme(checked_powercfg(["/getactivescheme"], runner))


def query_power_setting(
    scheme_guid: str,
    setting_guid: str,
    runner: Runner = run_command,
) -> PowerSettingValues:
    output = checked_powercfg(
        ["/qh", scheme_guid, SUB_PROCESSOR_GUID, setting_guid],
        runner,
    )
    return parse_power_setting_values(output)


def query_cpu_power_state(runner: Runner = run_command) -> CpuPowerState:
    scheme_guid, scheme_name = query_active_scheme(runner)
    return query_cpu_power_state_for_scheme(
        scheme_guid,
        scheme_name=scheme_name,
        runner=runner,
    )


def query_cpu_power_state_for_scheme(
    scheme_guid: str,
    *,
    scheme_name: str | None = None,
    runner: Runner = run_command,
) -> CpuPowerState:
    return CpuPowerState(
        scheme_guid=scheme_guid.lower(),
        scheme_name=scheme_name,
        max_frequency_mhz=query_power_setting(scheme_guid, MAX_FREQUENCY_GUID, runner),
        boost_mode=query_power_setting(scheme_guid, BOOST_MODE_GUID, runner),
    )


def protected_cpu_power_state(
    current: CpuPowerState,
    max_frequency_mhz: int = DEFAULT_MAX_FREQUENCY_MHZ,
) -> CpuPowerState:
    if not MIN_MAX_FREQUENCY_MHZ <= max_frequency_mhz <= MAX_MAX_FREQUENCY_MHZ:
        raise CpuProtectError(
            "--cpu-max-frequency-mhz must be between "
            f"{MIN_MAX_FREQUENCY_MHZ} and {MAX_MAX_FREQUENCY_MHZ}"
        )
    return CpuPowerState(
        scheme_guid=current.scheme_guid,
        scheme_name=current.scheme_name,
        max_frequency_mhz=PowerSettingValues(
            ac=max_frequency_mhz,
            dc=max_frequency_mhz,
        ),
        boost_mode=PowerSettingValues(ac=BOOST_DISABLED, dc=BOOST_DISABLED),
    )


def set_power_setting(
    scheme_guid: str,
    setting_guid: str,
    values: PowerSettingValues,
    runner: Runner = run_command,
) -> None:
    for source, value in (("ac", values.ac), ("dc", values.dc)):
        checked_powercfg(
            [
                f"/set{source}valueindex",
                scheme_guid,
                SUB_PROCESSOR_GUID,
                setting_guid,
                str(value),
            ],
            runner,
        )


def apply_cpu_power_state(
    state: CpuPowerState,
    *,
    activate: bool,
    runner: Runner = run_command,
) -> None:
    set_power_setting(
        state.scheme_guid,
        MAX_FREQUENCY_GUID,
        state.max_frequency_mhz,
        runner,
    )
    set_power_setting(
        state.scheme_guid,
        BOOST_MODE_GUID,
        state.boost_mode,
        runner,
    )
    if activate:
        checked_powercfg(["/setactive", state.scheme_guid], runner)


def verify_cpu_power_state(expected: CpuPowerState, runner: Runner = run_command) -> None:
    actual = query_cpu_power_state_for_scheme(
        expected.scheme_guid,
        scheme_name=expected.scheme_name,
        runner=runner,
    )
    if actual.max_frequency_mhz != expected.max_frequency_mhz:
        raise CpuProtectError(
            "maximum CPU frequency did not match after powercfg completed: "
            f"expected {expected.max_frequency_mhz}, got {actual.max_frequency_mhz}"
        )
    if actual.boost_mode != expected.boost_mode:
        raise CpuProtectError(
            "processor boost mode did not match after powercfg completed: "
            f"expected {expected.boost_mode}, got {actual.boost_mode}"
        )


def cpu_state_to_payload(state: CpuPowerState) -> dict[str, object]:
    return {
        "scheme_guid": state.scheme_guid,
        "scheme_name": state.scheme_name,
        "max_frequency_mhz": {
            "ac": state.max_frequency_mhz.ac,
            "dc": state.max_frequency_mhz.dc,
        },
        "boost_mode": {"ac": state.boost_mode.ac, "dc": state.boost_mode.dc},
    }


def _payload_values(payload: object, field_name: str) -> PowerSettingValues:
    if not isinstance(payload, dict):
        raise CpuProtectError(f"saved hardware state has invalid {field_name!r}")
    ac = payload.get("ac")
    dc = payload.get("dc")
    if not isinstance(ac, int) or isinstance(ac, bool) or ac < 0:
        raise CpuProtectError(f"saved hardware state has invalid {field_name}.ac")
    if not isinstance(dc, int) or isinstance(dc, bool) or dc < 0:
        raise CpuProtectError(f"saved hardware state has invalid {field_name}.dc")
    return PowerSettingValues(ac=ac, dc=dc)


def cpu_state_from_payload(payload: object) -> CpuPowerState:
    if not isinstance(payload, dict):
        raise CpuProtectError("saved CPU restore state must be a JSON object")
    scheme_guid = payload.get("scheme_guid")
    if not isinstance(scheme_guid, str) or _GUID_RE.fullmatch(scheme_guid) is None:
        raise CpuProtectError("saved hardware state has an invalid CPU scheme_guid")
    scheme_name = payload.get("scheme_name")
    if scheme_name is not None and not isinstance(scheme_name, str):
        raise CpuProtectError("saved hardware state has an invalid CPU scheme_name")
    return CpuPowerState(
        scheme_guid=scheme_guid.lower(),
        scheme_name=scheme_name,
        max_frequency_mhz=_payload_values(
            payload.get("max_frequency_mhz"),
            "cpu_restore.max_frequency_mhz",
        ),
        boost_mode=_payload_values(payload.get("boost_mode"), "cpu_restore.boost_mode"),
    )


def format_cpu_status(state: CpuPowerState) -> str:
    name = state.scheme_name or "unnamed"
    ac_frequency = (
        "unlimited (0 MHz)"
        if state.max_frequency_mhz.ac == 0
        else f"{state.max_frequency_mhz.ac} MHz"
    )
    dc_frequency = (
        "unlimited (0 MHz)"
        if state.max_frequency_mhz.dc == 0
        else f"{state.max_frequency_mhz.dc} MHz"
    )
    boost_names = {
        0: "Disabled",
        1: "Enabled",
        2: "Aggressive",
        3: "Efficient Enabled",
        4: "Efficient Aggressive",
        5: "Aggressive At Guaranteed",
        6: "Efficient Aggressive At Guaranteed",
    }
    ac_boost = boost_names.get(state.boost_mode.ac, "Unknown")
    dc_boost = boost_names.get(state.boost_mode.dc, "Unknown")
    return "\n".join(
        [
            f"power plan: {name} ({state.scheme_guid})",
            f"maximum frequency: AC {ac_frequency}, DC {dc_frequency}",
            (
                f"boost mode: AC {ac_boost} ({state.boost_mode.ac}), "
                f"DC {dc_boost} ({state.boost_mode.dc})"
            ),
        ]
    )
