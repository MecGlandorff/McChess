from __future__ import annotations

import pytest

from mcchess.eval.keep_awake import (
    ES_CONTINUOUS,
    ES_DISPLAY_REQUIRED,
    ES_SYSTEM_REQUIRED,
    keep_system_awake,
)


def test_keep_system_awake_requests_and_restores_windows_state() -> None:
    calls: list[int] = []

    def setter(flags: int) -> int:
        calls.append(flags)
        return 1

    with keep_system_awake(enabled=True, platform="win32", setter=setter):
        assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED]

    assert calls == [
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED,
        ES_CONTINUOUS,
    ]


@pytest.mark.parametrize("enabled,platform", [(False, "win32"), (True, "linux")])
def test_keep_system_awake_is_noop_when_disabled_or_not_windows(
    enabled: bool,
    platform: str,
) -> None:
    calls: list[int] = []

    with keep_system_awake(
        enabled=enabled,
        platform=platform,
        setter=lambda flags: calls.append(flags) or 1,
    ):
        pass

    assert calls == []


def test_keep_system_awake_rejects_failed_windows_request() -> None:
    with pytest.raises(OSError, match="keep-awake"):
        with keep_system_awake(enabled=True, platform="win32", setter=lambda _flags: 0):
            pass
