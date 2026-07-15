"""Process-scoped sleep prevention for long evaluation runs."""

from __future__ import annotations

import ctypes
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

ES_SYSTEM_REQUIRED: Final[int] = 0x00000001
ES_DISPLAY_REQUIRED: Final[int] = 0x00000002
ES_CONTINUOUS: Final[int] = 0x80000000

ExecutionStateSetter = Callable[[int], int]


@contextmanager
def keep_system_awake(
    *,
    enabled: bool,
    platform: str | None = None,
    setter: ExecutionStateSetter | None = None,
) -> Iterator[None]:
    """Prevent automatic Windows system/display sleep for the context lifetime."""

    current_platform = sys.platform if platform is None else platform
    if not enabled or current_platform != "win32":
        yield
        return

    set_execution_state = setter or _set_thread_execution_state
    required_state = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    if set_execution_state(required_state) == 0:
        raise OSError(
            _get_windows_last_error(),
            "failed to request Windows keep-awake state",
        )

    try:
        yield
    finally:
        if set_execution_state(ES_CONTINUOUS) == 0:
            warnings.warn(
                "failed to restore the Windows execution state",
                RuntimeWarning,
                stacklevel=2,
            )


def _get_windows_last_error() -> int:
    if sys.platform != "win32":
        return 0
    return ctypes.get_last_error()


def _set_thread_execution_state(flags: int) -> int:
    if sys.platform != "win32":
        raise OSError("Windows execution-state requests are only available on Windows")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]
    function.restype = ctypes.c_uint
    return int(function(flags))
