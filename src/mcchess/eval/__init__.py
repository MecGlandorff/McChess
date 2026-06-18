"""Evaluation and arena utilities."""

from typing import Any

__all__ = [
    "ArenaConfig",
    "BotConfig",
    "GameRecord",
    "build_bot",
    "play_game",
    "run_match",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from mcchess.eval import arena

        return getattr(arena, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
