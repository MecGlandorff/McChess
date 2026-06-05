#!/usr/bin/env python3
"""Filter PGN files by supported header filters.

This script is intended for large Lichess monthly archives. It can stream
`.pgn.zst` inputs through the external `zstd` command so unfiltered PGNs do not
need to be decompressed to disk before filtering.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from tqdm.auto import tqdm

from mcchess.data.pgn_reader import game_passes_filters


TMP_SUFFIX = ".part"


@dataclass(frozen=True)
class PgnFilterConfig:
    min_elo: int | None = None
    min_elo_mode: str = "both"
    require_rated: bool = False
    max_kept_games: int | None = None

    def as_filter_mapping(self) -> dict[str, object]:
        filters: dict[str, object] = {
            "min_elo_mode": self.min_elo_mode,
            "require_rated": self.require_rated,
        }
        if self.min_elo is not None:
            filters["min_elo"] = self.min_elo
        if self.max_kept_games is not None:
            filters["max_kept_games"] = self.max_kept_games
        return filters


@dataclass(frozen=True)
class PgnGameRecord:
    headers: dict[str, str]
    lines: list[str]
    malformed_header_lines: int


@dataclass
class PgnTextSource:
    stream: TextIO
    process: subprocess.Popen[str] | None = None
    stopped_early: bool = False


@dataclass(frozen=True)
class FilterManifest:
    source_paths: list[str]
    output_path: str
    filters: dict[str, object]
    games_read: int
    games_written: int
    games_skipped_filter: int
    games_skipped_corrupt: int
    stopped_early: bool
    created_at: str
    notes: str


@contextmanager
def open_pgn_text(path: Path) -> Iterator[PgnTextSource]:
    if path.suffix == ".zst":
        process = subprocess.Popen(
            ["zstd", "-dc", str(path)],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:
            raise RuntimeError("zstd did not provide stdout")
        source = PgnTextSource(stream=process.stdout, process=process)
        try:
            yield source
        finally:
            process.stdout.close()
            if source.stopped_early and process.poll() is None:
                process.terminate()
            return_code = process.wait()
            if return_code != 0 and not source.stopped_early:
                raise RuntimeError(f"zstd failed for {path} with exit code {return_code}")
    else:
        with path.open(encoding="utf-8", errors="replace") as file:
            yield PgnTextSource(stream=file)


def iter_pgn_records(stream: TextIO) -> Iterator[PgnGameRecord]:
    """Yield raw PGN games with parsed headers without parsing movetext."""

    current_lines: list[str] = []
    for line in stream:
        if current_lines and is_new_game_line(line):
            yield build_record(current_lines)
            current_lines = []
        if not current_lines and not line.strip():
            continue
        current_lines.append(line)

    if current_lines:
        yield build_record(current_lines)


def build_record(lines: list[str]) -> PgnGameRecord:
    headers: dict[str, str] = {}
    malformed_header_lines = 0
    saw_header = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if saw_header:
                break
            continue
        if not stripped.startswith("["):
            break

        saw_header = True
        header = parse_header_line(stripped)
        if header is None:
            malformed_header_lines += 1
            continue
        key, value = header
        headers[key] = value

    if not headers:
        malformed_header_lines += 1
    return PgnGameRecord(
        headers=headers,
        lines=lines,
        malformed_header_lines=malformed_header_lines,
    )


def is_new_game_line(line: str) -> bool:
    return line.lstrip("\ufeff").startswith("[Event ")


def parse_header_line(line: str) -> tuple[str, str] | None:
    if not line.startswith("[") or not line.endswith("]"):
        return None

    inner = line[1:-1]
    try:
        key, raw_value = inner.split(" ", maxsplit=1)
    except ValueError:
        return None

    raw_value = raw_value.strip()
    if not key or len(raw_value) < 2 or raw_value[0] != '"' or raw_value[-1] != '"':
        return None
    return key, unescape_pgn_string(raw_value[1:-1])


def unescape_pgn_string(value: str) -> str:
    chars: list[str] = []
    escaped = False
    for char in value:
        if escaped:
            chars.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            chars.append(char)
    if escaped:
        chars.append("\\")
    return "".join(chars)


def write_record(out: TextIO, record: PgnGameRecord) -> None:
    out.writelines(record.lines)
    if record.lines and not record.lines[-1].endswith("\n"):
        out.write("\n")
    out.write("\n")


def filter_pgn_files(
    sources: list[Path],
    output_path: Path,
    manifest_path: Path,
    config: PgnFilterConfig,
    *,
    show_progress: bool = True,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    filters = config.as_filter_mapping()

    games_read = 0
    games_written = 0
    games_skipped_filter = 0
    games_skipped_corrupt = 0
    stopped_early = False
    tmp_output_path = output_path.with_name(output_path.name + TMP_SUFFIX)
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    progress = (
        tqdm(desc="filtering PGN headers", unit="game", dynamic_ncols=True)
        if show_progress
        else None
    )
    try:
        with tmp_output_path.open("w", encoding="utf-8", newline="\n") as out:
            for source in sources:
                with open_pgn_text(source) as text_source:
                    for record in iter_pgn_records(text_source.stream):
                        games_read += 1
                        if record.malformed_header_lines:
                            games_skipped_corrupt += 1
                        elif game_passes_filters(record.headers, filters):
                            write_record(out, record)
                            games_written += 1
                        else:
                            games_skipped_filter += 1

                        if progress is not None:
                            progress.update(1)
                            if games_read == 1 or games_read % 10000 == 0:
                                progress.set_postfix(
                                    kept=games_written,
                                    skipped=games_skipped_filter,
                                    corrupt=games_skipped_corrupt,
                                )
                        if (
                            config.max_kept_games is not None
                            and games_written >= config.max_kept_games
                        ):
                            stopped_early = True
                            text_source.stopped_early = True
                            break
                if stopped_early:
                    break
    finally:
        if progress is not None:
            progress.close()

    os.replace(tmp_output_path, output_path)
    manifest = FilterManifest(
        source_paths=[str(source) for source in sources],
        output_path=str(output_path),
        filters=filters,
        games_read=games_read,
        games_written=games_written,
        games_skipped_filter=games_skipped_filter,
        games_skipped_corrupt=games_skipped_corrupt,
        stopped_early=stopped_early,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        notes=(
            "Filtered PGN is a derived artifact from human Lichess PGNs. Filtering "
            "uses PGN headers only and preserves matching game text for the dataset "
            "builder to parse and validate moves. No engine labels or tablebase data "
            "are used."
        ),
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter PGN archives by metadata headers.")
    parser.add_argument("sources", nargs="+", type=Path, help="Input .pgn or .pgn.zst files.")
    parser.add_argument("--output", required=True, type=Path, help="Output filtered PGN path.")
    parser.add_argument("--manifest", required=True, type=Path, help="Output filter manifest path.")
    parser.add_argument("--min-elo", type=int, help="Minimum Elo threshold.")
    parser.add_argument(
        "--min-elo-mode",
        choices=["both", "either", "white", "black", "average"],
        default="both",
        help="How to apply --min-elo. Default: both players must meet it.",
    )
    parser.add_argument(
        "--require-rated",
        action="store_true",
        help="Require a Lichess-style rated game header.",
    )
    parser.add_argument(
        "--max-kept-games",
        type=int,
        help="Stop after writing this many matching games. Omit to keep all matches.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PgnFilterConfig(
        min_elo=args.min_elo,
        min_elo_mode=args.min_elo_mode,
        require_rated=args.require_rated,
        max_kept_games=args.max_kept_games,
    )
    manifest_path = filter_pgn_files(
        args.sources,
        args.output,
        args.manifest,
        config,
        show_progress=not args.no_progress,
    )
    print(f"wrote filter manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
