#!/usr/bin/env python3
"""Download Lichess standard rated PGN archives.

This script intentionally downloads human PGN archives from the Lichess open
database, not LCZero chunks, puzzle CSVs, or Stockfish evaluation JSON. Some
Lichess PGNs contain eval comments; downstream dataset processing must ignore
comments and use only human moves, game metadata, and final results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


BASE_URL = "https://database.lichess.org/standard"
LICENSE_NOTE = "CC0; see https://database.lichess.org/"
SOURCE_DESCRIPTION = "Lichess standard rated games monthly PGN archive"


@dataclass(frozen=True)
class DownloadRecord:
    month: str
    source_url: str
    source_description: str
    license: str
    acquired_at: str
    output_path: str
    size_bytes: int
    sha256: str | None
    status: str
    notes: str


def parse_month(value: str) -> tuple[int, int]:
    """Parse a month in YYYY-MM format."""
    try:
        year_text, month_text = value.split("-", maxsplit=1)
        year = int(year_text)
        month = int(month_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid month {value!r}; expected YYYY-MM") from exc

    if year < 2013 or month < 1 or month > 12:
        raise argparse.ArgumentTypeError(f"invalid month {value!r}; expected YYYY-MM")
    return year, month


def format_month(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def month_range(start_month: str, end_month: str) -> list[str]:
    """Return inclusive YYYY-MM months from start_month through end_month."""
    start_year, start = parse_month(start_month)
    end_year, end = parse_month(end_month)
    if (start_year, start) > (end_year, end):
        raise argparse.ArgumentTypeError("--start-month must be before or equal to --end-month")

    months: list[str] = []
    year = start_year
    month = start
    while (year, month) <= (end_year, end):
        months.append(format_month(year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def archive_url(month: str) -> str:
    parse_month(month)
    return f"{BASE_URL}/lichess_db_standard_rated_{month}.pgn.zst"


def unique_months(months: Iterable[str]) -> list[str]:
    parsed = [format_month(*parse_month(month)) for month in months]
    return sorted(set(parsed))


def sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_response_to_file(
    response: urllib.request.addinfourl,
    output_path: Path,
    mode: str,
    chunk_size: int,
) -> int:
    started = time.monotonic()
    last_report = started
    downloaded = output_path.stat().st_size if output_path.exists() and mode == "a" else 0

    with output_path.open(mode + "b") as file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_report >= 5.0:
                mb = downloaded / 1_000_000
                elapsed = max(now - started, 1e-6)
                speed = (downloaded / 1_000_000) / elapsed
                print(f"{output_path.name}: {mb:.1f} MB downloaded ({speed:.1f} MB/s)", file=sys.stderr)
                last_report = now
    return downloaded


def parse_content_range_total(value: str | None) -> int | None:
    """Parse the total size from a Content-Range header."""

    if not value:
        return None
    try:
        unit_and_range, total_text = value.split("/", maxsplit=1)
        unit, _range_text = unit_and_range.split(" ", maxsplit=1)
    except ValueError:
        return None
    if unit.lower() != "bytes" or total_text == "*":
        return None
    try:
        return int(total_text)
    except ValueError:
        return None


def parse_content_length(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def expected_download_size(
    response: urllib.request.addinfourl,
    *,
    status: int,
    resume_bytes: int,
    mode: str,
) -> int | None:
    """Return expected final file size when HTTP headers provide one."""

    content_range_total = parse_content_range_total(response.headers.get("Content-Range"))
    if content_range_total is not None:
        return content_range_total

    content_length = parse_content_length(response.headers.get("Content-Length"))
    if content_length is None:
        return None
    if status == 206 or mode == "a":
        return resume_bytes + content_length
    return content_length


def download_file(url: str, output_path: Path, chunk_size: int, timeout: int) -> str:
    """Download url to output_path using a .part file and resume when possible."""
    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    resume_bytes = partial_path.stat().st_size if partial_path.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "McChess data downloader"})
    if resume_bytes > 0:
        request.add_header("Range", f"bytes={resume_bytes}-")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            if resume_bytes > 0 and status == 206:
                mode = "a"
            else:
                mode = "w"
            expected_size = expected_download_size(
                response,
                status=status,
                resume_bytes=resume_bytes,
                mode=mode,
            )
            downloaded = copy_response_to_file(response, partial_path, mode, chunk_size)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(f"{url} was not found; check that this month has been released") from exc
        raise

    if expected_size is not None and downloaded != expected_size:
        raise RuntimeError(
            f"incomplete download for {url}: got {downloaded} bytes, expected "
            f"{expected_size}; kept partial file at {partial_path}"
        )

    shutil.move(str(partial_path), output_path)
    return "downloaded"


def write_manifest_record(manifest_path: Path, record: DownloadRecord) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def write_sidecar_record(output_path: Path, record: DownloadRecord) -> None:
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_record(
    month: str,
    output_path: Path,
    status: str,
    checksum: str | None,
) -> DownloadRecord:
    return DownloadRecord(
        month=month,
        source_url=archive_url(month),
        source_description=SOURCE_DESCRIPTION,
        license=LICENSE_NOTE,
        acquired_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        output_path=str(output_path),
        size_bytes=output_path.stat().st_size,
        sha256=checksum,
        status=status,
        notes=(
            "Raw PGN may include eval comments. Dataset builders must ignore comments and "
            "avoid using engine evaluations as labels."
        ),
    )


def resolve_months(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    selected.extend(args.months)

    if args.start_month or args.end_month:
        if not args.start_month or not args.end_month:
            raise SystemExit("--start-month and --end-month must be provided together")
        selected.extend(month_range(args.start_month, args.end_month))

    if not selected:
        raise SystemExit("provide one or more months, or use --start-month and --end-month")

    return unique_months(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Lichess standard rated monthly PGN .zst archives."
    )
    parser.add_argument("months", nargs="*", help="Months to download, e.g. 2025-01 2025-02.")
    parser.add_argument("--start-month", help="First month in an inclusive YYYY-MM range.")
    parser.add_argument("--end-month", help="Last month in an inclusive YYYY-MM range.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/lichess"),
        help="Directory for downloaded .pgn.zst files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/lichess_downloads.jsonl"),
        help="JSONL provenance manifest to append to.",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="Compute SHA256 after download. This is useful but slow for large archives.",
    )
    parser.add_argument("--force", action="store_true", help="Download even if the file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without fetching.")
    parser.add_argument("--timeout", type=int, default=60, help="Network timeout in seconds.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Download/read chunk size in bytes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    months = resolve_months(args)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for month in months:
        url = archive_url(month)
        output_path = args.output_dir / f"lichess_db_standard_rated_{month}.pgn.zst"

        if args.dry_run:
            print(f"{month}: {url} -> {output_path}")
            continue

        if output_path.exists() and not args.force:
            status = "skipped_existing"
            print(f"{month}: {output_path} already exists; skipping", file=sys.stderr)
        else:
            print(f"{month}: downloading {url}", file=sys.stderr)
            status = download_file(url, output_path, args.chunk_size, args.timeout)

        checksum = sha256_file(output_path, args.chunk_size) if args.checksum else None
        record = build_record(month, output_path, status, checksum)
        write_manifest_record(args.manifest, record)
        write_sidecar_record(output_path, record)
        print(f"{month}: {status} -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
