from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_lichess.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("download_lichess", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_archive_url_uses_lichess_standard_rated_archive() -> None:
    script = load_script_module()

    assert (
        script.archive_url("2025-01")
        == "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
    )


def test_month_range_is_inclusive_across_year_boundary() -> None:
    script = load_script_module()

    assert script.month_range("2024-11", "2025-02") == [
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
    ]


def test_invalid_month_is_rejected() -> None:
    script = load_script_module()

    with pytest.raises(argparse.ArgumentTypeError):
        script.parse_month("2025-13")


def test_unique_months_normalizes_and_sorts() -> None:
    script = load_script_module()

    assert script.unique_months(["2025-02", "2025-01", "2025-02"]) == ["2025-01", "2025-02"]


def test_dry_run_does_not_create_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script_module()
    output_dir = tmp_path / "raw" / "lichess"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_lichess.py",
            "2025-01",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert script.main() == 0
    assert not output_dir.exists()


def test_content_range_total_is_parsed() -> None:
    script = load_script_module()

    assert script.parse_content_range_total("bytes 100-199/1234") == 1234
    assert script.parse_content_range_total("bytes 100-199/*") is None
    assert script.parse_content_range_total("bad") is None


def test_download_keeps_partial_when_response_is_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script_module()
    output_path = tmp_path / "archive.pgn.zst"

    class ShortResponse:
        headers = {"Content-Length": "10"}

        def __init__(self) -> None:
            self._chunks = [b"abc", b""]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def getcode(self) -> int:
            return 200

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    monkeypatch.setattr(script.urllib.request, "urlopen", lambda *_args, **_kwargs: ShortResponse())

    with pytest.raises(RuntimeError, match="incomplete download"):
        script.download_file("https://example.test/archive.pgn.zst", output_path, 1024, 60)

    assert not output_path.exists()
    assert output_path.with_suffix(".zst.part").read_bytes() == b"abc"
