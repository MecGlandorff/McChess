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
