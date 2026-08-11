import subprocess
import sys


def test_package_imports() -> None:
    import mcchess

    assert mcchess.__version__ == "0.1.0"


def test_search_imports_before_bots_in_fresh_interpreter() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import mcchess.search; import mcchess.bots"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
