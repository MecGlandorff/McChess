from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "check_repository_hygiene.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_repository_hygiene", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_hygiene_accepts_the_current_repository() -> None:
    script = _load_script()
    paths = script.candidate_repository_paths(REPOSITORY_ROOT)

    assert script.repository_hygiene_violations(REPOSITORY_ROOT, paths) == []


def test_repository_hygiene_rejects_run_and_extra_checkpoint_paths(tmp_path: Path) -> None:
    script = _load_script()
    paths = [
        PurePosixPath("runs/example/result.json"),
        PurePosixPath("models_archive/extra.pt"),
    ]

    violations = script.repository_hygiene_violations(tmp_path, paths)

    assert violations == [
        "checkpoint is not on the publication allowlist: models_archive/extra.pt",
        "local research artifact is not publishable: runs/example/result.json",
    ]
