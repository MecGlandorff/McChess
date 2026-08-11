"""Reject local research artifacts from the repository surface."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_MODEL_ARTIFACTS = frozenset({PurePosixPath("models_archive/resnet_c_epoch_030.pt")})
FORBIDDEN_DIRECTORY_PREFIXES = (
    PurePosixPath("runs"),
    PurePosixPath("checkpoints"),
    PurePosixPath("data/raw"),
    PurePosixPath("data/processed"),
    PurePosixPath("data/tensor_cache"),
)
CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})


def candidate_repository_paths(repository_root: Path) -> list[PurePosixPath]:
    """Return tracked and unignored working-tree paths from Git."""

    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return [
        PurePosixPath(raw_path.decode("utf-8"))
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    ]


def repository_hygiene_violations(
    repository_root: Path,
    repository_paths: Iterable[PurePosixPath],
) -> list[str]:
    """Return deterministic descriptions of repository-policy violations."""

    violations: list[str] = []
    for relative_path in sorted(repository_paths, key=str):
        if relative_path.name == ".gitkeep":
            continue
        if any(relative_path == prefix or prefix in relative_path.parents for prefix in FORBIDDEN_DIRECTORY_PREFIXES):
            violations.append(f"local research artifact is not publishable: {relative_path}")
            continue
        if relative_path.suffix.lower() in CHECKPOINT_SUFFIXES and relative_path not in ALLOWED_MODEL_ARTIFACTS:
            violations.append(f"checkpoint is not on the publication allowlist: {relative_path}")
            continue

        local_path = repository_root.joinpath(*relative_path.parts)
        if local_path.is_file() and local_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            violations.append(
                f"file exceeds {MAX_FILE_SIZE_BYTES} bytes: {relative_path} "
                f"({local_path.stat().st_size} bytes)"
            )
    return violations


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    violations = repository_hygiene_violations(
        repository_root,
        candidate_repository_paths(repository_root),
    )
    if violations:
        print("Repository hygiene check failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
