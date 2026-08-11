"""Export an inference-only McChess model artifact."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mcchess.model import export_inference_artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export model weights and audit metadata without optimizer state.",
    )
    parser.add_argument("source", type=Path, help="Source supervised training checkpoint.")
    parser.add_argument("output", type=Path, help="Immutable inference artifact to create.")
    parser.add_argument("--artifact-id", required=True, help="Stable published artifact ID.")
    parser.add_argument(
        "--exported-at",
        default=None,
        help="Optional fixed provenance timestamp for byte-for-byte reproduction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output artifact. Disabled by default.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exported = export_inference_artifact(
        args.source,
        args.output,
        artifact_id=args.artifact_id,
        overwrite=args.overwrite,
        exported_at=args.exported_at,
    )
    print(f"artifact: {exported.artifact_id}")
    print(f"source: {exported.source_path}")
    print(f"source sha256: {exported.source_sha256}")
    print(f"output: {exported.output_path}")
    print(f"output sha256: {exported.output_sha256}")
    print(f"output bytes: {exported.output_size_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
