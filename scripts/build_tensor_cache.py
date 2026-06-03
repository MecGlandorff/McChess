#!/usr/bin/env python3
"""Build a tensor cache from a supervised JSONL shard."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcchess.data import build_supervised_tensor_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a supervised tensor cache.")
    parser.add_argument("shard", type=Path, help="Input supervised JSONL shard.")
    parser.add_argument("output_dir", type=Path, help="Output cache directory.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress while encoding samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = build_supervised_tensor_cache(
        args.shard,
        args.output_dir,
        show_progress=not args.no_progress,
    )
    print(f"wrote tensor cache manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
