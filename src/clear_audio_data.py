#!/usr/bin/env python3
"""Safely clear local audio workflow data while preserving its folders."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
TARGETS = (DATA_DIR / "archive", DATA_DIR / "incoming", DATA_DIR / "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the contents of the local audio archive, incoming, and results folders."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete without the interactive confirmation prompt.",
    )
    return parser.parse_args()


def count_entries() -> int:
    return sum(1 for target in TARGETS if target.exists() for _ in target.iterdir())


def clear_directory(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for entry in target.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def main() -> int:
    args = parse_args()
    entry_count = count_entries()
    print("This permanently removes local files from:")
    for target in TARGETS:
        print(f"  - {target}")
    print(f"Top-level entries to remove: {entry_count}")

    if not args.yes:
        answer = input("Type CLEAR to continue: ").strip()
        if answer != "CLEAR":
            print("Nothing was removed.")
            return 0

    for target in TARGETS:
        clear_directory(target)
    print("Audio archive, incoming, and results folders are empty.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, PermissionError) as error:
        print(f"Error while clearing audio data: {error}", file=sys.stderr)
        raise SystemExit(1)
