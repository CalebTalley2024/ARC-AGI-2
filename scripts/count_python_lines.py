#!/usr/bin/env python3
"""
Utility to count lines of Python source code in the repository.

This script traverses the project directory (one level up from scripts/)
and sums the number of lines across all files with the .py extension.
It prints a per-file breakdown and the total.

Usage:
  python scripts/count_python_lines.py [path]  # path optional, default is repo root
"""

import os
import sys
from typing import List, Tuple


def is_python_file(path: str) -> bool:
    return path.endswith(".py")


def count_lines_in_file(file_path: str) -> int:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def scan_directory(root_path: str) -> Tuple[List[Tuple[str, int]], int]:
    per_file: List[Tuple[str, int]] = []
    total_lines = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        # skip hidden/system dirs for speed
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
        for filename in filenames:
            if is_python_file(filename) or filename.endswith(".pyi"):
                file_path = os.path.join(dirpath, filename)
                lines = count_lines_in_file(file_path)
                per_file.append((file_path, lines))
                total_lines += lines
    return per_file, total_lines


def main() -> None:
    # default root is directory two levels up from scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, ".."))
    root = sys.argv[1] if len(sys.argv) > 1 else default_root

    per_file, total_lines = scan_directory(root)
    # sort by path for stable output
    per_file.sort(key=lambda t: t[0])
    for path, lines in per_file:
        print(f"{lines:8}  {path}")
    print(f"Total Python lines: {total_lines}")


if __name__ == "__main__":
    main()

