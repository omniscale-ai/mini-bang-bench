from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_MAX_LINES = 500
DEFAULT_EXCLUDES = {
    ".git",
    "workdir",  # All generated data lives outside the source line budget.
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "src/mini_bang.egg-info",
}
BINARY_SUFFIXES = {
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pyc",
    ".pyo",
    ".svg",
    ".webp",
}
UNBOUNDED_TEXT_SUFFIXES = {
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".ipynb",
    ".md",
    ".mdx",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if source files exceed the line budget.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    violations = []
    for path in iter_files(root):
        lines = line_count(path)
        if lines > args.max_lines:
            violations.append((lines, path.relative_to(root)))
    if violations:
        for lines, path in sorted(violations, reverse=True):
            print(f"{lines:5d} {path}", file=sys.stderr)
        return 1
    return 0


def iter_files(root: Path):
    for directory, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        parent = Path(directory)
        directories[:] = [name for name in directories
            if not _excluded_location((parent / name).relative_to(root).as_posix(), parent / name)]
        for name in filenames:
            path = parent / name
            if not is_excluded(path.relative_to(root).as_posix(), path):
                yield path


def _excluded_location(relative: str, path: Path) -> bool:
    parts = set(path.parts)
    if parts & DEFAULT_EXCLUDES:
        return True
    return any(relative == excluded or relative.startswith(f"{excluded}/") for excluded in DEFAULT_EXCLUDES)


def is_excluded(relative: str, path: Path) -> bool:
    return (_excluded_location(relative, path) or path.suffix.lower() in BINARY_SUFFIXES
            or path.suffix.lower() in UNBOUNDED_TEXT_SUFFIXES)


def line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except UnicodeDecodeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
