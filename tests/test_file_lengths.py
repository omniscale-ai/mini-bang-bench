from __future__ import annotations

from mini_bang.testing.source_file_lengths import main


def test_repository_files_stay_under_line_budget() -> None:
    assert main(["--root", ".", "--max-lines", "500"]) == 0
