"""Exercise the files actually copied into the public image, without Docker/LLMs."""
from __future__ import annotations

from pathlib import Path

from mini_bang.client import readable_files, result_files
from tests.test_readable_result_files import write_retired_csv_record

PROJECT = Path(__file__).resolve().parents[1]


def test_public_formatter_reads_legacy_without_rewriting_history(tmp_path):
    value = write_retired_csv_record(tmp_path)
    directory = tmp_path / "run_record-sections"
    original = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert readable_files.read_document(directory) == value == result_files.read_document(directory)
    assert original == {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
