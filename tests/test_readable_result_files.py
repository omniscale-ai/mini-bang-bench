from __future__ import annotations

import json
from types import SimpleNamespace
import zipfile

import pytest

from mini_bang.client.cli import records
from mini_bang.client.cli.io import read_json, write_json
from mini_bang.client.cli.commands_run import cmd_results_export
from mini_bang.client.cli.official_payloads import encode_payload
from mini_bang.client.readable_files import MAX_JSON_BYTES, read_document, write_document
from mini_bang.execution.payloads import FileSystemPayloadStore, payload_to_transport_payload
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import TaskSubmission


def assert_ide_readable(directory):
    """Only indented JSON and plain text; no CSV, no one-line JSON objects."""
    assert not [p for p in directory.rglob("*") if p.suffix.lower() in {".csv", ".html"}]
    for path in directory.rglob("*.json"):
        assert path.stat().st_size <= MAX_JSON_BYTES
        body = path.read_text()
        if body.lstrip()[:1] in {"{", "["} and len(json.loads(body)) > 0:
            assert "\n  " in body, path


def test_large_cli_record_can_be_found_read_exported_and_zipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(records, "load_config", lambda: {"runs": {"root": str(tmp_path)}})
    record = RunRecord.from_parts(run_id="previous-internal-id", task_id="flow_lenia/describe", instance_id="event-01",
        submission=TaskSubmission(answer={"description": "Research observation " * 3000}),
        trace=[{"event": "observation", "output": "some output\n" * 3000}])
    directory = records.save_run_outputs(record, None, mode="dev-local")
    assert "flow-lenia-describe" in directory.name and "event-01" in directory.name
    assert "previous-internal-id" not in directory.name
    assert records.find_run_dir("previous-internal-id") == directory
    assert read_json(directory / "run_record.json") == record.to_dict()
    assert json.loads((directory / "run_record.json").read_text())["directory"] == "run_record-sections"
    assert any(p.parent.name.endswith("__description") for p in directory.glob("run_record-sections/**/text.txt"))
    assert_ide_readable(directory)
    assert cmd_results_export(SimpleNamespace(run_id=str(directory), format="json")) == 0
    assert json.loads(capsys.readouterr().out) == record.to_dict()
    archive = tmp_path / "result.zip"
    records.write_run_zip(directory, archive, include_artifacts=False)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(tmp_path / "restored")
    assert read_json(tmp_path / "restored/run_record.json") == record.to_dict()


def test_document_updates_do_not_read_stale_sections(tmp_path):
    directory = tmp_path / "result"
    for value in ({"text": "small"}, {"text": "large" * 20000}, [1, 2], list(range(12000))):
        write_document(directory, value)
        assert read_document(directory) == value
    assert (directory / "0001__items-0-to-2185/value.json").is_file()
    assert_ide_readable(directory)


def write_retired_csv_record(run_dir):
    """The layout the retired CSV writer produced for large run records."""
    files = {
        "run_record.json": '{"$format": "mini_bang.document.v1", "directory": "run_record-sections"}\n',
        "run_record-sections/index.json": '{"kind": "object", "index": "fields.csv"}\n',
        "run_record-sections/fields.csv": "key,directory\nrun_id,0001-run-id\ntrace,0002-trace\n"
                                          "values,0003-values\nnotes,0004-notes\n",
        "run_record-sections/0001-run-id/value.json": '"2000-01-01-10-00-00-000000--run--raf-timing-v1"\n',
        "run_record-sections/0002-trace/index.json": '{"kind": "list", "index": "fields.csv"}\n',
        "run_record-sections/0002-trace/fields.csv": "key,directory\n0,0001-item\n1,0002-item\n",
        "run_record-sections/0002-trace/0001-item/value.json": '{"event": "observation", "calls": 1}\n',
        "run_record-sections/0002-trace/0002-item/value.json": '{"event": "submission"}\n',
        "run_record-sections/0003-values/index.json": '{"kind": "scalar-list", "file": "values.csv"}\n',
        "run_record-sections/0003-values/values.csv": "value\n0.5\nnull\ntrue\n",
        "run_record-sections/0004-notes/index.json": '{"kind": "text", "file": "text.txt"}\n',
        "run_record-sections/0004-notes/text.txt": "line one\nline two\n",
    }
    for name, body in files.items():
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_text(body)
    return {"run_id": "2000-01-01-10-00-00-000000--run--raf-timing-v1",
            "trace": [{"event": "observation", "calls": 1}, {"event": "submission"}],
            "values": [0.5, None, True], "notes": "line one\nline two\n"}


def test_old_csv_indexed_run_records_stay_readable_without_rewriting(tmp_path, capsys):
    run_dir = tmp_path / "2000-01-01__10-00-00-000000-utc__raf-timing-v1__default__dev-local"
    expected = write_retired_csv_record(run_dir)
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    assert read_json(run_dir / "run_record.json") == expected
    assert read_document(run_dir / "run_record-sections") == expected
    assert cmd_results_export(SimpleNamespace(run_id=str(run_dir), format="json")) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert before == {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}


def test_document_members_cannot_escape_their_directory(tmp_path):
    (tmp_path / "document").mkdir()
    (tmp_path / "document/index.json").write_text(
        '{"kind": "object", "entries": [{"key": "secret", "directory": "../private"}]}')
    with pytest.raises(ValueError, match="inside"):
        read_document(tmp_path / "document")


def test_structured_pointer_cannot_escape_its_directory(tmp_path):
    (tmp_path / "pointer.json").write_text('{"$format":"mini_bang.document.v1","directory":"../private"}')
    with pytest.raises(ValueError, match="adjacent"):
        read_json(tmp_path / "pointer.json")


def test_readable_artifact_handles_reuse_payload_without_leaking_storage(tmp_path):
    store = FileSystemPayloadStore(tmp_path)
    first, second = store.put(b"one"), store.put(b"two")
    refs = {}
    a = encode_payload(payload_to_transport_payload(None, first), session_id="run", refs=refs)
    b = encode_payload(payload_to_transport_payload(None, second), session_id="run", refs=refs)
    assert a["artifact_id"] == "observation-0001" and b["artifact_id"] == "observation-0002"
    assert encode_payload(payload_to_transport_payload(None, first), session_id="run", refs=refs)["artifact_id"] == a["artifact_id"]
    assert str(tmp_path) not in json.dumps(a)
    assert a["payload_ref"]["sha256"] == first.sha256


def test_similarly_named_payload_fields_do_not_collide_in_manifest():
    ref = {"uri": "file:///private", "store_kind": "filesystem", "codec": "json", "media_type": "application/json"}
    manifest = records.build_artifact_manifest({"a b": ref, "a-b": ref})
    assert len(manifest) == 2 and len({row["artifact_id"] for row in manifest}) == 2
