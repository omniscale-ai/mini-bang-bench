"""The one benchmark document format: indented JSON split into small readable files.

A document directory holds one of:
  value.json                  the whole value as indented JSON (at most MAX_JSON_BYTES);
  index.json + text.txt       one long string as plain text;
  index.json + NNNN__<name>/  an object or list split into numbered parts: consecutive
                              small members share one part (`first_key`..`last_key` or
                              `first_item`..`last_item`), a large member is its own part.
Directories written by the retired CSV writer (fields.csv / values.csv indexes)
remain readable; nothing writes CSV any more. This file is self-contained: public
agent images copy it alone, and remote readers execute its source text.
"""
import csv
import json
from pathlib import Path
import re
import tempfile

MAX_JSON_BYTES = 16 * 1024
_NAMED_BY = ("criterion", "name", "case_id", "task_id", "event")


def _format(value, allow_nan=False):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n"


def dump(path, value, *, allow_nan=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(_format(value, allow_nan), encoding="utf-8")
    temporary.replace(path)


def write_document(directory, value, *, allow_nan=False):
    """Replace a complete document only after every member is written."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Structured document directory must not be a symlink")
    with tempfile.TemporaryDirectory(prefix=directory.name + "-writing-", dir=directory.parent) as temporary:
        staged, previous = Path(temporary) / "document", Path(temporary) / "previous"
        _write(staged, value, allow_nan)
        if directory.exists():
            directory.rename(previous)
        try:
            staged.rename(directory)
        except BaseException:
            if previous.exists():
                previous.rename(directory)
            raise


def _write(directory, value, allow_nan):
    directory.mkdir(parents=True, exist_ok=True)
    body = _format(value, allow_nan)
    if len(body.encode()) <= MAX_JSON_BYTES or not isinstance(value, (str, dict, list)):
        (directory / "value.json").write_text(body, encoding="utf-8")
        return
    if isinstance(value, str):
        (directory / "text.txt").write_text(value, encoding="utf-8")
        dump(directory / "index.json", {"kind": "text", "file": "text.txt"})
        return
    entries = []
    for number, (entry, label, child, grouped) in enumerate(_members(value, allow_nan), 1):
        name = f"{number:04d}__{_slug(label, 70)}"
        if grouped:  # Sized to fit one file; never split again.
            (directory / name).mkdir()
            (directory / name / "value.json").write_text(_format(child, allow_nan), encoding="utf-8")
        else:
            _write(directory / name, child, allow_nan)
        entries.append({**entry, "directory": name})
    dump(directory / "index.json", {"kind": "object" if isinstance(value, dict) else "list", "entries": entries})


def _members(value, allow_nan):
    """Consecutive members that fit together share one file; a larger member gets its own directory."""
    is_object = isinstance(value, dict)
    group, size = [], 4

    def chunk():
        (first, _), (last, _) = group[0], group[-1]
        if is_object:
            label = _slug(first, 70) if first == last else f"{_slug(first, 33)}-to-{_slug(last, 33)}"
            return {"first_key": first, "last_key": last}, label, dict(group), True
        return {"first_item": first, "last_item": last}, f"items-{first}-to-{last}", [c for _, c in group], True

    for key, child in value.items() if is_object else enumerate(value):
        body = _format(child, allow_nan).rstrip("\n")
        width = len(body.encode()) + 2 * body.count("\n") + 4
        width += len(json.dumps(key, ensure_ascii=False).encode()) + 2 if is_object else 0
        if group and size + width > MAX_JSON_BYTES:
            yield chunk()
            group, size = [], 4
        if width + 4 > MAX_JSON_BYTES:
            label = key if is_object else _list_label(child)
            yield {"key": key}, label, child, False
            continue
        group.append((key, child))
        size += width
    if group:
        yield chunk()


def _slug(value, limit):
    return (re.sub("[^a-zA-Z0-9_-]+", "-", str(value)).strip("-") or "entry")[:limit]


def _list_label(child):
    if isinstance(child, dict):
        return next((child[key] for key in _NAMED_BY if isinstance(child.get(key), (str, int))), "item")
    return "item"


def _member(directory, name):
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Document members must stay inside their directory")
    return directory / name


def read_document(directory):
    directory = Path(directory)
    if (directory / "value.json").exists():
        return json.loads((directory / "value.json").read_text(encoding="utf-8"))
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if index["kind"] == "text":
        return _member(directory, index["file"]).read_text(encoding="utf-8")
    if "entries" not in index:
        return _read_csv_indexed(directory, index)
    values = {} if index["kind"] == "object" else []
    for entry in index["entries"]:
        child = read_document(_member(directory, entry["directory"]))
        if "first_key" in entry:
            values.update(child)
        elif "first_item" in entry:
            values.extend(child)
        elif isinstance(values, dict):
            values[entry["key"]] = child
        else:
            values.append(child)
    return values


def _read_csv_indexed(directory, index):
    """Read a directory written by the retired CSV writer."""
    with _member(directory, index.get("file") or index["index"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if index["kind"] == "scalar-list":
        return [json.loads(row["value"]) for row in rows]
    entries = [(row["key"], read_document(_member(directory, row["directory"]))) for row in rows]
    return dict(entries) if index["kind"] == "object" else [value for _, value in entries]
