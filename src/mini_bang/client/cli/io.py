from __future__ import annotations

import datetime as dt
import json
import re
import socket
from pathlib import Path
from typing import Any

from mini_bang.client.paths import CACHE_ROOT
from mini_bang.client.cli.config import load_config, nested_get
from mini_bang.client.cli.errors import CliError, EXIT_USAGE
from mini_bang.framework.records import to_serializable
from mini_bang.client.readable_files import MAX_JSON_BYTES, read_document, write_document


def emit(payload: Any, output_format: str, text_renderer: Any) -> None:
    if output_format == "json":
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    else:
        print(text_renderer())

def normalize_table_format(value: str) -> str:
    return "text" if value == "table" else value

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if isinstance(value, dict) and value.get("$format") == "mini_bang.document.v1":
        relative = Path(value["directory"])
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {".", ".."}:
            raise ValueError("Structured document must be in an adjacent directory")
        return read_document(path.parent / relative)
    return value

def write_json(path: Path, value: Any) -> None:
    """Indented JSON; a large value goes to `<stem>-sections/` and the file points there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = json_ready(value)
    body = json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if len(body.encode()) > MAX_JSON_BYTES:
        directory = path.with_name(path.stem + "-sections")
        write_document(directory, ready, allow_nan=True)
        body = json.dumps({"$format": "mini_bang.document.v1", "directory": directory.name}, indent=2) + "\n"
    path.write_text(body, encoding="utf-8")

def write_trace_jsonl(path: Path, trace: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in trace:
            fh.write(json.dumps(json_ready(event), sort_keys=True))
            fh.write("\n")

def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"base64": value.hex(), "encoding": "hex"}
    return to_serializable(value)

def looks_like_raw_file_ref(value: str) -> bool:
    return value.startswith("file://") or value.startswith("/") or value.startswith("./") or value.startswith("../")

def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")

def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return cleaned or "unnamed"

def get_cache_dir() -> Path:
    return Path(nested_get(load_config(), "cache.root", f"./{CACHE_ROOT.as_posix()}"))

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def parse_duration_seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+)([smhd]?)", value.strip())
    if not match:
        raise CliError(f"invalid duration: {value}", EXIT_USAGE)
    amount = int(match.group(1))
    unit = match.group(2) or "s"
    return amount * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]

def close_backend(backend: Any) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()
