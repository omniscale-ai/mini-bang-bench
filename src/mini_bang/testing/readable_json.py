"""Indented JSON for reading in an IDE: nested objects one key per line, scalar lists and
small flat objects inline, long scalar lists wrapped at about 110 characters per line."""
from __future__ import annotations

import json
from pathlib import Path

WIDTH = 110


def _scalar(value):
    return value is None or isinstance(value, (bool, int, float, str))


def _render(value, indent):
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        flat = json.dumps(value, ensure_ascii=False)
        if all(_scalar(v) for v in value.values()) and len(flat) + len(pad) <= WIDTH:
            return flat
        items = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {_render(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if all(_scalar(v) for v in value):
            parts = [json.dumps(v, ensure_ascii=False) for v in value]
            if sum(len(p) + 2 for p in parts) + len(pad) <= WIDTH:
                return "[" + ", ".join(parts) + "]"
            lines, line = [], ""
            for part in parts:
                if line and len(inner) + len(line) + len(part) + 2 > WIDTH:
                    lines.append(inner + line.rstrip())
                    line = ""
                line += part + ", "
            lines.append(inner + line.rstrip().rstrip(","))
            return "[\n" + "\n".join(lines) + "\n" + pad + "]"
        return "[\n" + ",\n".join(inner + _render(v, indent + 1) for v in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False)


def readable_json(value) -> str:
    text = _render(value, 0) + "\n"
    json.loads(text)  # round-trip check
    return text


def write_readable_json(path: Path, value) -> None:
    text = readable_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
