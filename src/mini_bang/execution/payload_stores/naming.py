"""Readable names for stored runtime payloads.

A payload is stored as `<UTC date>/<UTC time>--<label>--<counter><suffix>`, e.g.
`YYYY-MM-DD/YYYY-MM-DD-hh-mm-ss-ffffff--my-task--my-simulator--saturation-4--runs-1--macro-seed-2001--0001.json`.
The label says what was simulated (task, simulator, request size and seeds);
"--" separates its parts. The counter keeps names unique inside a process, and
stores also check that a name is free before they write. The sha256 is never
part of the name: it stays in the PayloadRef and in object metadata and is
verified on every read. Refs written before this naming carry their stored
paths and resolve as before.
"""
from __future__ import annotations

import itertools
import re
import threading
from datetime import datetime, timezone
from pathlib import PurePosixPath

from mini_bang.naming import shorten_label

MAX_LABEL_LENGTH = 96
DEFAULT_LABEL = "payload"

_COUNTER = itertools.count(1)
_COUNTER_LOCK = threading.Lock()


def payload_label(*parts: object) -> str:
    """Join readable parts with "--"; each part becomes lowercase words joined by "-"."""
    slugs = [_slug(part) for part in parts if part is not None]
    return clean_payload_label("--".join(slug for slug in slugs if slug))


def clean_payload_label(label: str | None) -> str:
    """Storage-safe label that keeps "-" words, "--" part separators and decimal dots."""
    text = re.sub(r"[^a-z0-9.-]+", "-", str(label or "").lower())
    text = re.sub(r"-{3,}", "--", re.sub(r"\.{2,}", ".", text)).strip("-.")
    return shorten_label(text, MAX_LABEL_LENGTH) if text else DEFAULT_LABEL


def readable_payload_name(
    label: str | None,
    suffix: str,
    *,
    now: datetime | None = None,
    counter: int | None = None,
) -> PurePosixPath:
    """Return `<date>/<time>--<label>--<counter><suffix>` relative to a store root."""
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    number = next_payload_counter() if counter is None else counter
    stamp = moment.strftime("%Y-%m-%d-%H-%M-%S-%f")
    name = f"{stamp}--{clean_payload_label(label)}--{number:04d}{suffix}"
    return PurePosixPath(moment.strftime("%Y-%m-%d")) / name


def payload_suffix(default: str, codec: str, metadata: dict | None) -> str:
    """Raw bytes declared as a file format (``metadata["format"]``, e.g. npz) keep its extension."""
    declared = str((metadata or {}).get("format") or "").lower()
    if codec == "bytes" and re.fullmatch(r"[a-z0-9]{1,8}", declared):
        return f".{declared}"
    return default


def next_payload_counter() -> int:
    with _COUNTER_LOCK:
        return next(_COUNTER)


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9.]+", "-", str(value).lower())
    return re.sub(r"-*\.-*", ".", text).strip("-.")
