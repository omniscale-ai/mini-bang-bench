"""Small JSON-only control messages; arrays stay in the existing payload store."""
from __future__ import annotations

import json
import re
from typing import Any
from mini_bang.naming import readable_id

VERSION = "persistent-pool-v1"
ROLES = frozenset({"observation", "scoring", "executable", "agent"})
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_CHUNK_BYTES = 1024 * 1024
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")


class PoolUnavailable(RuntimeError):
    """An infrastructure/queue failure, never evidence of an agent mistake."""


class PoolBackpressure(PoolUnavailable):
    """The fixed queue is healthy but full; re-admit the same work identity."""


class PoolConflict(ValueError):
    """A work identity was reused with different immutable inputs."""


def validate_name(value: Any) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("Pool work names must be readable, safe and at most 200 characters")
    return value


def validate_work(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != VERSION:
        raise ValueError("Unsupported persistent pool protocol")
    validate_name(value.get("work_id"))
    if value.get("role") not in ROLES:
        raise ValueError("Unknown pool role")
    if not isinstance(value.get("payload"), dict):
        raise ValueError("Pool work requires an object payload")
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded.encode()) > MAX_MESSAGE_BYTES:
        raise ValueError("Pool control message is too large; use artifact references")
    return value


def work_name(role: str, task: str | None = None) -> str:
    return readable_id(role, task or role)


def failure(error: BaseException, *, reason="pool_worker_failed") -> dict[str, Any]:
    return {"category": "infrastructure_failed", "reason": reason,
            "type": type(error).__name__, "message": str(error)}
