from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


def serialize_payload(
    payload: Any,
    *,
    allow_pickle_payloads: bool = False,
) -> tuple[bytes, str, str, str]:
    """Serialize a Python payload into bytes plus codec metadata."""
    if isinstance(payload, bytes):
        return payload, "bytes", "application/octet-stream", ".bin"
    if isinstance(payload, np.ndarray) and not payload.dtype.hasobject:
        buffer = io.BytesIO()
        np.save(buffer, payload, allow_pickle=False)
        return buffer.getvalue(), "npy", "application/x-npy", ".npy"
    if _json_compatible(payload):
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return body, "json", "application/json", ".json"
    if not allow_pickle_payloads:
        raise ValueError(
            "payload is not JSON/bytes/file compatible; trusted pickle is disabled"
        )
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return body, "trusted_pickle", "application/vnd.mini-bang.trusted-pickle", ".pickle"


def deserialize_payload(
    body: bytes,
    *,
    codec: str,
    allow_pickle_payloads: bool = False,
) -> Any:
    """Deserialize bytes produced by ``serialize_payload``."""
    if codec == "bytes":
        return body
    if codec == "json":
        return json.loads(body.decode("utf-8"))
    if codec == "npy":
        return np.load(io.BytesIO(body), allow_pickle=False)
    if codec in {"pickle", "trusted_pickle"}:
        if not allow_pickle_payloads:
            raise ValueError("trusted pickle payload deserialization is disabled")
        return pickle.loads(body)
    raise ValueError(f"Unsupported payload codec '{codec}'")


def estimate_payload_size(payload: Any) -> int:
    """Return an approximate serialized payload size in bytes."""
    if isinstance(payload, os.PathLike):
        try:
            return Path(payload).stat().st_size
        except OSError:
            return 0
    try:
        body, _, _, _ = serialize_payload(payload, allow_pickle_payloads=True)
    except Exception:
        return len(repr(payload).encode("utf-8", errors="replace"))
    return len(body)


def verify_payload_file(path: Path, *, uri: str, size_bytes: int | None, sha256: str | None) -> None:
    """Verify a stored payload file against optional ref metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Payload ref points to a missing file: {path}")
    if size_bytes is not None and path.stat().st_size != size_bytes:
        raise ValueError(f"Payload size mismatch for '{uri}'")
    if sha256 is None:
        return
    if hash_file(path) != sha256:
        raise ValueError(f"Payload digest mismatch for '{uri}'")


def verify_payload_bytes(body: bytes, *, uri: str, size_bytes: int | None, sha256: str | None) -> None:
    """Verify stored payload bytes against optional ref metadata."""
    if size_bytes is not None and len(body) != size_bytes:
        raise ValueError(f"Payload size mismatch for '{uri}'")
    if sha256 is not None and hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError(f"Payload digest mismatch for '{uri}'")


def hash_file(path: Path) -> str:
    """Return a sha256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_key(key: str) -> str:
    """Return a storage-safe object/file name component."""
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in key)
    return cleaned.strip("._") or uuid4().hex


def _json_compatible(value: Any) -> bool:
    if dataclasses.is_dataclass(value):
        return False
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return False
    return True
