from __future__ import annotations

import os
import threading
from typing import Any
from urllib.parse import urlparse

from mini_bang.execution.payload_stores.base import PayloadRef
from mini_bang.execution.payload_stores.codecs import estimate_payload_size
from mini_bang.execution.payload_stores.naming import readable_payload_name


class InMemoryPayloadStore:
    """Process-local payload store for debugging and one-machine testing."""

    store_kind = "memory"
    aliases = ("in_memory", "in-memory")

    _payloads: dict[str, Any] = {}
    _lock = threading.RLock()

    def __init__(self, *, namespace: str | None = None) -> None:
        self.namespace = namespace or f"memory-{os.getpid()}"

    @classmethod
    def from_env(cls) -> "InMemoryPayloadStore":
        return cls(namespace=os.getenv("MINI_BANG_PAYLOAD_NAMESPACE"))

    def put(
        self,
        payload: Any,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PayloadRef:
        # A readable label is not unique (two requests can share it), so every
        # put gets its own time-and-counter name, as in the file stores.
        storage_key = f"{self.namespace}/{readable_payload_name(key, '')}"
        with self._lock:
            self._payloads[storage_key] = payload
        return PayloadRef(
            uri=f"memory://{storage_key}",
            store_kind=self.store_kind,
            codec="python-object",
            media_type="application/x-python-object",
            size_bytes=estimate_payload_size(payload),
            sha256=None,
            metadata=dict(metadata or {}),
        )

    def get(self, ref: PayloadRef) -> Any:
        if ref.store_kind != self.store_kind:
            raise ValueError(f"Cannot load {ref.store_kind!r} payload with memory store")
        key = _memory_key_from_uri(ref.uri)
        with self._lock:
            try:
                return self._payloads[key]
            except KeyError as exc:
                raise KeyError(f"Unknown in-memory payload ref '{ref.uri}'") from exc

    def describe(self) -> dict[str, Any]:
        return {"kind": self.store_kind, "namespace": self.namespace}


def _memory_key_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "memory":
        raise ValueError(f"Expected memory:// payload URI, got '{uri}'")
    return f"{parsed.netloc}{parsed.path}"
