from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_bang.execution.payload_stores import (
    FileSystemPayloadStore,
    InMemoryPayloadStore,
    PayloadRef,
    PayloadStore,
    S3PayloadStore,
    create_payload_store_for_ref,
    create_payload_store_from_env,
    register_payload_store,
)
from mini_bang.execution.payload_stores.codecs import (
    deserialize_payload,
    estimate_payload_size,
    serialize_payload,
)
from mini_bang.execution.payload_stores.registry import payload_mode_and_store_from_env

PAYLOAD_REF_SENTINEL = "__mini_bang_payload_ref__"
DEFAULT_AUTO_INLINE_THRESHOLD_BYTES = 1024 * 1024


@dataclass(frozen=True)
class PayloadStorageConfig:
    """Controls whether payloads stay inline or become references."""

    mode: str = "inline"
    store: PayloadStore | None = None
    inline_threshold_bytes: int | None = None

    @classmethod
    def inline(cls) -> "PayloadStorageConfig":
        return cls(mode="inline")

    @classmethod
    def reference(cls, store: PayloadStore, *, mode: str | None = None) -> "PayloadStorageConfig":
        return cls(mode=mode or _store_kind(store), store=store)

    def externalize(
        self,
        payload: Any,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, PayloadRef | None]:
        """Return ``(inline_payload, payload_ref)`` for a result payload."""
        normalized_mode = self.mode.lower()
        if normalized_mode in {"inline", "none", "off"}:
            return payload, None
        if normalized_mode == "auto":
            threshold = (
                self.inline_threshold_bytes
                if self.inline_threshold_bytes is not None
                else DEFAULT_AUTO_INLINE_THRESHOLD_BYTES
            )
            if estimate_payload_size(payload) <= threshold:
                return payload, None
        if self.store is None:
            raise ValueError(f"Payload storage mode '{self.mode}' requires a store")
        return None, self.store.put(payload, key=key, metadata=metadata)

    def describe(self) -> dict[str, Any]:
        description: dict[str, Any] = {"mode": self.mode}
        if self.inline_threshold_bytes is not None:
            description["inline_threshold_bytes"] = self.inline_threshold_bytes
        if self.store is not None:
            description["store"] = self.store.describe()
        return description


def create_payload_storage_from_env() -> PayloadStorageConfig:
    """Create payload storage config from ``MINI_BANG_PAYLOAD_*`` env vars."""
    mode, store_kind = payload_mode_and_store_from_env()
    if store_kind is None:
        return PayloadStorageConfig.inline()
    threshold_raw = os.getenv("MINI_BANG_PAYLOAD_INLINE_THRESHOLD")
    threshold = int(threshold_raw) if threshold_raw else None
    return PayloadStorageConfig(
        mode=mode,
        store=create_payload_store_from_env(store_kind),
        inline_threshold_bytes=threshold,
    )


def create_payload_storage_from_context(context) -> PayloadStorageConfig:
    """Create whole-result payload storage from ``RuntimeContext.payload``."""
    payload_cfg = dict(getattr(context, "payload", {}) or {})
    kind = payload_cfg.get("store_kind") or payload_cfg.get("kind") or payload_cfg.get("mode")
    if not kind or str(kind).lower() in {"inline", "none", "off"}:
        return PayloadStorageConfig.inline()
    mode = str(payload_cfg.get("mode") or kind).lower().replace("-", "_")
    threshold = payload_cfg.get("inline_threshold_bytes") or payload_cfg.get("inline_threshold")
    store_kind = payload_cfg.get("store_kind") or payload_cfg.get("kind") or kind
    store = create_payload_store_for_context(store_kind, payload_cfg)
    return PayloadStorageConfig(
        mode=mode,
        store=store,
        inline_threshold_bytes=int(threshold) if threshold is not None else None,
    )


def create_payload_store_for_context(kind: str, payload_cfg: dict[str, Any]) -> PayloadStore:
    normalized = str(kind).lower().replace("-", "_")
    config = dict(payload_cfg.get("config") or payload_cfg)
    if normalized in {"filesystem", "fs", "file", "k8s", "pvc", "filesystem_pvc"}:
        return FileSystemPayloadStore(
            config.get("root") or config.get("dir") or config.get("path"),
            allow_pickle_payloads=bool(config.get("allow_pickle_payloads", False)),
        )
    if normalized in {"memory", "in_memory"}:
        return InMemoryPayloadStore(namespace=config.get("namespace"))
    if normalized in {"s3", "minio", "aws_s3", "object", "object_store"}:
        return S3PayloadStore(
            bucket=str(config.get("bucket") or ""),
            prefix=str(config.get("prefix") or ""),
            endpoint_url=config.get("endpoint_url"),
            region_name=config.get("region_name") or config.get("region"),
            download_dir=config.get("download_dir"),
            allow_pickle_payloads=bool(config.get("allow_pickle_payloads", False)),
        )
    return create_payload_store_from_env(str(kind))


def payload_to_transport_payload(payload: Any, payload_ref: PayloadRef | None) -> Any:
    """Convert a payload/result pair into the value sent over a transport."""
    if payload_ref is None:
        return payload
    return {
        PAYLOAD_REF_SENTINEL: True,
        "ref": payload_ref.to_dict(),
    }


def payload_from_transport_payload(value: Any) -> tuple[Any, PayloadRef | None]:
    """Decode a legacy inline payload or a reference envelope."""
    if (
        isinstance(value, dict)
        and value.get(PAYLOAD_REF_SENTINEL) is True
        and isinstance(value.get("ref"), dict)
    ):
        return None, PayloadRef.from_dict(value["ref"])
    return value, None


def load_payload_ref(ref: PayloadRef) -> Any:
    """Load a payload reference using the registered store implementation."""
    return create_payload_store_for_ref(ref).get(ref)


class PayloadHandle:
    """Lazy agent-facing handle for a simulation payload."""

    def __init__(
        self,
        *,
        payload: Any = None,
        payload_ref: PayloadRef | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._payload = payload
        self.ref = payload_ref
        self.metadata = dict(metadata or {})
        self._loaded = payload_ref is None

    @property
    def is_materialized(self) -> bool:
        """Return whether the payload body is already loaded in this process."""
        return self._loaded

    @property
    def payload(self) -> Any:
        """Return loaded payload, or raise when this handle only has a ref."""
        if not self._loaded:
            raise RuntimeError("Payload is not materialized; call load() first")
        return self._payload

    def load(self) -> Any:
        """Load and cache the payload body when this handle carries a ref."""
        if not self._loaded:
            if self.ref is None:
                raise RuntimeError("Payload handle has neither payload nor ref")
            self._payload = load_payload_ref(self.ref)
            self._loaded = True
        return self._payload

    def as_path(self) -> Path:
        """Load the payload and return it as a local file path."""
        payload = self.load()
        if isinstance(payload, Path):
            return payload
        if isinstance(payload, os.PathLike):
            return Path(payload)
        raise TypeError(f"Payload is not path-like: {type(payload).__name__}")

    def to_metadata(self) -> dict[str, Any]:
        """Return a small serializable description of this handle."""
        data = dict(self.metadata)
        data["materialized"] = self._loaded
        if self.ref is not None:
            data["payload_ref"] = self.ref.to_dict()
        return data

    def __repr__(self) -> str:
        ref_uri = self.ref.uri if self.ref is not None else None
        return (
            f"PayloadHandle(materialized={self._loaded}, "
            f"ref_uri={ref_uri!r}, metadata_keys={sorted(self.metadata)})"
        )


def _store_kind(store: PayloadStore) -> str:
    return str(getattr(store, "store_kind", "reference"))


__all__ = [
    "DEFAULT_AUTO_INLINE_THRESHOLD_BYTES",
    "PAYLOAD_REF_SENTINEL",
    "FileSystemPayloadStore",
    "InMemoryPayloadStore",
    "PayloadRef",
    "PayloadHandle",
    "PayloadStorageConfig",
    "PayloadStore",
    "S3PayloadStore",
    "create_payload_storage_from_env",
    "create_payload_storage_from_context",
    "create_payload_store_for_context",
    "deserialize_payload",
    "estimate_payload_size",
    "load_payload_ref",
    "payload_from_transport_payload",
    "payload_to_transport_payload",
    "register_payload_store",
    "serialize_payload",
]
