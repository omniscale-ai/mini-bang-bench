from __future__ import annotations

import json
from typing import Any

from mini_bang.execution.payload_stores import PayloadRef, create_payload_store_for_ref


def payload_ref_metadata(
    *,
    artifact_id: str,
    ref: PayloadRef,
    run_id: str,
) -> dict[str, Any]:
    metadata = _public_metadata(ref.metadata)
    return {
        "artifact_id": artifact_id,
        "media_type": ref.media_type,
        "size_bytes": ref.size_bytes,
        "sha256": ref.sha256,
        "codec": ref.codec,
        "created_by": {
            "run_id": run_id,
            "request_hash": metadata.get("request_hash"),
            "request_id": metadata.get("request_id"),
        },
        "metadata": metadata,
    }


def payload_ref_bytes(ref: PayloadRef) -> bytes:
    store = create_payload_store_for_ref(ref)
    get_bytes = getattr(store, "get_bytes", None)
    if callable(get_bytes):
        return get_bytes(ref)
    payload = store.get(ref)
    if isinstance(payload, bytes):
        return payload
    if ref.codec == "json":
        return json.dumps(payload, sort_keys=True).encode("utf-8")
    raise ValueError(f"payload codec '{ref.codec}' is not byte-streamable")


def iter_payload_ref_chunks(ref: PayloadRef):
    store = create_payload_store_for_ref(ref)
    iter_bytes = getattr(store, "iter_bytes", None)
    if callable(iter_bytes):
        yield from iter_bytes(ref)
        return
    yield payload_ref_bytes(ref)


def payload_ref_json(ref: PayloadRef) -> Any:
    if ref.codec != "json":
        raise ValueError(f"payload codec '{ref.codec}' is not JSON-compatible")
    if ref.size_bytes is None or ref.size_bytes > 1024 * 1024:
        raise ValueError("JSON endpoint accepts only payloads up to 1 MiB; use download")
    return json.loads(payload_ref_bytes(ref).decode("utf-8"))


def legacy_payload_json(ref: PayloadRef, *, max_bytes: int = 1024 * 1024) -> Any:
    if ref.codec != "json":
        raise ValueError("deprecated artifact endpoint only supports JSON payloads")
    if ref.size_bytes is not None and ref.size_bytes > max_bytes:
        raise ValueError("deprecated artifact endpoint refuses large payloads")
    return payload_ref_json(ref)


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {"format", "shape", "dtype", "request_hash", "request_id"}
    return {key: value for key, value in metadata.items() if key in allowed}
