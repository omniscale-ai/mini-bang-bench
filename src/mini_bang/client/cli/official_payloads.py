from __future__ import annotations

from mini_bang.client.cli.io import json_ready
from mini_bang.execution.payloads import PAYLOAD_REF_SENTINEL, PayloadHandle, PayloadRef
from threading import Lock

_REGISTRY_LOCK = Lock()


def encode_payload(value, *, session_id, refs):
    """Register only returned handles and hide their internal storage locations."""
    ref = getattr(value, "ref", None)
    if isinstance(value,PayloadHandle) and ref is None and value.is_materialized:
        return encode_payload(value.payload,session_id=session_id,refs=refs)
    if ref is None and isinstance(value, dict) and value.get(PAYLOAD_REF_SENTINEL):
        ref = PayloadRef.from_dict(value["ref"])
    if ref is not None:
        with _REGISTRY_LOCK:
            artifact_id = next((name for name, existing in refs.items()
                if existing.uri == ref.uri and existing.sha256 == ref.sha256 and existing.codec == ref.codec), None)
            if artifact_id is None:
                artifact_id = f"observation-{len(refs) + 1:04d}"
                refs[artifact_id] = ref
        return {"session_id": session_id, "artifact_id": artifact_id,
                "payload_ref": {"uri": f"artifact://{artifact_id}", "store_kind": ref.store_kind,
                                "codec": ref.codec, "media_type": ref.media_type,
                                "size_bytes": ref.size_bytes, "sha256": ref.sha256}}
    if isinstance(value, dict):
        return {key: encode_payload(item, session_id=session_id, refs=refs) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_payload(item, session_id=session_id, refs=refs) for item in value]
    return json_ready(value)


def public_result(result):
    """Agent-facing status only; hidden scoring traces belong to the organizer."""
    output = {"status": result["status"]}
    record = result.get("run_record")
    if record:
        output["evaluation_completed"] = True
    if result.get("status") == "failed":
        output["error"] = "Evaluation failed; organizer can inspect the private run record"
    return output
