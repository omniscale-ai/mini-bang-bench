from __future__ import annotations

from typing import Any

from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.payloads import payload_from_transport_payload


def request_to_transport(request: SimulationRequest) -> dict[str, Any]:
    """Serialize a request for job manifests."""
    return request.to_payload()


def request_from_transport(payload: dict[str, Any]) -> SimulationRequest:
    """Deserialize a request from a job manifest."""
    return SimulationRequest.from_payload(
        simulator_id=str(payload["simulator_id"]),
        payload=payload,
    )


def result_to_transport(result: SimulationResult) -> dict[str, Any]:
    """Serialize a result without materializing any payload refs."""
    return {
        "request": request_to_transport(result.request),
        "request_hash": result.request_hash,
        "payload": result.to_transport_payload(),
        "backend_metadata": result.backend_metadata,
        "resource_usage": result.resource_usage,
        "cache_hit": result.cache_hit,
        "versions": result.versions,
        "trace": result.trace,
    }


def result_from_transport(payload: dict[str, Any]) -> SimulationResult:
    """Deserialize a transport result envelope."""
    inline_payload, payload_ref = payload_from_transport_payload(payload.get("payload"))
    request = request_from_transport(dict(payload["request"]))
    return SimulationResult(
        request=request,
        payload=inline_payload,
        request_hash=str(payload.get("request_hash") or request.request_hash),
        payload_ref=payload_ref,
        backend_metadata=dict(payload.get("backend_metadata") or {}),
        resource_usage=dict(payload.get("resource_usage") or {}),
        cache_hit=bool(payload.get("cache_hit", False)),
        versions=dict(payload.get("versions") or {}),
        trace=list(payload.get("trace") or []),
    )
