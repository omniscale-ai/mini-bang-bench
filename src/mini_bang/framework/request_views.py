"""Agent-visible and backend views of one admitted simulation request.

A task policy may resolve values the agent must not see (for example the hidden
macro seed behind a public network name) by defining two optional methods::

    backend_request(*, session, request) -> SimulationRequest
    public_payload(*, session, request, payload) -> payload

``RunSession`` validates, budgets, hashes and traces the public request; only the
backend receives the resolved one. A result of a resolved request is returned
inline after ``public_payload`` rewrote it. Payload refs and storage keys, the
executor's job trace and backend errors (they name the resolved request) stay
private, so the session trace, the official HTTP API and its artifact endpoints
carry only public values. Policies without these methods are unaffected.
"""
from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from typing import Any

from mini_bang.execution.payloads import PayloadHandle

RESOLVED = "hidden_values_resolved_by_policy"


def backend_request(session: Any, request: Any) -> Any:
    resolve = getattr(session.policy, "backend_request", None)
    return request if resolve is None else resolve(session=session, request=request)


def resolved(context: dict[str, Any]) -> bool:
    return context.get("backend_request", context["request"]) is not context["request"]


@contextmanager
def private_backend_errors(contexts: list[dict[str, Any]]):
    """Backend errors of a batch with resolved requests may quote them; withhold the text."""
    try:
        yield
    except Exception:
        if any(resolved(context) for context in contexts):
            raise RuntimeError("Simulation failed for a request with server-resolved values; "
                               "the organizer can inspect private diagnostics") from None
        raise


def _scalars(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items()
            if isinstance(value, (str, int, float, bool)) or value is None}


def record_resolved_result(session: Any, context: dict[str, Any], result: Any, materialize: bool) -> Any:
    """Trace and return a resolved result with public values only."""
    payload = session.policy.public_payload(session=session, request=context["request"],
                                            payload=result.materialize_payload())
    backend = _scalars(getattr(result, "backend_metadata", {}) or {})
    resources = {key: value for key, value in (getattr(result, "resource_usage", {}) or {}).items()
                 if isinstance(value, (int, float)) and not isinstance(value, bool)}
    entry = {
        "event": "simulation_request", "method": context["method_name"],
        "request_hash": context["request_hash"], "request": context["request_payload"],
        "timestamp_start": context["started_at"], "timestamp_end": time.time(),
        "wall_seconds": time.perf_counter() - context["started"], "backend_metadata": backend,
        "budget_before": context["budget_before"], "budget_after": context["budget_after"],
        "resource_delta": dict(resources),
        "response_metadata": {"cache_hit": bool(getattr(result, "cache_hit", False)),
                              "materialized": True, RESOLVED: True,
                              "versions": copy.deepcopy(getattr(result, "versions", {}) or {})},
        "execution_trace": [],
    }
    with session._lock:
        for key, value in resources.items():
            session.resource_usage[key] = session.resource_usage.get(key, 0) + value
        session.backend_metadata.update(backend)
        session.trace.append(entry)
    if materialize:
        return payload
    return PayloadHandle(payload=payload, metadata={"request_hash": context["request_hash"], RESOLVED: True})
