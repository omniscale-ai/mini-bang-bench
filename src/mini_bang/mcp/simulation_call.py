"""Server side of the MCP ``get_simulation`` tool, without the FastMCP transport.

``MCPBackend`` sends the session's ``RuntimeContext.to_dict()`` as the tool
argument ``runtime_context``. The server rebuilds it and executes the request
under it, exactly like a Kubernetes worker executes a job manifest. Without
that argument the server keeps its own environment context (direct tool use).
``seed_mode`` arrives only when ``SimulationRequest.to_payload`` sends a
non-default mode; a missing value means ``planner_derived``.
"""
from __future__ import annotations

from typing import Any

from mini_bang.execution import SimulationRequest
from mini_bang.execution.runtime import RuntimeContext


def run_get_simulation(
    backend: Any,
    *,
    simulator_id: str,
    saturation: Any,
    runs: int,
    extras: list[str],
    macro_params: dict[str, Any],
    micro_params: dict[str, Any],
    sample_params: dict[str, Any],
    instance_id: str | None = None,
    seed_mode: str | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> Any:
    request = SimulationRequest(
        simulator_id=simulator_id,
        saturation=saturation,
        runs=runs,
        extras=tuple(extras),
        macro_params=dict(macro_params),
        micro_params=dict(micro_params),
        sample_params=dict(sample_params),
        instance_id=instance_id,
        seed_mode="planner_derived" if seed_mode is None else seed_mode,
    )
    if runtime_context is None:
        return backend.run(request).to_transport_payload()
    if not isinstance(runtime_context, dict):
        raise ValueError("runtime_context must be a RuntimeContext.to_dict() object")
    context = RuntimeContext.from_dict(runtime_context)
    return backend.run(request, context=context).to_transport_payload()
