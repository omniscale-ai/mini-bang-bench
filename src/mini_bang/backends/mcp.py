from __future__ import annotations

import asyncio
import atexit
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable
from uuid import uuid4

from mini_bang.backends.base import JobHandle, JobStatus, SimulationBackend
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.payloads import payload_from_transport_payload
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices

RUNTIME_CONTEXT_ARGUMENT = "runtime_context"


class MCPBackend(SimulationBackend):
    """Simulation backend that delegates to the MCP ``get_simulation`` tool.

    The session's serializable ``RuntimeContext`` crosses MCP with every call
    (tool argument ``runtime_context``) and the server executes under it, the
    same contract as Kubernetes worker manifests. Live ``RuntimeServices``
    cannot cross a transport; ``check_runtime`` rejects them at session setup,
    before any budget is spent.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_workers: int | None = None,
        client_count: int | None = None,
    ) -> None:
        if client is None:
            from mini_bang.mcp.client import MCPClient

            count = max(1, int(client_count or 1))
            self._clients = [MCPClient() for _ in range(count)]
            self._owns_clients = True
        else:
            self._clients = [client]
            self._owns_clients = False
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._next_client = 0
        self._client_lock = threading.Lock()
        self._closed = False
        self._jobs: dict[str, SimulationResult | list[SimulationResult]] = {}
        atexit.register(self.close)

    @classmethod
    def from_env(cls) -> "MCPBackend":
        workers_raw = os.getenv("MINI_BANG_MCP_BATCH_WORKERS")
        clients_raw = os.getenv("MINI_BANG_MCP_CLIENTS")
        workers = int(workers_raw) if workers_raw else None
        clients = int(clients_raw) if clients_raw else (workers or 8)
        return cls(max_workers=workers, client_count=clients)

    def check_runtime(
        self,
        *,
        context: RuntimeContext | None,
        services: RuntimeServices | None,
    ) -> None:
        """Reject, at session setup, runtime that cannot cross the MCP transport."""
        _context_argument(context=context, services=services)

    def run(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        sent_context = _context_argument(context=context, services=services)
        return self._run_with_client(request, self._pick_client(), sent_context)

    def _run_with_client(
        self,
        request: SimulationRequest,
        client: Any,
        sent_context: dict[str, Any] | None = None,
    ) -> SimulationResult:
        payload = request.to_payload()
        payload.pop("simulator_id", None)
        if sent_context is not None:
            payload[RUNTIME_CONTEXT_ARGUMENT] = sent_context
        response = client.call_get_simulation(request.simulator_id, payload)
        inline_payload, payload_ref = payload_from_transport_payload(response)
        return SimulationResult(
            request=request,
            payload=inline_payload,
            request_hash=request.request_hash,
            payload_ref=payload_ref,
            backend_metadata=self.describe(),
            versions={"transport": "mcp"},
        )

    def run_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        sent_context = _context_argument(context=context, services=services)
        request_list = list(requests)
        if not request_list:
            return []
        futures = [
            self._pool.submit(self._run_with_client, request, self._client_for_index(index), sent_context)
            for index, request in enumerate(request_list)
        ]
        return [future.result() for future in futures]

    async def arun(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        sent_context = _context_argument(context=context, services=services)
        loop = asyncio.get_running_loop()
        client = self._pick_client()
        return await loop.run_in_executor(self._pool, self._run_with_client, request, client, sent_context)

    async def arun_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        sent_context = _context_argument(context=context, services=services)
        request_list = list(requests)
        if not request_list:
            return []
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                self._pool,
                self._run_with_client,
                request,
                self._client_for_index(index),
                sent_context,
            )
            for index, request in enumerate(request_list)
        ]
        return list(await asyncio.gather(*tasks))

    def submit(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        _context_argument(context=context, services=services)
        job_id = uuid4().hex
        self._jobs[job_id] = self.run(request, context=context, services=services)
        return JobHandle(
            backend="mcp",
            job_id=job_id,
            run_id=context.run_id if context else None,
            task_id=context.task_id if context else None,
            instance_id=context.instance_id if context else None,
            metadata={"batch_size": 1, "async_shim": "immediate"},
        )

    def submit_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        _context_argument(context=context, services=services)
        request_list = list(requests)
        job_id = uuid4().hex
        self._jobs[job_id] = self.run_batch(request_list, context=context, services=services)
        return JobHandle(
            backend="mcp",
            job_id=job_id,
            run_id=context.run_id if context else None,
            task_id=context.task_id if context else None,
            instance_id=context.instance_id if context else None,
            metadata={"batch_size": len(request_list), "async_shim": "immediate"},
        )

    def poll(self, handle: JobHandle) -> JobStatus:
        if handle.job_id in self._jobs:
            return JobStatus(state="succeeded")
        return JobStatus(state="unknown", detail="No such MCP shim job")

    def fetch(self, handle: JobHandle) -> SimulationResult | list[SimulationResult]:
        try:
            return self._jobs[handle.job_id]
        except KeyError as exc:
            raise KeyError(f"Unknown MCP shim job '{handle.job_id}'") from exc

    def describe(self) -> dict:
        client = self._clients[0]
        describe = getattr(client, "describe", None)
        mcp_info = describe() if callable(describe) else {}
        description = {"backend": "mcp", "mcp": mcp_info}
        if len(self._clients) > 1:
            description["client_count"] = len(self._clients)
        return description

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        if self._owns_clients:
            for client in self._clients:
                close = getattr(client, "close", None)
                if callable(close):
                    close()

    def _pick_client(self) -> Any:
        with self._client_lock:
            client = self._clients[self._next_client % len(self._clients)]
            self._next_client += 1
        return client

    def _client_for_index(self, index: int) -> Any:
        return self._clients[index % len(self._clients)]


def _context_argument(
    *,
    context: RuntimeContext | None,
    services: RuntimeServices | None,
) -> dict[str, Any] | None:
    """Return the JSON context sent with the tool call, or raise if it cannot cross MCP."""
    if services is not None:
        raise ValueError(
            "MCPBackend cannot send live RuntimeServices over MCP; pass a serializable "
            "RuntimeContext or use LocalBackend/ExecutionServices."
        )
    if context is None:
        return None
    try:
        sent = context.to_dict()
        json.dumps(sent, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MCPBackend can send only a JSON RuntimeContext over MCP: {exc}") from exc
    return sent
