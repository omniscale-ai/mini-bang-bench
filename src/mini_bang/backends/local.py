from __future__ import annotations

import asyncio
import os
from uuid import uuid4
from typing import Iterable

from mini_bang.artifacts import ArtifactResolver
from mini_bang.backends.base import JobHandle, JobStatus, SimulationBackend
from mini_bang.execution.engines import ProcessPoolEngine, SequentialEngine
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.payloads import create_payload_storage_from_env
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.execution.services import ExecutionServices


class LocalBackend(SimulationBackend):
    """In-process backend that executes simulations through a local executor."""

    def __init__(
        self,
        executor: SimulationExecutor | None = None,
        *,
        artifact_resolver: ArtifactResolver | None = None,
        runtime_context: RuntimeContext | None = None,
        runtime_services: RuntimeServices | None = None,
    ) -> None:
        self._executor = executor or SimulationExecutor()
        self._runtime_context = runtime_context
        self._runtime_services = runtime_services
        if artifact_resolver is not None:
            self.set_artifact_resolver(artifact_resolver)
        self._jobs: dict[str, SimulationResult | list[SimulationResult]] = {}

    @classmethod
    def from_env(cls) -> "LocalBackend":
        """Create a local backend from ``MINI_BANG_EXECUTION_ENGINE`` settings."""
        engine_name = os.getenv("MINI_BANG_EXECUTION_ENGINE", "sequential").lower()
        reuse_raw = os.getenv("MINI_BANG_REUSE_STRATEGY")
        if reuse_raw is None:
            reuse_raw = "grouped" if engine_name in {"process_grouped"} else "none"
        reuse_strategy = "grouped" if reuse_raw.lower() in {"grouped", "reuse", "true", "1"} else "none"

        if engine_name in {"process", "process_pool", "multiprocessing", "process_grouped"}:
            workers_raw = os.getenv("MINI_BANG_WORKERS")
            workers = int(workers_raw) if workers_raw else None
            engine = ProcessPoolEngine(max_workers=workers, reuse_strategy=reuse_strategy)
        elif engine_name in {"sequential", "local", "local_grouped"}:
            if engine_name == "local_grouped":
                reuse_strategy = "grouped"
            engine = SequentialEngine(reuse_strategy=reuse_strategy)
        else:
            raise ValueError(f"Unsupported MINI_BANG_EXECUTION_ENGINE='{engine_name}'")
        return cls(
            executor=SimulationExecutor(
                engine=engine,
                payload_storage=create_payload_storage_from_env(),
                context=RuntimeContext.from_env(),
            )
        )

    def run(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        return self._executor.run(
            request,
            context=context or self._runtime_context,
            services=services or self._runtime_services,
        )

    def run_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        return self._executor.run_batch(
            requests,
            context=context or self._runtime_context,
            services=services or self._runtime_services,
        )

    def set_artifact_resolver(self, artifact_resolver: ArtifactResolver) -> None:
        """Set dev/single-session default services. Do not use in official multi-session paths."""
        self._runtime_services = RuntimeServices(artifact_resolver=artifact_resolver)

    def set_runtime_context(self, context: RuntimeContext) -> None:
        """Set a dev/single-session default context. Official paths pass context per call."""
        self._runtime_context = context
        self._runtime_services = None

    def set_runtime_services(self, services: RuntimeServices) -> None:
        """Set dev/single-session default services. Official paths pass context per call."""
        self._runtime_services = services

    def execution_services(self) -> ExecutionServices:
        """Return the shared executor substrate without task API policy."""
        return ExecutionServices(executor=self._executor)

    async def arun(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        return await asyncio.to_thread(self.run, request, context=context, services=services)

    async def arun_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        request_list = list(requests)
        return await asyncio.to_thread(
            self.run_batch,
            request_list,
            context=context,
            services=services,
        )

    def submit(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        """Submit a local job-oriented request and keep the completed result."""
        job_id = uuid4().hex
        effective_context = context or self._runtime_context
        self._jobs[job_id] = self.run(request, context=context, services=services)
        return _handle("local", job_id, effective_context, batch_size=1)

    def submit_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        """Submit a batch of local job-oriented requests."""
        request_list = list(requests)
        job_id = uuid4().hex
        effective_context = context or self._runtime_context
        self._jobs[job_id] = self.run_batch(
            request_list,
            context=context,
            services=services,
        )
        return _handle("local", job_id, effective_context, batch_size=len(request_list))

    def poll(self, handle: JobHandle) -> JobStatus:
        """Poll an immediately executed local job handle."""
        if handle.job_id in self._jobs:
            return JobStatus(state="succeeded")
        return JobStatus(state="unknown", detail="No such local job")

    def fetch(self, handle: JobHandle) -> SimulationResult | list[SimulationResult]:
        """Fetch a completed local job-oriented result."""
        try:
            return self._jobs[handle.job_id]
        except KeyError as exc:
            raise KeyError(f"Unknown local job '{handle.job_id}'") from exc

    def describe(self) -> dict:
        return {"backend": "local", **self._executor.describe()}

    def close(self) -> None:
        close = getattr(self._executor, "close", None)
        if callable(close):
            close()


def _handle(
    backend: str,
    job_id: str,
    context: RuntimeContext | None,
    *,
    batch_size: int,
) -> JobHandle:
    return JobHandle(
        backend=backend,
        job_id=job_id,
        run_id=context.run_id if context else None,
        task_id=context.task_id if context else None,
        instance_id=context.instance_id if context else None,
        metadata={"batch_size": batch_size},
    )
