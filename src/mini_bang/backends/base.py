from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices


class SimulationBackend(Protocol):
    """Backend abstraction for simulator execution transport."""

    def run(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        """Execute one simulation request and return its result."""
        ...

    def run_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        """Execute multiple simulation requests."""
        ...

    def describe(self) -> dict:
        """Return machine-readable backend metadata."""
        ...


def check_backend_runtime(
    backend: Any,
    *,
    context: RuntimeContext | None,
    services: RuntimeServices | None,
) -> None:
    """Session-setup check, before any budget is spent.

    A backend that cannot honor some runtime (for example live services over a
    transport) defines ``check_runtime(context=..., services=...)`` and raises
    there, so a session never starts only to fail every paid call.
    """
    check = getattr(backend, "check_runtime", None)
    if callable(check):
        check(context=context, services=services)


class AsyncSimulationBackend(SimulationBackend, Protocol):
    """Optional async extension for backends that can overlap remote work."""

    async def arun(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        """Execute one simulation request asynchronously."""
        ...

    async def arun_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        """Execute multiple simulation requests asynchronously."""
        ...


@dataclass(frozen=True)
class JobHandle:
    """Opaque handle for job-oriented backend extensions."""

    backend: str
    job_id: str
    run_id: str | None = None
    task_id: str | None = None
    instance_id: str | None = None
    result_ref: dict[str, Any] | None = None
    result_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositeJobHandle(JobHandle):
    """Handle for a batch represented by multiple child backend jobs."""

    child_handles: tuple[JobHandle, ...] = ()
    child_indexes: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class JobStatus:
    """Status returned by asynchronous job backends."""

    state: str
    detail: str | None = None


class JobBackend(Protocol):
    """Optional asynchronous extension for remote or cluster backends."""

    def submit(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        """Submit one request for asynchronous execution."""
        ...

    def submit_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        """Submit a batch of asynchronous requests."""
        ...

    def poll(self, handle: JobHandle) -> JobStatus:
        """Poll an asynchronous job handle."""
        ...

    def fetch(self, handle: JobHandle) -> SimulationResult | list[SimulationResult]:
        """Fetch the completed result for an asynchronous job handle."""
        ...
