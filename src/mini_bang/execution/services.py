from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mini_bang.artifacts import ArtifactHandle, ArtifactRef
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices


class ExecutionServices:
    """Shared execution substrate entrypoint for trusted runtime callers.

    The caller supplies the authority-bearing ``RuntimeContext`` for every call.
    This layer does not know task semantics, policies, budgets, or scorer logic.
    """

    def __init__(self, *, executor: SimulationExecutor | None = None) -> None:
        self._executor = executor or SimulationExecutor()

    def run_simulation(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext,
    ) -> SimulationResult:
        """Run one simulation through the configured ``SimulationExecutor``."""
        return self._executor.run(request, context=context)

    def run_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext,
    ) -> list[SimulationResult]:
        """Run simulations with the supplied authority context."""
        return self._executor.run_batch(requests, context=context)

    def materialize_artifact(
        self,
        ref: ArtifactRef | dict[str, Any],
        *,
        context: RuntimeContext,
    ) -> ArtifactHandle:
        """Materialize an external source artifact using the supplied context."""
        return RuntimeServices.from_context(context).artifact_resolver.materialize(ref)

    def describe(self) -> dict[str, Any]:
        """Return execution substrate metadata."""
        return self._executor.describe()


class BackendExecutionServices:
    """ExecutionServices-compatible wrapper over a SimulationBackend.

    Scoring workers use this when hidden simulations should fan out through the
    same backend/scaling layer instead of running inside the scoring process.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def run_simulation(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext,
    ) -> SimulationResult:
        return self.run_batch([request], context=context)[0]

    def run_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext,
    ) -> list[SimulationResult]:
        return self._backend.run_batch(requests, context=context)

    def materialize_artifact(
        self,
        ref: ArtifactRef | dict[str, Any],
        *,
        context: RuntimeContext,
    ) -> ArtifactHandle:
        return RuntimeServices.from_context(context).artifact_resolver.materialize(ref)

    def describe(self) -> dict[str, Any]:
        describe = getattr(self._backend, "describe", None)
        return describe() if callable(describe) else {"backend": self._backend.__class__.__name__}
