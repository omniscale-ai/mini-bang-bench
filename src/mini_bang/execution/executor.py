from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from mini_bang.artifacts import ArtifactResolver
from mini_bang.execution.aggregator import SimulationAggregator
from mini_bang.execution.engines import (
    ExecutionEngine,
    ProcessPoolEngine,
    SequentialEngine,
    ensure_unique_job_identities,
)
from mini_bang.execution.models import SimulationRequest, SimulationResult, SimulationSample
from mini_bang.execution.payload_stores.naming import payload_label
from mini_bang.execution.payloads import PayloadStorageConfig
from mini_bang.execution.planner import SimulationPlanner
from mini_bang.execution.resources import ResourceSpec
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.simulators.registry import get_simulator_entry


class SimulationExecutor:
    """Plan, execute, and aggregate simulator jobs for backend implementations."""

    def __init__(
        self,
        *,
        engine: ExecutionEngine | None = None,
        planner: SimulationPlanner | None = None,
        aggregator: SimulationAggregator | None = None,
        payload_storage: PayloadStorageConfig | None = None,
        artifact_resolver: ArtifactResolver | None = None,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> None:
        self._engine = engine or SequentialEngine()
        self._planner = planner or SimulationPlanner()
        self._aggregator = aggregator or SimulationAggregator()
        self._payload_storage = payload_storage or PayloadStorageConfig.inline()
        self._context = context or RuntimeContext.from_env()
        if services is not None and artifact_resolver is not None:
            raise ValueError("pass either services or artifact_resolver, not both")
        self._services = services
        if artifact_resolver is not None:
            self._services = RuntimeServices(artifact_resolver=artifact_resolver)

    def run(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        """Execute one high-level simulation request."""
        return self.run_batch([request], context=context, services=services)[0]

    def run_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        """Execute a batch of requests through one engine submission."""
        effective_context, effective_services = self._runtime_for_call(
            context=context,
            services=services,
        )
        request_list = list(requests)
        if isinstance(self._engine, ProcessPoolEngine):
            _validate_process_pool_resources(request_list, self._engine.max_workers)
        all_jobs = []
        for index, request in enumerate(request_list):
            all_jobs.extend(self._planner.plan(request, request_index=index))
        ensure_unique_job_identities(all_jobs)

        planned_groups = self._engine.plan_groups(all_jobs)
        if isinstance(self._engine, ProcessPoolEngine) and effective_services is not None:
            raise ValueError(
                "ProcessPoolEngine requires serializable RuntimeContext; "
                "live RuntimeServices/ArtifactResolver cannot cross process boundary."
            )
        samples = self._engine.run_jobs(
            all_jobs,
            context=effective_context,
            services=effective_services,
        )
        grouped: dict[int, list[SimulationSample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.job.request_index].append(sample)

        results: list[SimulationResult] = []
        for index, request in enumerate(request_list):
            request_samples = grouped.get(index, [])
            returned_runs = sorted(sample.job.run_index for sample in request_samples)
            if returned_runs != list(range(request.runs)):
                raise RuntimeError(
                    f"Request {index} asked for runs 0..{request.runs - 1} "
                    f"but the engine returned runs {returned_runs}"
                )
            payload = self._aggregator.aggregate(request, request_samples)
            payload, payload_ref = self._payload_storage.externalize(
                payload,
                key=request_payload_label(request, effective_context),
                metadata={
                    "request_hash": request.request_hash,
                    "simulator_id": request.simulator_id,
                    "request_id": request.request_id,
                },
            )
            versions = self._versions(request_samples)
            payload_metadata: dict[str, object] = self._payload_storage.describe()
            if payload_ref is not None:
                payload_metadata["ref"] = payload_ref.to_dict()
            results.append(
                SimulationResult(
                    request=request,
                    payload=payload,
                    request_hash=request.request_hash,
                    payload_ref=payload_ref,
                    backend_metadata={
                        "backend": "local",
                        **self._engine.describe(),
                        "payload_storage": payload_metadata,
                        "job_count": len(request_samples),
                        "group_count": len(
                            {
                                sample.metadata.get("group_id")
                                for sample in request_samples
                                if sample.metadata.get("group_id")
                            }
                        ),
                    },
                    resource_usage={
                        "jobs": len(request_samples),
                        **(
                            {"payload_bytes": payload_ref.size_bytes}
                            if payload_ref is not None and payload_ref.size_bytes is not None
                            else {}
                        ),
                    },
                    versions=versions,
                    trace=self._trace_for_request(
                        request_index=index,
                        request=request,
                        samples=request_samples,
                        planned_groups=planned_groups,
                    ),
                )
            )
        return results

    def describe(self) -> dict:
        """Return machine-readable executor metadata."""
        return {
            "executor": "simulation_executor",
            **self._engine.describe(),
            "payload_storage": self._payload_storage.describe(),
        }

    def set_artifact_resolver(self, artifact_resolver: ArtifactResolver) -> None:
        """Set the default external source artifact resolver for convenience paths."""
        self._services = RuntimeServices(artifact_resolver=artifact_resolver)

    def set_runtime_context(self, context: RuntimeContext) -> None:
        """Set the default runtime settings for calls without explicit context."""
        self._context = context
        self._services = None

    def set_runtime_services(self, services: RuntimeServices) -> None:
        """Set default live runtime services for in-process convenience paths."""
        self._services = services

    def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()

    def _versions(self, samples: list[SimulationSample]) -> dict:
        response_builders = {
            sample.metadata.get("response_builder")
            for sample in samples
            if sample.metadata.get("response_builder")
        }
        simulator_versions = {
            sample.simulator_version
            for sample in samples
            if sample.simulator_version is not None
        }
        return {
            "simulator_version": sorted(simulator_versions),
            "response_builder_version": sorted(response_builders),
        }

    def _runtime_for_call(
        self,
        *,
        context: RuntimeContext | None,
        services: RuntimeServices | None,
    ) -> tuple[RuntimeContext, RuntimeServices | None]:
        if services is not None:
            return context or self._context, services
        if context is not None:
            return context, None
        return self._context, self._services

    def _trace_for_request(
        self,
        *,
        request_index: int,
        request: SimulationRequest,
        samples: list[SimulationSample],
        planned_groups,
    ) -> list[dict]:
        jobs = [sample.job for sample in sorted(samples, key=lambda item: item.job.run_index)]
        group_ids = {
            sample.metadata.get("group_id")
            for sample in samples
            if sample.metadata.get("group_id")
        }
        groups = [
            {
                "group_id": group.group_id,
                "compatibility_key": group.compatibility_key,
                "job_ids": [job.job_id for job in group.jobs],
                "size": len(group.jobs),
            }
            for group in planned_groups
            if group.group_id in group_ids
        ]
        return [
            {
                "event": "simulation_planned",
                "request_hash": request.request_hash,
                "request_index": request_index,
                "request": request.to_payload(),
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "run_index": job.run_index,
                        "base_seed": job.base_seed,
                        "effective_seed": job.effective_seed,
                        "seed_mode": job.seed_mode,
                        "seed_source": job.seed_source,
                        "simulator_id": job.simulator_id,
                        "saturation": job.saturation,
                    }
                    for job in jobs
                ],
                "groups": groups,
                "engine": self._engine.describe(),
            }
        ]


def request_payload_label(request: SimulationRequest, context: RuntimeContext | None) -> str:
    """Readable name part of a stored result: task, simulator, size, runs and given seeds.

    Only values written in the request are used (seeds the planner derives are
    not), so the name never says more than the stored payload itself.
    """
    saturation = request.saturation
    size = "-".join(str(item) for item in saturation) if isinstance(saturation, list) else saturation
    parts: list[object] = [getattr(context, "task_id", None), request.simulator_id,
                           f"saturation {size}", f"runs {request.runs}"]
    for label, value in (("macro seed", request.macro_params.get("seed")),
                         ("sample seed", request.sample_params.get("seed")),
                         ("case", request.macro_params.get("case_id"))):
        if isinstance(value, (int, str)) and not isinstance(value, bool):
            parts.append(f"{label} {value}")
    start, end = request.sample_params.get("start_step"), request.sample_params.get("end_step")
    if isinstance(start, int) and isinstance(end, int):
        parts.append(f"steps {start} to {end}")
    return payload_label(*parts)


def _validate_process_pool_resources(
    requests: list[SimulationRequest],
    max_workers: int | None,
) -> None:
    if max_workers is None:
        return
    for request in requests:
        spec = get_simulator_entry(request.simulator_id).resource_spec
        override = request.sample_params.get("resource_spec") or request.sample_params.get("resources")
        spec = spec.merge(override)
        spec.validate_against_local_workers(max_workers=max_workers)
