from __future__ import annotations

import atexit
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Any, Literal, Protocol

from mini_bang.execution.models import (
    SimulationJob,
    SimulationJobGroup,
    SimulationSample,
    stable_json,
)
from mini_bang.execution.resources import ResourceSpec
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.framework.simulation import MacroSession
from mini_bang.simulators.registry import get_simulator_entry

ReuseStrategy = Literal["none", "grouped"]


class ExecutionEngine(Protocol):
    """Execution strategy for independent simulation job groups."""

    def run_jobs(
        self,
        jobs: list[SimulationJob],
        *,
        context: RuntimeContext,
        services: RuntimeServices | None = None,
    ) -> list[SimulationSample]:
        """Run jobs and return samples in original job order."""
        ...

    def plan_groups(self, jobs: list[SimulationJob]) -> list[SimulationJobGroup]:
        """Group compatible jobs according to this engine's reuse strategy."""
        ...

    def describe(self) -> dict:
        """Return machine-readable engine metadata."""
        ...


class SequentialEngine:
    """Execute simulation jobs in the current process."""

    def __init__(
        self,
        *,
        reuse_strategy: ReuseStrategy = "none",
        local_worker_capacity: int | None = None,
    ) -> None:
        self.reuse_strategy = reuse_strategy
        self.local_worker_capacity = local_worker_capacity

    def run_jobs(
        self,
        jobs: list[SimulationJob],
        *,
        context: RuntimeContext,
        services: RuntimeServices | None = None,
    ) -> list[SimulationSample]:
        if not jobs:
            return []
        resolved_services = services or RuntimeServices.from_context(context)
        groups = self.plan_groups(jobs)
        _validate_group_capacity(groups, self.local_worker_capacity)
        samples: list[SimulationSample] = []
        for group in groups:
            samples.extend(_execute_job_group(group, services=resolved_services,
                progress_offset=len(samples), progress_total=len(jobs)))
            from mini_bang.execution.progress import report_progress
            report_progress('samples', completed=len(samples), total=len(jobs), unit='rollouts')
        return order_samples_by_job(jobs, samples)

    def plan_groups(self, jobs: list[SimulationJob]) -> list[SimulationJobGroup]:
        return group_compatible_jobs(jobs, reuse_strategy=self.reuse_strategy)

    def describe(self) -> dict:
        return {
            "engine": "sequential",
            "reuse_strategy": self.reuse_strategy,
            "local_worker_capacity": self.local_worker_capacity,
        }


class ProcessPoolEngine:
    """Execute compatible job groups in a local process pool."""

    def __init__(
        self,
        max_workers: int | None = None,
        chunksize: int = 1,
        *,
        reuse_strategy: ReuseStrategy = "grouped",
    ) -> None:
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.reuse_strategy = reuse_strategy
        self._pool: ProcessPoolExecutor | None = None
        atexit.register(self.close)

    def run_jobs(
        self,
        jobs: list[SimulationJob],
        *,
        context: RuntimeContext,
        services: RuntimeServices | None = None,
    ) -> list[SimulationSample]:
        if not jobs:
            return []
        if services is not None:
            raise ValueError(
                "ProcessPoolEngine requires serializable RuntimeContext; "
                "live RuntimeServices/ArtifactResolver cannot cross process boundary."
            )
        groups = self.plan_groups(jobs)
        max_workers = self._effective_max_workers()
        _validate_group_capacity(groups, max_workers)
        pool = self._ensure_pool()
        group_results = self._run_weighted(pool, groups, context, max_workers=max_workers)
        samples = [sample for group in group_results for sample in group]
        return order_samples_by_job(jobs, samples)

    def plan_groups(self, jobs: list[SimulationJob]) -> list[SimulationJobGroup]:
        return group_compatible_jobs(jobs, reuse_strategy=self.reuse_strategy)

    def describe(self) -> dict:
        return {
            "engine": "process_pool",
            "max_workers": self.max_workers,
            "chunksize": self.chunksize,
            "reuse_strategy": self.reuse_strategy,
            "pool": "persistent",
        }

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # Forking after JAX initializes its threads deadlocks on Linux.
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=multiprocessing.get_context("spawn")
            )
        return self._pool

    def _effective_max_workers(self) -> int:
        if self.max_workers is not None:
            return int(self.max_workers)
        process_cpu_count = getattr(os, "process_cpu_count", None)
        if callable(process_cpu_count):
            return process_cpu_count() or 1
        return os.cpu_count() or 1

    def _run_weighted(
        self,
        pool: ProcessPoolExecutor,
        groups: list[SimulationJobGroup],
        context: RuntimeContext,
        *,
        max_workers: int,
    ) -> list[list[SimulationSample]]:
        pending = [(group, _group_resource_weight(group)) for group in groups]
        active: dict[Future[list[SimulationSample]], int] = {}
        active_weight = 0
        results: list[list[SimulationSample]] = []

        while pending or active:
            while pending and active_weight + pending[0][1] <= max_workers:
                group, weight = pending.pop(0)
                # Spawn workers do not inherit dynamically registered plugins.
                # Carry the immutable manifest, never a live factory or authority.
                spec = get_simulator_entry(group.jobs[0].simulator_id)
                future = pool.submit(_execute_job_group_from_context, (group, context, spec))
                active[future] = weight
                active_weight += weight

            if not active:
                raise RuntimeError("weighted process-pool scheduler has no runnable group")

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                weight = active.pop(future)
                active_weight -= weight
                results.append(future.result())

        return results


def job_identity(job: SimulationJob) -> tuple[int, int]:
    """Exact identity of a job inside one batch: ``(request_index, run_index)``.

    ``job_id`` is a 32-bit digest kept for traces; two jobs of one large batch
    can share it, so results, groups and checks never key on it.
    """
    return (job.request_index, job.run_index)


def ensure_unique_job_identities(jobs: list[SimulationJob]) -> None:
    """Fail loudly when two planned jobs claim the same request and run."""
    seen: set[tuple[int, int]] = set()
    for job in jobs:
        identity = job_identity(job)
        if identity in seen:
            raise RuntimeError(
                f"Two planned jobs share request {identity[0]} run {identity[1]}; "
                "results would move between requests"
            )
        seen.add(identity)


def order_samples_by_job(
    jobs: list[SimulationJob],
    samples: list[SimulationSample],
) -> list[SimulationSample]:
    """Return exactly one sample per planned job, in job order, or fail loudly."""
    ensure_unique_job_identities(jobs)
    by_identity: dict[tuple[int, int], SimulationSample] = {}
    for sample in samples:
        identity = job_identity(sample.job)
        if identity in by_identity:
            raise RuntimeError(f"Engine returned two samples for request {identity[0]} run {identity[1]}")
        by_identity[identity] = sample
    expected = [job_identity(job) for job in jobs]
    if set(by_identity) != set(expected):
        missing = sorted(set(expected) - set(by_identity))
        unexpected = sorted(set(by_identity) - set(expected))
        raise RuntimeError(
            f"Engine samples do not match planned jobs: missing {missing[:5]}, unexpected {unexpected[:5]}"
        )
    ordered = [by_identity[identity] for identity in expected]
    for job, sample in zip(jobs, ordered):
        if sample.job.request_hash != job.request_hash:
            raise RuntimeError(
                f"Sample for request {job.request_index} run {job.run_index} belongs to another request"
            )
    return ordered


def group_compatible_jobs(
    jobs: list[SimulationJob],
    *,
    reuse_strategy: ReuseStrategy,
) -> list[SimulationJobGroup]:
    """Group jobs by deterministic simulator compatibility keys."""
    if reuse_strategy == "none":
        return [
            SimulationJobGroup(
                group_id=_group_label(job),
                compatibility_key=_compatibility_key(job),
                jobs=(job,),
            )
            for job in jobs
        ]
    indexed: dict[str, list[SimulationJob]] = defaultdict(list)
    first_index: dict[str, int] = {}
    for index, job in enumerate(jobs):
        if not _grouped_sampling_safe(job.simulator_id):
            # Unsafe simulators never share macro/micro state: one group per job.
            key = f"ungrouped-request-{job.request_index}-run-{job.run_index}"
            indexed[key].append(job)
            first_index.setdefault(key, index)
            continue
        key = _compatibility_key(job)
        indexed[key].append(job)
        first_index.setdefault(key, index)
    groups: list[SimulationJobGroup] = []
    for key in sorted(indexed, key=lambda item: first_index[item]):
        group_jobs = tuple(indexed[key])
        groups.append(
            SimulationJobGroup(
                group_id=_group_label(group_jobs[0]),
                compatibility_key=key,
                jobs=group_jobs,
            )
        )
    return groups


def _group_label(first_job: SimulationJob) -> str:
    # Every job is in exactly one group, so the first job names its group exactly.
    return f"group-from-request-{first_job.request_index}-run-{first_job.run_index}"


def _compatibility_key(job: SimulationJob) -> str:
    sample_params_without_seed = {
        key: value
        for key, value in job.sample_params.items()
        if key != "seed"
    }
    return stable_json(
        {
            "simulator_id": job.simulator_id,
            "saturation": job.saturation,
            "macro_params": job.macro_params,
            "micro_params": job.micro_params,
            "sample_params_without_seed": sample_params_without_seed,
            "extras": list(job.extras),
            "instance_id": job.instance_id,
        }
    )


def _validate_group_capacity(
    groups: list[SimulationJobGroup],
    max_workers: int | None,
) -> None:
    if max_workers is None:
        return
    for group in groups:
        spec = _group_resource_spec(group)
        spec.validate_against_local_workers(max_workers=max_workers)


def _group_resource_weight(group: SimulationJobGroup) -> int:
    return _group_resource_spec(group).local_worker_weight()


def _group_resource_spec(group: SimulationJobGroup) -> ResourceSpec:
    # A group runs in one worker process and samples its jobs sequentially while
    # reusing macro/micro state, so local scheduling accounts for the active
    # group weight rather than multiplying by the number of jobs in the group.
    spec = ResourceSpec()
    for job in group.jobs:
        job_spec = _job_resource_spec(job)
        if job_spec.local_worker_weight() > spec.local_worker_weight():
            spec = job_spec
    return spec


def _job_resource_spec(job: SimulationJob) -> ResourceSpec:
    spec = get_simulator_entry(job.simulator_id).resource_spec
    override = job.sample_params.get("resource_spec") or job.sample_params.get("resources")
    return spec.merge(override)


def _execute_job_group(
    group: SimulationJobGroup,
    *,
    services: RuntimeServices,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> list[SimulationSample]:
    jobs = list(group.jobs)
    if not jobs:
        return []
    first = jobs[0]
    started = time.perf_counter()
    entry = get_simulator_entry(first.simulator_id)
    resolver = services.artifact_resolver
    macro = entry.create_macro(
        artifact_resolver=resolver,
        runtime_services=services,
        **first.macro_params,
    )
    macro_session = MacroSession(macro)
    micro = macro_session.create_micro(first.saturation, **first.micro_params)
    saturation_is_sequence = isinstance(first.saturation, (list, tuple))
    version = (
        entry.metadata.get("version")
        or entry.config.get("version")
        or entry.config.get("simulator_version")
    )
    response_builder = entry.response_builder()

    samples: list[SimulationSample] = []
    from mini_bang.execution.progress import report_progress
    for job in jobs:
        report_progress('samples', phase='executing', completed=progress_offset+len(samples),
                        total=progress_total or len(jobs), unit='rollouts')
        job_started = time.perf_counter()
        if saturation_is_sequence:
            micro_list = micro if isinstance(micro, list) else [micro]
            trajectories = [[session.sample(**job.sample_params)] for session in micro_list]
        else:
            session = micro[0] if isinstance(micro, list) else micro
            trajectories = [session.sample(**job.sample_params)]

        response = response_builder(
            simulator_id=job.simulator_id,
            saturation=job.saturation,
            runs=1,
            macro_params=job.macro_params,
            micro_params=job.micro_params,
            sample_params=job.sample_params,
            micro_session=micro,
            trajectories=trajectories,
            extras=list(job.extras),
        )
        elapsed = time.perf_counter() - job_started
        samples.append(
            SimulationSample(
                job=job,
                payload=response,
                metadata={
                    "effective_seed": job.effective_seed,
                    "base_seed": job.base_seed,
                    "seed_mode": job.seed_mode,
                    "seed_source": job.seed_source,
                    "response_builder": entry.response_builder_path or "default",
                    "group_id": group.group_id,
                    "group_size": len(jobs),
                    "group_wall_seconds": time.perf_counter() - started,
                    "compatibility_key": group.compatibility_key,
                },
                timings={"wall_seconds": elapsed},
                simulator_version=str(version) if version is not None else None,
            )
        )
        report_progress('samples', phase='executing', completed=progress_offset+len(samples),
                        total=progress_total or len(jobs), unit='rollouts')
    return samples


_WORKER_SERVICES: dict[str, RuntimeServices] = {}


def _execute_job_group_from_context(args: tuple) -> list[SimulationSample]:
    from mini_bang.simulators.registry import register_simulator_entry

    group, context, spec = args
    try:
        existing = get_simulator_entry(spec.simulator_id)
    except KeyError:
        register_simulator_entry(spec)
    else:
        execution_keys = ("factory_path", "response_builder_path", "aggregator_path", "grouped_sampling_safe", "config")
        if any(getattr(existing, key) != getattr(spec, key) for key in execution_keys):
            raise ValueError(f"Worker simulator manifest differs for {spec.simulator_id}")
    key = context.stable_key()
    services = _WORKER_SERVICES.get(key)
    if services is None:
        services = RuntimeServices.from_context(context)
        _WORKER_SERVICES[key] = services
    return _execute_job_group(group, services=services)


def _grouped_sampling_safe(simulator_id: str) -> bool:
    try:
        entry = get_simulator_entry(simulator_id)
    except Exception:
        return False
    return bool(entry.grouped_sampling_safe)
