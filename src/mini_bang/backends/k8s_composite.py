from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from mini_bang.backends.base import CompositeJobHandle, JobHandle, JobStatus
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.resources import ResourceSpec
from mini_bang.execution.progress import report_progress
from . import observation_recovery, observation_wait


@dataclass
class CompositeChild:
    indices: list[int]
    handle: JobHandle
    manifest: dict[str, Any]
    submitted: bool = False


@dataclass
class CompositeState:
    children: list[CompositeChild]
    max_active: int


def submit_batch(backend: Any, requests: list[SimulationRequest], context: RuntimeContext) -> CompositeJobHandle:
    chunks = list(backend._request_chunks(requests))
    run_id = context.run_id or f"k8s-run-{uuid4().hex[:12]}"
    composite_id = f"k8s-batch-{uuid4().hex[:20]}"
    children: list[CompositeChild] = []
    for indices, chunk in chunks:
        child_handle, manifest = backend._prepare_job(
            chunk,
            context=context,
            run_id=run_id,
            request_indices=indices,
        )
        children.append(CompositeChild(indices=indices, handle=child_handle, manifest=manifest))
    state = CompositeState(children=children, max_active=backend._max_parallel_jobs(requests))
    handle = CompositeJobHandle(
        backend="k8s",
        job_id=composite_id,
        run_id=run_id,
        task_id=context.task_id,
        instance_id=context.instance_id,
        child_handles=tuple(child.handle for child in children),
        child_indexes=tuple(tuple(child.indices) for child in children),
        metadata={
            "namespace": backend.config.namespace,
            "batch_size": len(requests),
            "child_count": len(children),
            "child_job_ids": [child.handle.job_id for child in children],
            "chunk_size": backend.config.batch_chunk_size,
            "max_parallel_jobs": state.max_active,
        },
    )
    # The caller's durable receipt must own the entire queue before any create.
    report_progress('backend_job', backend_kind='k8s_handle',
        handle={'kind': 'composite', 'value': asdict(handle)}, job_id=handle.job_id,
        run_id=handle.run_id, result_uri=handle.result_uri)
    backend._composites[composite_id] = state
    try:
        refresh_statuses(backend, state)
    except BaseException as error:
        if isinstance(error, observation_wait.ObservationInfrastructureFailure):
            _abort(backend, handle, state, {'category': 'infrastructure_failed',
                'reason': 'batch_child_infrastructure_failed'})
            error.original_handle = handle
            raise
        if observation_wait.transient(error):
            error.original_handle = handle
            raise
        _abort(backend, handle, state, {'category': 'infrastructure_failed',
            'reason': 'batch_submission_interrupted'})
        raise
    return handle


def poll(backend: Any, handle: CompositeJobHandle) -> JobStatus:
    terminal = _record(backend, handle, 'terminal')
    if terminal is not None:
        backend._composites.pop(handle.job_id, None)
        return JobStatus(state=terminal['state'], detail=terminal.get('detail'))
    state = backend._composites.get(handle.job_id)
    abort = _record(backend, handle, 'abort-intent')
    if abort is not None:
        return _finish_abort(backend, handle, abort)
    if state is None:
        state = _restore_state(backend, handle)
        backend._composites[handle.job_id] = state
    try:
        statuses = refresh_statuses(backend, state)
    except BaseException as error:
        if isinstance(error, observation_wait.ObservationInfrastructureFailure):
            _abort(backend, handle, state, {'category': 'infrastructure_failed',
                'reason': 'batch_child_infrastructure_failed'})
            error.original_handle = handle
            raise
        if observation_wait.transient(error):
            error.original_handle = handle
            raise
        _abort(backend, handle, state, {'category': 'infrastructure_failed',
            'reason': 'batch_poll_interrupted'})
        raise
    if any(status.state == "failed" for status in statuses):
        failed = next(status for status in statuses if status.state == "failed")
        return _abort(backend, handle, state, failed.detail)
    if all(status.state == "succeeded" for status in statuses):
        _write_record(backend, handle, 'terminal', {'state': 'succeeded', 'detail': None})
        backend._composites.pop(handle.job_id, None)
        return JobStatus(state="succeeded")
    if any(status.state == "running" for status in statuses):
        return JobStatus(state="running")
    return JobStatus(state="pending")


def fetch(backend: Any, handle: CompositeJobHandle) -> list[SimulationResult]:
    status = backend.poll(handle)
    if status.state != "succeeded":
        raise RuntimeError(status.detail or f"Kubernetes batch {handle.job_id} is not complete")
    results: list[SimulationResult | None] = [None] * int(handle.metadata["batch_size"])
    for indices, child_handle in zip(handle.child_indexes, handle.child_handles):
        fetched = backend.fetch(child_handle)
        fetched_list = fetched if isinstance(fetched, list) else [fetched]
        if len(fetched_list) != len(indices):
            raise RuntimeError(
                f"Kubernetes job {child_handle.job_id} returned "
                f"{len(fetched_list)} results for {len(indices)} requests"
            )
        for index, result in zip(indices, fetched_list):
            results[index] = result
    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"Kubernetes batch completed without results for indexes {missing}")
    return [result for result in results if result is not None]


def refresh_statuses(backend: Any, state: CompositeState) -> list[JobStatus]:
    statuses = [
        backend._poll_single(child.handle) if child.submitted else JobStatus(state="pending")
        for child in state.children
    ]
    # Observe every already submitted sibling BEFORE admitting another child.
    if any(status.state == 'failed' for status in statuses):
        return statuses
    active = sum(
        1
        for child, status in zip(state.children, statuses)
        if child.submitted and status.state in {"pending", "running"}
    )
    for index, child in enumerate(state.children):
        if active >= state.max_active:
            break
        if not child.submitted:
            # A transport exception may mean the API accepted this exact name.
            child.submitted = True
            try:
                observation_wait.create_or_rejoin(backend, child.manifest, handle=child.handle)
            except Exception as error:
                status = str(getattr(error, 'status', ''))
                if status.isdigit() and 400 <= int(status) < 500 and int(status) not in {408, 409, 429}:
                    # A definite rejection did not start this new child. Older
                    # accepted/uncertain siblings still require cancellation.
                    child.submitted = False
                raise
            statuses[index] = JobStatus(state="pending")
            active += 1
    return statuses


def _restore_state(backend: Any, handle: CompositeJobHandle) -> CompositeState:
    """Rebuild only from the saved immutable inputs and durable launch identities."""
    from .observation_launch import launch_state
    maximum = handle.metadata.get('max_parallel_jobs')
    flattened = [index for indices in handle.child_indexes for index in indices]
    if (type(maximum) is not int or maximum < 1 or len(handle.child_handles) != len(handle.child_indexes) or
            handle.metadata.get('namespace') != backend.config.namespace or
            sorted(flattened) != list(range(handle.metadata.get('batch_size', -1))) or
            len({child.job_id for child in handle.child_handles}) != len(handle.child_handles)):
        raise observation_wait.infrastructure_error('Invalid retained batch scheduler parameters', handle)
    children = []
    for child, indices in zip(handle.child_handles, handle.child_indexes):
        started = launch_state(backend, child)
        ready = observation_recovery.terminal_result(backend, child)
        manifest = backend.result_store.get_job_input(child.metadata['input_uri'])
        if (manifest.get('job_id') != child.job_id or manifest.get('run_id') != handle.run_id or
                manifest.get('result_uri') != child.result_uri):
            raise observation_wait.infrastructure_error('Retained batch input identity differs', handle)
        body = backend._job_manifest(child.job_id, child.metadata['input_uri'],
            ResourceSpec.from_dict(manifest['resource_spec']), RuntimeContext(**manifest['runtime_context']))
        children.append(CompositeChild(list(indices), child, body, submitted=ready is not None or started != 'never_started'))
    configured = backend.config.max_parallel_jobs
    return CompositeState(children=children, max_active=maximum if configured is None else min(maximum, configured))


def _record(backend, handle, suffix):
    uri = backend.result_store.job_result_uri(handle.run_id, handle.job_id + '-' + suffix)
    if not backend.result_store.stat(uri):
        return None
    try:
        value = backend.result_store.get_job_result(uri)
    except (TypeError, ValueError) as error:
        raise observation_wait.infrastructure_error('Invalid retained batch control record', handle) from error
    if (not isinstance(value, dict) or value.get('kind') != 'observation_batch_' + suffix or
            value.get('job_id') != handle.job_id or value.get('run_id') != handle.run_id or
            value.get('child_job_ids') != [child.job_id for child in handle.child_handles]):
        raise observation_wait.infrastructure_error('Invalid retained batch control record', handle)
    if suffix == 'terminal' and value.get('state') not in {'failed', 'succeeded'}:
        raise observation_wait.infrastructure_error('Invalid retained batch terminal state', handle)
    return value


def _write_record(backend, handle, suffix, payload):
    existing = _record(backend, handle, suffix)
    if existing is not None:
        return existing
    value = {'kind': 'observation_batch_' + suffix, 'job_id': handle.job_id,
             'run_id': handle.run_id, 'child_job_ids': [child.job_id for child in handle.child_handles],
             **payload}
    try:
        backend.result_store.put_job_result(handle.run_id, handle.job_id + '-' + suffix, value)
    except FileExistsError:
        retained = _record(backend, handle, suffix)
        if retained is None:
            raise
        return retained
    return value


def _abort(backend, handle, state, detail):
    intent = _write_record(backend, handle, 'abort-intent', {
        'detail': detail, 'submitted_job_ids': [child.handle.job_id for child in state.children if child.submitted]})
    return _finish_abort(backend, handle, intent)


def _finish_abort(backend, handle, intent):
    submitted = intent.get('submitted_job_ids')
    allowed = {child.job_id: child for child in handle.child_handles}
    if not isinstance(submitted, list) or not set(submitted) <= set(allowed):
        raise observation_wait.infrastructure_error('Invalid retained batch stop identities', handle)
    targets = []
    for name in submitted:
        original = allowed[name]
        candidates = [original]
        for number in range(1, backend.config.observation_max_retries + 1):
            retried = observation_recovery.retry_handle(backend, original, number)
            if retried is not None:
                candidates.append(retried)
        for child in candidates:
            if observation_recovery.terminal_result(backend, child) is not None:
                continue
            # Persist every stop intent before the FIRST DELETE. An API CREATE
            # completing after that DELETE404 must not start simulation later.
            uri = backend.result_store.job_result_uri(child.run_id, child.job_id + '-watchdog-diagnostics')
            if not backend.result_store.stat(uri):
                try:
                    backend.result_store.put_job_result(child.run_id, child.job_id + '-watchdog-diagnostics', {
                        'status': 'diagnostic', 'job_id': child.job_id, 'run_id': child.run_id,
                        'failure': {'category': 'infrastructure_failed', 'reason': 'batch_aborted',
                                    'batch_job_id': handle.job_id},
                        'progress': None, 'log_tail': '', 'log_error': None})
                except FileExistsError:
                    # A concurrent watchdog may have already committed its own
                    # immutable stop intent; verify it rather than overwrite it.
                    if observation_recovery.watchdog_failure(backend, child) is None:
                        raise
            targets.append(child)
    for child in targets:
        try:
            backend._client.stop_job(backend.config.namespace, child.job_id)
        except Exception as error:
            raise observation_wait.infrastructure_error(
                'Batch sibling stop was not confirmed; durable abort intent retained', handle) from error
    terminal = _write_record(backend, handle, 'terminal', {'state': 'failed', 'detail': intent.get('detail')})
    backend._composites.pop(handle.job_id, None)
    return JobStatus(state=terminal['state'], detail=terminal.get('detail'))
