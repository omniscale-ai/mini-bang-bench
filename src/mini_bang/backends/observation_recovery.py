"""Bounded, deterministic observation recovery without another policy debit."""
from copy import deepcopy
import time
import json
import os
from pathlib import Path

from mini_bang.backends.base import JobHandle, JobStatus
from mini_bang.backends.observation_launch import VERSION as LAUNCH_TRACKING_VERSION
from mini_bang.execution.resources import ResourceSpec
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.progress import report_progress
from mini_bang.backends.observation_scheduling import scheduling_failure, heartbeat_stalled, stop_with_evidence


def capacity_from_env():
    path = os.getenv("MINI_BANG_RUNTIME_REFERENCE_PROFILE")
    if not path:
        if os.getenv("MINI_BANG_REQUIRE_RUNTIME_CALIBRATION") == "1":
            raise OSError("A frozen runtime reference profile is required before evaluation")
        return None
    return {"required": True, "reference_identity": Path(path).name,
        "reference": json.loads(Path(path).read_text()), "samples": 20,
        "maximum_slowdown": float(os.getenv("MINI_BANG_RUNTIME_MAXIMUM_SLOWDOWN", "5")),
        "minimum_cpu_count": float(os.getenv("MINI_BANG_RUNTIME_MINIMUM_CPU", "1")),
        "minimum_memory_mb": float(os.getenv("MINI_BANG_RUNTIME_MINIMUM_MEMORY_MB", "256"))}


def configure_job(body, config):
    """A Job deadline is an optional outer fuse, never the compute clock."""
    body["spec"].pop("activeDeadlineSeconds", None)
    if config.emergency_job_timeout_seconds is not None:
        if config.emergency_job_timeout_seconds <= config.observation_idle_seconds:
            raise ValueError("Outer observation fuse must exceed the inner idle watchdog")
        body["spec"]["activeDeadlineSeconds"] = config.emergency_job_timeout_seconds
    return body


def policy(config):
    if config.observation_idle_seconds <= 0 or not 0 <= config.observation_max_retries <= 2:
        raise ValueError("Observation watchdog must be positive and retries between zero and two")
    if config.observation_scheduling_idle_seconds <= 0:
        raise ValueError("Scheduling watchdog must be positive")
    if config.observation_cpu_limit_seconds is not None and config.observation_cpu_limit_seconds <= 0:
        raise ValueError("Configured observation CPU fuse must be positive")
    return {"idle_seconds": config.observation_idle_seconds,
            "cpu_limit_seconds": config.observation_cpu_limit_seconds,
            "maximum_retries": config.observation_max_retries,
            "queue_counts_as_compute": False}


def retry_handle(backend, original, number, *, create=False):
    job_id = original.job_id + f"-retry-{number:02d}"
    old_uri = original.metadata["input_uri"]
    input_uri = old_uri.rsplit("/", 1)[0] + "/" + job_id + ".json"
    if not create and not backend.result_store.stat(input_uri):
        return None
    manifest = deepcopy(backend.result_store.get_job_input(original.metadata["input_uri"]))
    manifest.update(job_id=job_id, result_uri=backend.result_store.job_result_uri(original.run_id, job_id))
    manifest["metadata"] = {**manifest.get("metadata", {}),
        "original_job_id": original.job_id, "infrastructure_retry": number,
        "launch_tracking_version": LAUNCH_TRACKING_VERSION,
        "additional_observation_budget_debits": 0}
    if create:
        backend.result_store.put_job_input(original.run_id, job_id, manifest)
    handle = JobHandle(backend="k8s", job_id=job_id, run_id=original.run_id,
        task_id=original.task_id, instance_id=original.instance_id, result_uri=manifest["result_uri"],
        metadata={**original.metadata, "input_uri": input_uri, "job_name": job_id,
                  "original_job_id": original.job_id, "infrastructure_retry": number})
    return handle


def current_handle(backend, original):
    if terminal_result(backend, original):
        return original
    current = original
    for number in range(1, backend.config.observation_max_retries + 1):
        candidate = retry_handle(backend, original, number)
        if candidate is None:
            break
        current = candidate
        if terminal_result(backend, candidate):
            return candidate
    return current


def start_existing(backend, handle):
    manifest = backend.result_store.get_job_input(handle.metadata["input_uri"])
    context = RuntimeContext(**manifest["runtime_context"])
    body = backend._job_manifest(handle.job_id, handle.metadata["input_uri"],
        ResourceSpec.from_dict(manifest["resource_spec"]), context)
    from .observation_wait import create_or_rejoin
    return create_or_rejoin(backend, body, handle)


def watchdog_failure(backend, handle):
    """The immutable diagnostic is a durable stop intent, not a new request."""
    uri = backend.result_store.job_result_uri(handle.run_id,handle.job_id+'-watchdog-diagnostics')
    if not backend.result_store.stat(uri):
        return None
    diagnostic = backend.result_store.get_job_result(uri)
    failure = diagnostic.get('failure') or {}
    if (diagnostic.get('status')!='diagnostic' or not isinstance(failure,dict) or
            failure.get('category')!='infrastructure_failed' or not failure.get('reason')):
        raise OSError('infrastructure_failed: invalid retained observation stop diagnostic')
    return failure


def failure_detail(error):
    if isinstance(error, dict) and error.get('category') in ('request_rejected', 'infrastructure_failed'):
        value = dict(error)
        if value['category'] == 'infrastructure_failed':
            value.setdefault('reason', 'observation_worker_failure')
        return value
    return {'category': 'infrastructure_failed', 'reason': 'unattributed_worker_failure'}


def terminal_result(backend, handle):
    if not handle.result_uri or not backend.result_store.stat(handle.result_uri):
        return None
    envelope = backend.result_store.get_job_result(handle.result_uri)
    state, error = envelope.get('status'), envelope.get('error')
    if state == 'succeeded' or (state == 'failed' and isinstance(error, dict)
            and error.get('category') == 'request_rejected'):
        return JobStatus(state=state, detail=error)
    return None


def completed(backend, original, current):
    for handle in (original,current):
        ready = terminal_result(backend, handle)
        if ready:
            return ready
    return None


def recover(backend, original, current, failure, *, already_stopped=False):
    ready = completed(backend,original,current)
    if ready:
        return ready
    if not already_stopped:
        diagnostic_id = current.job_id+'-watchdog-diagnostics'
        uri = backend.result_store.job_result_uri(current.run_id, diagnostic_id)
        if not backend.result_store.stat(uri):
            backend.result_store.put_job_result(current.run_id, diagnostic_id,
                {'status': 'diagnostic', 'failure': failure, 'progress': None,
                 'log_tail': '', 'log_error': None})
        try:
            backend._client.stop_job(backend.config.namespace, current.job_id)
        except Exception:
            ready = completed(backend, original, current)
            if ready:
                return ready
            raise
        ready = completed(backend, original, current)
        if ready:
            return ready
    number = int(current.metadata.get("infrastructure_retry", 0)) + 1
    report_progress("observation_infrastructure_failure", job_id=current.job_id,
                    reason=failure["reason"], retry_number=number,
                    maximum_retries=backend.config.observation_max_retries)
    if number > backend.config.observation_max_retries:
        return JobStatus(state="failed", detail={"category": "infrastructure_failed",
            "reason": "observation recovery exhausted", "job_id": current.job_id})
    fresh = retry_handle(backend, original, number, create=True)
    ready = completed(backend,original,current)
    if ready:
        return ready
    start_existing(backend, fresh)
    report_progress("backend_job", job_id=fresh.job_id, result_uri=fresh.result_uri,
                    run_id=fresh.run_id, batch_size=fresh.metadata.get("batch_size"),
                    infrastructure_retry=number, original_job_id=original.job_id)
    return JobStatus(state="pending")


def poll(backend, original):
    current = current_handle(backend, original)
    ready = completed(backend,original,current)
    if ready:
        return ready
    retained_failure = watchdog_failure(backend,current)
    if retained_failure:
        # Rejoin deletion even if the Job is now404 or still Terminating. A retry
        # whose own stop intent exists must never be resurrected under its old ID.
        try:
            backend._client.stop_job(backend.config.namespace,current.job_id)
        except Exception:
            ready = completed(backend,original,current)
            if ready:
                return ready
            raise
        return recover(backend,original,current,retained_failure,already_stopped=True)
    from .observation_launch import ensure_launched
    if ensure_launched(backend, current):
        return JobStatus(state='pending')
    try:
        status = backend._poll_single_raw(current)
    except Exception as error:
        if getattr(error, "status", None) != 404:
            raise
        return recover(backend, original, current, {'category': 'infrastructure_failed',
            'reason': 'original_job_missing_without_result' if current is original else
                      'retry_job_missing_after_launch_intent'})
    failure, already_stopped = None, False
    if current.result_uri and backend.result_store.stat(current.result_uri):
        envelope = backend.result_store.get_job_result(current.result_uri)
        error = failure_detail(envelope.get("error"))
        if envelope.get('status') == 'succeeded':
            return JobStatus(state='succeeded')
        if envelope.get("status") == "failed":
            if error.get("category") == "request_rejected":
                return JobStatus(state='failed', detail=error)
            failure = error
    elif status.state == "failed":
        # Without a canonical worker envelope there is no semantic evidence
        # against the agent. Unknown exits/BackoffLimitExceeded are organizer
        # infrastructure failures, just like an explicitly diagnosed OOM.
        failure = {"category": "infrastructure_failed", "reason": "worker_exited_without_result"}
        raw = backend._client.read_job_status(backend.config.namespace, current.job_id)
        if raw.get("reason") in {"DeadlineExceeded", "Evicted", "NodeLost"}:
            failure = {"category": "infrastructure_failed", "reason": raw["reason"]}
        elif hasattr(backend._client, "read_job_pod_states"):
            states = backend._client.read_job_pod_states(backend.config.namespace, current.job_id)
            causes = [row["reason"] for row in states if row.get("reason") in {"OOMKilled", "NodeLost", "Evicted"}]
            if causes:
                failure = {"category": "infrastructure_failed", "reason": causes[0]}
    elif status.state == "succeeded":
        failure = {"category": "infrastructure_failed", "reason": "worker_exited_without_result"}
    elif hasattr(backend.result_store, "get_job_progress"):
        progress = backend.result_store.get_job_progress(current.run_id, current.job_id)
        if progress and heartbeat_stalled(backend, current, progress):
            failure = {"category": "infrastructure_failed", "reason": "observation_supervisor_stalled"}
            if not stop_with_evidence(backend, current, failure, progress):
                return poll(backend,original)
            already_stopped = True
        elif progress is None:
            failure = scheduling_failure(backend, current)
            if failure and not stop_with_evidence(backend, current, failure):
                return poll(backend,original)
            already_stopped = failure is not None
    if failure is None:
        return status
    return recover(backend,original,current,failure,already_stopped=already_stopped)
