"""Scheduling/startup liveness is infrastructure protection, not compute budget."""
import time


def scheduling_failure(backend, handle, *, now=None):
    reader = getattr(backend._client, "read_job_runtime", None)
    if reader is None:
        return None
    now = time.monotonic() if now is None else now
    states = reader(backend.config.namespace, handle.job_id)
    clocks = backend.__dict__.setdefault("_observation_startup_clocks", {})
    signature = tuple(sorted((row.get("phase", ""), tuple(sorted(row.get("waiting_reasons", [])))) for row in states))
    clock = clocks.setdefault(handle.job_id, {"first_seen": now, "last_progress": now, "signature": signature})
    if clock["signature"] != signature:
        clock.update(signature=signature, last_progress=now)
    permanent = {"ErrImagePull", "ImagePullBackOff", "InvalidImageName", "CreateContainerConfigError",
                 "CreateContainerError", "CrashLoopBackOff", "RunContainerError"}
    if any(permanent.intersection(row.get("waiting_reasons", [])) for row in states):
        return {"category": "infrastructure_failed", "reason": "observation_pod_startup_failed", "pod_states": states}
    if now-clock["last_progress"] >= backend.config.observation_scheduling_idle_seconds:
        return {"category": "infrastructure_failed", "reason": "observation_scheduling_stalled", "pod_states": states}
    return None


def heartbeat_stalled(backend, handle, progress, *, now=None):
    """Use observed sequence changes and local monotonic time, never node UTC."""
    now = time.monotonic() if now is None else now
    clocks = backend.__dict__.setdefault("_observation_receipt_clocks", {})
    sequence = progress.get("sequence")
    old = clocks.setdefault(handle.job_id, {"sequence": sequence, "seen": now})
    if old["sequence"] != sequence:
        old.update(sequence=sequence, seen=now)
    return now-old["seen"] > backend.config.observation_idle_seconds + 60


def stop_with_evidence(backend, handle, failure, progress=None):
    logs, log_error = "", None
    try:
        logs = backend._client.read_job_logs(backend.config.namespace, handle.job_id, container="worker")
    except Exception as error:
        # A container waiting for its image has no logs (the API returns 400).
        # Diagnostic collection must not prevent stopping the stale Job.
        status = getattr(error, "status", None)
        log_error = {"category": "logs_unavailable", "type": type(error).__name__,
                     "http_status": status if type(status) is int else None}
    if backend.result_store.stat(handle.result_uri):
        return False
    diagnostic_id = handle.job_id + "-watchdog-diagnostics"
    uri = backend.result_store.job_result_uri(handle.run_id, diagnostic_id)
    if not backend.result_store.stat(uri):
        backend.result_store.put_job_result(handle.run_id, diagnostic_id,
            {"status": "diagnostic", "failure": failure, "progress": progress,
             "log_tail": logs, "log_error": log_error})
    if backend.result_store.stat(handle.result_uri):
        return False
    backend._client.stop_job(backend.config.namespace, handle.job_id)
    return True
