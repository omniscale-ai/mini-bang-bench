"""Observation envelopes on the existing immutable result-store contract."""
from __future__ import annotations

import os

from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.payloads import create_payload_storage_from_context
from mini_bang.execution.result_store import result_store_from_config
from mini_bang.execution.transport import request_from_transport, result_to_transport
from mini_bang.pool.protocol import validate_name
from mini_bang.worker import k8s_worker as worker
from mini_bang.worker.storage_retry import commit_prepared, retry_storage, retrying_store


def work_store(work):
    payload = work["payload"]
    manifest, config = payload["manifest"], payload["result_store_config"]
    if not isinstance(manifest, dict) or not isinstance(config, dict):
        raise ValueError("Worker needs an immutable manifest and result-store configuration")
    validate_name(manifest["run_id"])
    validate_name(manifest["job_id"])
    store = retrying_store(result_store_from_config(config), manifest)
    expected = store.job_result_uri(manifest["run_id"], manifest["job_id"])
    if manifest.get("result_uri") != expected:
        raise ValueError("Worker result URI differs from immutable work identity")
    return manifest, store


def retained_result(work):
    """A committed result wins a race with worker death or a lost IPC reply."""
    manifest, store = work_store(work)
    # A second work_id does not grant permission to reuse this run/job identity
    # for different scientific inputs, even if an old result already exists.
    retry_storage(lambda: store.put_job_input(manifest["run_id"], manifest["job_id"], manifest),
                  idle_seconds=store.idle_seconds)
    if not store.stat(manifest["result_uri"]):
        return None
    envelope = store.get_job_result(manifest["result_uri"])
    if (envelope.get("status") not in {"succeeded", "failed"} or
            any(str(envelope.get(key)) != str(manifest[key]) for key in ("job_id", "run_id"))):
        raise OSError("Retained worker result identity differs")
    return {"status": envelope["status"], "result_uri": manifest["result_uri"],
            "retained": True}


def prepare_work(work):
    manifest, store = work_store(work)
    # Never replace an input manifest. Retrying an uncertain PUT uses the same
    # captured body and the existing ResultStore rejects conflicting writes.
    retry_storage(lambda: store.put_job_input(manifest["run_id"], manifest["job_id"], manifest),
                  idle_seconds=store.idle_seconds)
    return manifest, store


def process(work):
    from mini_bang.worker.observation_startup import retained_outcome

    manifest, store = prepare_work(work)
    cached = retained_outcome(store, manifest)
    if cached is not None:
        return {**retained_result(work), "worker_pid": os.getpid()}
    created_at = worker._now()
    executor = None
    try:
        from mini_bang.worker.runtime_calibration import startup_admission
        startup_admission(manifest, store)
        context = worker._runtime_context_from_dict(manifest["runtime_context"])
        requests = [request_from_transport(item) for item in manifest["requests"]]
        executor = SimulationExecutor(payload_storage=create_payload_storage_from_context(context),
                                      context=context)
        results = executor.run_batch(requests, context=context)
        prepared = {"job_id": manifest["job_id"], "run_id": manifest["run_id"],
            "task_id": manifest.get("task_id"), "instance_id": manifest.get("instance_id"),
            "status": "succeeded", "results": [result_to_transport(value) for value in results],
            "trace": worker._combined_trace(results), "resource_usage": worker._combined_usage(results),
            "error": None, "created_at": created_at, "finished_at": worker._now()}
        commit_prepared(store, manifest, prepared)
    except Exception as error:
        worker._write_failure(store, manifest, created_at, error)
        result = retained_result(work)
        if result is None:
            raise
        return {**result, "worker_pid": os.getpid(), "retained": False}
    finally:
        if executor is not None:
            executor.close()
    return {**retained_result(work), "worker_pid": os.getpid(), "retained": False}
