"""K8s simulation transport over a fixed warm pool, with existing result refs."""
from __future__ import annotations

import time

from mini_bang.backends.base import CompositeJobHandle, JobHandle, JobStatus
from mini_bang.execution.progress import report_progress
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.transport import result_from_transport
from mini_bang.pool.client import PoolClient
from mini_bang.pool.protocol import PoolBackpressure, PoolUnavailable, VERSION, work_name

from .observation_wait import infrastructure_error, raise_failure


def is_pool_handle(handle):
    return handle.metadata.get("execution_mode") == "persistent_pool"


class PoolSimulationTransport:
    def __init__(self, backend):
        self.backend = backend
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = PoolClient(self.backend.config.pool_url,
                                      poll_seconds=self.backend.config.polling_interval_seconds)
        return self._client

    def submit_batch(self, requests, *, context=None, services=None):
        if services is not None:
            raise ValueError("Persistent K8s work requires serializable RuntimeContext")
        requests = list(requests)
        if not requests:
            raise ValueError("Cannot submit an empty pool batch")
        self.backend._validate_request_runs(requests)
        context = context or RuntimeContext.from_env()
        chunks = list(self.backend._request_chunks(requests))
        prepared = [self._prepare(chunk, indices, context) for indices, chunk in chunks]
        if len(prepared) == 1:
            handle = prepared[0][0]
        else:
            handle = CompositeJobHandle(backend="k8s", job_id=work_name("observation", context.task_id),
                run_id=context.run_id, task_id=context.task_id, instance_id=context.instance_id,
                child_handles=tuple(h for h, _ in prepared), child_indexes=tuple(i for i, _ in chunks),
                metadata={"execution_mode": "persistent_pool", "batch_size": len(requests),
                          "pool_url": self.backend.config.pool_url})
        from dataclasses import asdict
        # Save the complete paid batch identity before admitting any child.
        report_progress("backend_job", backend_kind="k8s_handle", job_id=handle.job_id,
            run_id=handle.run_id, handle={"kind": "composite" if len(prepared) > 1 else "job", "value": asdict(handle)})
        admitted = []
        for child, work in prepared:
            self._admit(work, handle, admitted)
            admitted.append(child)
        return handle

    def _admit(self, work, original, admitted):
        last_progress, signatures, index = time.monotonic(), {}, 0
        while True:
            try:
                self.client.submit(work)
                return
            except PoolBackpressure as error:
                # A healthy full queue did not admit this child. Wait on already
                # accepted siblings; never create a replacement work identity.
                if admitted:
                    child = admitted[index % len(admitted)]
                    index += 1
                    try:
                        row = self.client.status(child.job_id, wait_seconds=1)
                        import json
                        signature = json.dumps({k: row.get(k) for k in ("state", "progress")}, sort_keys=True)
                        if signatures.get(child.job_id) != signature:
                            signatures[child.job_id] = signature
                            last_progress = time.monotonic()
                    except PoolUnavailable:
                        pass
                if time.monotonic()-last_progress >= self.backend.config.observation_idle_seconds:
                    raise infrastructure_error("Fixed pool admission made no progress; all original handles retained", original) from error
                time.sleep(max(.01, min(.25, self.backend.config.polling_interval_seconds)))
            except PoolUnavailable as error:
                raise infrastructure_error("Persistent pool admission unavailable; original handles retained", original) from error
            except ValueError as error:
                raise infrastructure_error("Fixed pool cannot admit this work; original handles retained", original) from error

    def _prepare(self, requests, indices, context):
        job_id = work_name("observation", context.task_id)
        run_id = context.run_id or "unassigned-local-run"
        store = self.backend.result_store
        result_uri = store.job_result_uri(run_id, job_id)
        spec = self.backend._resource_spec(requests)
        manifest = self.backend._job_input_manifest(job_id=job_id, run_id=run_id,
            requests=requests, context=context, result_uri=result_uri, resource_spec=spec)
        manifest["metadata"].update(execution_mode="persistent_pool")
        input_uri = store.put_job_input(run_id, job_id, manifest)
        handle = JobHandle(backend="k8s", job_id=job_id, run_id=run_id,
            task_id=context.task_id, instance_id=context.instance_id, result_uri=result_uri,
            metadata={"execution_mode": "persistent_pool", "input_uri": input_uri,
                      "pool_url": self.backend.config.pool_url, "batch_size": len(requests),
                      "request_indices": indices, "resource_spec": spec.to_dict()})
        return handle, self._work(handle, manifest)

    def _work(self, handle, manifest=None):
        manifest = manifest or self.backend.result_store.get_job_input(handle.metadata["input_uri"])
        return {"version": VERSION, "work_id": handle.job_id, "role": "observation",
                "payload": {"manifest": manifest,
                            "result_store_config": self.backend.config.worker_result_store_config
                            or self.backend.result_store.describe()}}

    def poll(self, handle):
        if isinstance(handle, CompositeJobHandle):
            statuses = [self.poll(child) for child in handle.child_handles]
            failed = next((s for s in statuses if s.state == "failed"), None)
            if failed:
                return failed
            if all(s.state == "succeeded" for s in statuses):
                return JobStatus("succeeded")
            return JobStatus("running" if any(s.state == "running" for s in statuses) else "pending")
        store = self.backend.result_store
        if handle.result_uri and store.stat(handle.result_uri):
            row = store.get_job_result(handle.result_uri)
            return JobStatus(str(row["status"]), row.get("error"))
        try:
            row = self.client.status(handle.job_id)
        except PoolUnavailable:
            # A saved-but-not-admitted batch child uses its original identity.
            self.client.submit(self._work(handle))
            row = self.client.status(handle.job_id)
        state = row["state"]
        return JobStatus(state if state in {"pending", "succeeded", "failed"} else "running", row.get("error"))

    def fetch(self, handle):
        if isinstance(handle, CompositeJobHandle):
            output = [None] * handle.metadata["batch_size"]
            for indices, child in zip(handle.child_indexes, handle.child_handles):
                fetched = self.fetch(child)
                values = fetched if isinstance(fetched, list) else [fetched]
                if len(values) != len(indices):
                    raise infrastructure_error("Warm worker returned an invalid result count", handle)
                for i, value in zip(indices, values):
                    output[i] = value
            return output
        envelope = self.backend.result_store.get_job_result(handle.result_uri)
        if envelope.get("status") != "succeeded":
            raise_failure(envelope.get("error"), handle)
        results = [result_from_transport(value) for value in envelope.get("results", [])]
        if len(results) != handle.metadata["batch_size"]:
            raise infrastructure_error("Warm worker result manifest has the wrong batch size", handle)
        return results[0] if len(results) == 1 else results

    def run_batch(self, requests, *, context=None, services=None):
        requests = list(requests)
        if not requests:
            return []
        handle = self.submit_batch(requests, context=context, services=services)
        last_contact = time.monotonic()
        while True:
            try:
                status = self.poll(handle)
                last_contact = time.monotonic()
                if status.state == "failed":
                    raise_failure(status.detail, handle)
                if status.state == "succeeded":
                    result = self.fetch(handle)
                    return result if isinstance(result, list) else [result]
                children = handle.child_handles if isinstance(handle, CompositeJobHandle) else (handle,)
                for child in children:
                    if self.poll(child).state not in {"succeeded", "failed"}:
                        self.client.status(child.job_id, wait_seconds=1)
                        break
            except PoolUnavailable as error:
                if time.monotonic() - last_contact >= self.backend.config.observation_idle_seconds:
                    raise infrastructure_error("Persistent pool transport lost contact", handle) from error
                time.sleep(max(.01, min(.25, self.backend.config.polling_interval_seconds)))
