"""Submit private scorer work to a fixed warm scoring slot."""
from __future__ import annotations

from mini_bang.backends.base import JobStatus
from mini_bang.pool.client import PoolClient
from mini_bang.pool.protocol import PoolUnavailable, VERSION, validate_name


class PoolScoringTransport:
    def __init__(self, backend):
        self.backend = backend
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = PoolClient(self.backend.config.pool_url)
        return self._client

    def submit(self, job):
        from .scoring_jobs import ScoringJobHandle
        import re
        task_label = re.sub(r"[^a-z0-9-]+", "-", job.task_id.lower()).strip("-")[:48]
        job_id = validate_name(f"scoring--{task_label}--run-{job.run_id}")
        store = self.backend.result_store
        result_uri = store.job_result_uri(job.run_id, job_id)
        manifest = self.backend._job_input_manifest(job, job_id=job_id, result_uri=result_uri)
        manifest["metadata"].update(execution_mode="persistent_pool")
        input_location = getattr(store, "job_input_uri", None)
        if callable(input_location):
            uri = input_location(job.run_id, job_id)
            if store.stat(uri):
                original = store.get_job_input(uri)
                for key in ("run_id", "task_id", "instance_id", "submission", "gold", "scorer_class_path",
                            "task_spec", "instance_spec", "scoring_context", "executable_runner_config", "resource_spec"):
                    if original.get(key) != manifest.get(key):
                        raise ValueError("Retained scoring scientific inputs differ: " + key)
                # Restart timing is not a new scientific input or a new paid score.
                manifest = original
        input_uri = store.put_job_input(job.run_id, job_id, manifest)
        handle = ScoringJobHandle(backend="k8s-scoring", job_id=job_id, run_id=job.run_id,
            task_id=job.task_id, instance_id=job.instance_id, result_uri=result_uri,
            metadata={"execution_mode": "persistent_pool", "input_uri": input_uri,
                      "pool_url": self.backend.config.pool_url, "namespace": self.backend.config.namespace})
        # The immutable input can always reconstruct the same accepted work.
        self.client.submit(self.work(handle, manifest))
        return handle

    def work(self, handle, manifest=None):
        return {"version": VERSION, "work_id": handle.job_id, "role": "scoring", "payload": {
            "manifest": manifest or self.backend.result_store.get_job_input(handle.metadata["input_uri"]),
            "result_store_config": self.backend.config.worker_result_store_config
            or self.backend.result_store.describe()}}

    def poll(self, handle):
        if self.backend.result_store.stat(handle.result_uri):
            envelope = self.backend.result_store.get_job_result(handle.result_uri)
            return JobStatus(str(envelope["status"]), envelope.get("error"))
        try:
            row = self.client.status(handle.job_id)
        except PoolUnavailable:
            self.client.submit(self.work(handle))
            row = self.client.status(handle.job_id)
        state = row["state"]
        return JobStatus(state if state in {"pending", "succeeded", "failed"} else "running", row.get("error"))
