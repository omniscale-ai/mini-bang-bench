from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from mini_bang.backends.base import JobStatus
from mini_bang.backends.k8s import (
    K8sBackendConfig,
    KubernetesJobClient,
    _k8s_resources,
    _runtime_storage_env,
    _storage_env,
    _ttl_spec,
)
from mini_bang.execution.resources import ResourceSpec
from mini_bang.execution.result_store import ResultStore, result_store_from_config
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.framework.records import RunRecord, to_serializable
from mini_bang.framework.task import TaskSubmission


@dataclass(frozen=True)
class ScoringJob:
    run_id: str
    task_id: str
    instance_id: str
    provisional_record: RunRecord | dict[str, Any]
    submission: TaskSubmission | dict[str, Any]
    gold: dict[str, Any]
    scorer_class_path: str
    task_spec: dict[str, Any]
    instance_spec: dict[str, Any]
    scoring_context: RuntimeContext
    executable_runner_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self, *, job_id: str, result_uri: str) -> dict[str, Any]:
        return {
            "kind": "mini_bang.scoring_job",
            "job_id": job_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "instance_id": self.instance_id,
            "provisional_record": _record_dict(self.provisional_record),
            "submission": _submission_dict(self.submission),
            "gold": to_serializable(self.gold),
            "scorer_class_path": self.scorer_class_path,
            "task_spec": to_serializable(self.task_spec),
            "instance_spec": to_serializable(self.instance_spec),
            "scoring_context": self.scoring_context.to_dict(),
            "executable_runner_config": self.executable_runner_config,
            "result_uri": result_uri,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScoringJobHandle:
    backend: str
    job_id: str
    run_id: str
    task_id: str
    instance_id: str
    result_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ScoringJobBackend(Protocol):
    def submit(self, job: ScoringJob) -> ScoringJobHandle:
        ...

    def poll(self, handle: ScoringJobHandle) -> JobStatus:
        ...

    def fetch(self, handle: ScoringJobHandle) -> dict[str, Any]:
        ...


class K8sScoringJobBackend:
    """Fixed warm scoring slots, with explicit historical Job recovery."""

    def __init__(
        self,
        *,
        config: K8sBackendConfig,
        result_store: ResultStore | None = None,
        k8s_client: Any | None = None,
        resources: ResourceSpec | None = None,
        use_k8s_for_hidden_simulations: bool = False,
    ) -> None:
        self.config = config
        self.result_store = result_store or result_store_from_config(
            config.result_store_config or {"kind": "filesystem"}
        )
        self._client = k8s_client or (KubernetesJobClient() if config.worker_mode == "legacy_jobs" else None)
        from .pool_scoring import PoolScoringTransport
        self._pool = PoolScoringTransport(self)
        self.resources = resources or config.default_resources
        self.use_k8s_for_hidden_simulations = use_k8s_for_hidden_simulations

    def submit(self, job: ScoringJob) -> ScoringJobHandle:
        if self.config.worker_mode == "pool":
            return self._pool.submit(job)
        job_id = f"score-{uuid4().hex[:20]}"
        result_uri = self.result_store.job_result_uri(job.run_id, job_id)
        input_uri = self.result_store.put_job_input(
            job.run_id,
            job_id,
            self._job_input_manifest(job, job_id=job_id, result_uri=result_uri),
        )
        self._client.create_job(
            self.config.namespace,
            self._job_manifest(job_id, input_uri, job.scoring_context),
        )
        return ScoringJobHandle(
            backend="k8s-scoring",
            job_id=job_id,
            run_id=job.run_id,
            task_id=job.task_id,
            instance_id=job.instance_id,
            result_uri=result_uri,
            metadata={"input_uri": input_uri, "namespace": self.config.namespace},
        )

    def poll(self, handle: ScoringJobHandle) -> JobStatus:
        if handle.metadata.get("execution_mode") == "persistent_pool":
            return self._pool.poll(handle)
        if self.result_store.stat(handle.result_uri):
            envelope = self.result_store.get_job_result(handle.result_uri)
            return JobStatus(state=str(envelope.get("status")), detail=envelope.get("error"))
        if self._client is None:
            self._client = KubernetesJobClient()  # Retained legacy scoring Job.
        status = self._client.read_job_status(self.config.namespace, handle.job_id)
        if status.get("succeeded"):
            return JobStatus(state="succeeded")
        if status.get("failed"):
            return JobStatus(state="failed", detail=status.get("message"))
        return JobStatus(state="running" if status.get("active") else "pending")

    def fetch(self, handle: ScoringJobHandle) -> dict[str, Any]:
        envelope = self.result_store.get_job_result(handle.result_uri)
        if envelope.get("status") != "succeeded":
            raise RuntimeError(f"scoring job failed: {envelope.get('error')}")
        return envelope

    def run(self, job: ScoringJob) -> dict[str, Any]:
        directory = os.getenv('MINI_BANG_SCORING_RECEIPTS_DIR')
        if directory:
            from mini_bang.framework.scoring_recovery import run_durable
            return run_durable(self, job, directory)
        handle = self.submit(job)
        # Scoring can wait for many successive waves of hidden observations.
        # Individual executable sandboxes enforce their own execution limits.
        while True:
            status = self.poll(handle)
            if status.state == "succeeded":
                return self.fetch(handle)
            if status.state == "failed":
                raise RuntimeError(status.detail or f"scoring job {handle.job_id} failed")
            time.sleep(self.config.polling_interval_seconds)

    def _job_input_manifest(
        self,
        job: ScoringJob,
        *,
        job_id: str,
        result_uri: str,
    ) -> dict[str, Any]:
        manifest = job.to_manifest(job_id=job_id, result_uri=result_uri)
        manifest["resource_spec"] = self.resources.to_dict()
        if manifest.get("executable_runner_config") is None:
            manifest["executable_runner_config"] = {
                "namespace": self.config.namespace,
                "image": self.config.executable_image or "python:3.13-slim",
                "service_account": self.config.executable_service_account,
                "result_store_config": self.config.worker_result_store_config
                or self.config.result_store_config,
                "worker_result_store_config": self.config.worker_result_store_config,
                "image_pull_policy": self.config.image_pull_policy,
                "ttl_seconds_after_finished": self.config.ttl_seconds_after_finished,
                "polling_interval_seconds": self.config.polling_interval_seconds,
                "staging_image": job.scoring_context.env.get("MINI_BANG_K8S_STAGER_IMAGE"),
                "staging_storage_class": job.scoring_context.env.get("MINI_BANG_K8S_STAGING_STORAGE_CLASS"),
            }
        if self.config.worker_mode == "pool":
            manifest["executable_runner_config"] = {**manifest["executable_runner_config"],
                "execution_mode": "pool", "pool_url": self.config.worker_pool_url or self.config.pool_url}
        if self.use_k8s_for_hidden_simulations:
            manifest["scoring_simulation_backend"] = {
                "kind": "k8s",
                "config": {
                    "namespace": self.config.namespace,
                    "worker_image": self.config.worker_image,
                    "service_account": self.config.service_account,
                    "job_timeout_seconds": self.config.job_timeout_seconds,
                    "polling_interval_seconds": self.config.polling_interval_seconds,
                    "image_pull_policy": self.config.image_pull_policy,
                    "ttl_seconds_after_finished": self.config.ttl_seconds_after_finished,
                    "result_store_config": self.config.result_store_config,
                    "worker_result_store_config": self.config.worker_result_store_config,
                    "default_resources": self.config.default_resources.to_dict(),
                    "max_parallel_jobs": self.config.max_parallel_jobs,
                    "batch_chunk_size": self.config.batch_chunk_size,
                    "request_runs_policy": self.config.request_runs_policy,
                    "gpu_worker_image": self.config.gpu_worker_image,
                    "gpu_runtime_class": self.config.gpu_runtime_class,
                    "gpu_node_selector": self.config.gpu_node_selector,
                    "observation_idle_seconds": self.config.observation_idle_seconds,
                    "observation_max_retries": self.config.observation_max_retries,
                    "observation_cpu_limit_seconds": self.config.observation_cpu_limit_seconds,
                    "observation_scheduling_idle_seconds": self.config.observation_scheduling_idle_seconds,
                    "runtime_capacity": self.config.runtime_capacity,
                    "worker_mode": self.config.worker_mode,
                    "pool_url": self.config.worker_pool_url or self.config.pool_url,
                    "worker_pool_url": self.config.worker_pool_url,
                },
            }
        return manifest

    def _job_manifest(self, job_id: str, input_uri: str, context: RuntimeContext) -> dict[str, Any]:
        env = [
            {"name": "MINI_BANG_K8S_JOB_INPUT_URI", "value": input_uri},
            {
                "name": "MINI_BANG_K8S_RESULT_STORE_CONFIG",
                "value": json.dumps(self.config.worker_result_store_config or self.result_store.describe()),
            },
        ]
        env.extend(_storage_env(self.config.worker_result_store_config or {}))
        env.extend(_runtime_storage_env(context))
        env.extend({"name": key, "value": value} for key, value in context.env.items())
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_id, "labels": {"app": "mini-bang-scoring-worker"}},
            "spec": {
                "backoffLimit": 0,
                **_ttl_spec(self.config.ttl_seconds_after_finished),
                "template": {
                    "metadata": {"labels": {"app": "mini-bang-scoring-worker"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "serviceAccountName": self.config.service_account,
                        "containers": [
                            {
                                "name": "scoring-worker",
                                "image": self.config.worker_image,
                                "imagePullPolicy": self.config.image_pull_policy,
                                "command": ["python", "-m", "mini_bang.worker.k8s_scoring_worker"],
                                "env": env,
                                "resources": _k8s_resources(self.resources),
                            }
                        ],
                    },
                },
            },
        }


def _record_dict(record: RunRecord | dict[str, Any]) -> dict[str, Any]:
    return record.to_dict() if isinstance(record, RunRecord) else to_serializable(record)


def _submission_dict(submission: TaskSubmission | dict[str, Any]) -> dict[str, Any]:
    if isinstance(submission, TaskSubmission):
        return {"answer": submission.answer, "artifacts": submission.artifacts}
    return to_serializable(submission)


def task_spec_to_manifest(task_spec) -> dict[str, Any]:
    data = dataclasses.asdict(task_spec)
    data["package_dir"] = str(data["package_dir"]) if data.get("package_dir") else None
    return to_serializable(data)


def instance_spec_to_manifest(instance_spec) -> dict[str, Any]:
    data = dataclasses.asdict(instance_spec)
    for key in ("gold_path", "config_path"):
        data[key] = str(data[key]) if data.get(key) else None
    return to_serializable(data)
