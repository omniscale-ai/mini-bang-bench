from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable
from uuid import uuid4

from . import k8s_composite, observation_recovery, observation_wait
from .observation_launch import VERSION as LAUNCH_TRACKING_VERSION
from .k8s_config import K8sBackendConfig
from mini_bang.backends.base import CompositeJobHandle, JobHandle, JobStatus, SimulationBackend
from mini_bang.backends.k8s_client import KubernetesJobClient, OBSERVATION_TERMINATION_GRACE_SECONDS
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.resources import ResourceSpec
from mini_bang.execution.result_store import ResultStore, result_store_from_config, result_store_from_env
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.execution.transport import request_to_transport, result_from_transport
from mini_bang.simulators.registry import get_simulator_entry


class K8sBackend(SimulationBackend):
    """Fixed warm Kubernetes pools, plus recovery of explicit historical Jobs."""

    def __init__(
        self,
        *,
        config: K8sBackendConfig,
        result_store: ResultStore | None = None,
        k8s_client: Any | None = None,
    ) -> None:
        self.config = config
        self.result_store = result_store or result_store_from_config(
            config.result_store_config or {"kind": "filesystem"}
        )
        if config.worker_mode not in {"pool", "legacy_jobs"}:
            raise ValueError("K8s worker_mode must be pool or explicit legacy_jobs")
        self._client = k8s_client or (KubernetesJobClient() if config.worker_mode == "legacy_jobs" else None)
        from .persistent_pool import PoolSimulationTransport
        self._pool = PoolSimulationTransport(self)
        self._composites: dict[str, k8s_composite.CompositeState] = {}

    @classmethod
    def from_env(cls) -> "K8sBackend":
        store = result_store_from_env()
        result_config = store.describe()
        worker_result_config = dict(result_config)
        worker_endpoint = os.getenv("MINI_BANG_K8S_WORKER_RESULT_S3_ENDPOINT_URL")
        if worker_endpoint:
            worker_result_config["endpoint_url"] = worker_endpoint
        _add_worker_s3_credentials(worker_result_config)
        timeout = int(os.getenv("MINI_BANG_K8S_JOB_TIMEOUT_SECONDS", "600"))
        config = K8sBackendConfig(
            namespace=os.getenv("MINI_BANG_K8S_NAMESPACE", "mini-bang"),
            worker_image=os.getenv("MINI_BANG_K8S_WORKER_IMAGE", "mini-bang-worker:latest"),
            service_account=os.getenv("MINI_BANG_K8S_SERVICE_ACCOUNT", "mini-bang-executor"),
            executable_image=os.getenv("MINI_BANG_K8S_EXECUTABLE_IMAGE") or "python:3.13-slim",
            executable_service_account=os.getenv(
                "MINI_BANG_K8S_EXECUTABLE_SERVICE_ACCOUNT",
                "mini-bang-executable",
            ),
            job_timeout_seconds=timeout,
            polling_interval_seconds=float(os.getenv("MINI_BANG_K8S_POLL_INTERVAL", "1")),
            image_pull_policy=os.getenv("MINI_BANG_K8S_IMAGE_PULL_POLICY", "IfNotPresent"),
            ttl_seconds_after_finished=_ttl_from_env(),
            result_store_config=result_config,
            worker_result_store_config=worker_result_config,
            default_resources=ResourceSpec.from_dict(_env_resources(timeout)),
            max_parallel_jobs=_optional_int(os.getenv("MINI_BANG_K8S_MAX_PARALLEL_JOBS")),
            batch_chunk_size=max(1, int(os.getenv("MINI_BANG_K8S_BATCH_CHUNK_SIZE", "1"))),
            request_runs_policy=os.getenv("MINI_BANG_K8S_REQUEST_RUNS_POLICY", "reject"),
            gpu_worker_image=os.getenv("MINI_BANG_K8S_GPU_WORKER_IMAGE"),
            gpu_runtime_class=os.getenv("MINI_BANG_K8S_GPU_RUNTIME_CLASS"),
            gpu_node_selector=json.loads(os.getenv("MINI_BANG_K8S_GPU_NODE_SELECTOR", "{}")),
            observation_idle_seconds=float(os.getenv("MINI_BANG_OBSERVATION_IDLE_SECONDS", "600")),
            observation_max_retries=int(os.getenv("MINI_BANG_OBSERVATION_MAX_RETRIES", "2")),
            observation_cpu_limit_seconds=float(value) if (value := os.getenv("MINI_BANG_OBSERVATION_CPU_SECONDS")) else None,
            emergency_job_timeout_seconds=_optional_int(os.getenv("MINI_BANG_K8S_EMERGENCY_JOB_TIMEOUT_SECONDS")),
            observation_scheduling_idle_seconds=float(os.getenv("MINI_BANG_OBSERVATION_SCHEDULING_IDLE_SECONDS", "600")),
            runtime_capacity=observation_recovery.capacity_from_env(),
            worker_mode=os.getenv("MINI_BANG_K8S_WORKER_MODE", "pool"),
            pool_url=os.getenv("MINI_BANG_POOL_URL", "http://mini-bang-pool:8770"),
            worker_pool_url=os.getenv("MINI_BANG_WORKER_POOL_URL") or None,
        )
        return cls(config=config, result_store=store)

    def run(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        return self.run_batch([request], context=context, services=services)[0]

    def run_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        if self.config.worker_mode == "pool":
            return self._pool.run_batch(requests, context=context, services=services)
        request_list = list(requests)
        if not request_list:
            return []
        handle = self.submit_batch(request_list, context=context, services=services)
        fetched = self._wait_and_fetch(handle)
        results = fetched if isinstance(fetched, list) else [fetched]
        if len(results) != len(request_list):
            raise RuntimeError(
                f"Kubernetes batch returned {len(results)} results for {len(request_list)} requests"
            )
        return results

    def submit(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        if self.config.worker_mode == "pool":
            return self._pool.submit_batch([request], context=context, services=services)
        if services is not None:
            raise ValueError("K8sBackend cannot serialize live RuntimeServices")
        self._validate_request_runs([request])
        return self._submit_job([request], context=context or RuntimeContext.from_env())

    def submit_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        if self.config.worker_mode == "pool":
            return self._pool.submit_batch(requests, context=context, services=services)
        if services is not None:
            raise ValueError("K8sBackend cannot serialize live RuntimeServices")
        request_list = list(requests)
        if not request_list:
            raise ValueError("submit_batch requires at least one request")
        self._validate_request_runs(request_list)
        effective_context = context or RuntimeContext.from_env()
        chunks = list(self._request_chunks(request_list))
        if len(chunks) == 1:
            return self._submit_job(chunks[0][1], context=effective_context)
        return k8s_composite.submit_batch(self, request_list, effective_context)

    def _submit_job(
        self,
        requests: list[SimulationRequest],
        *,
        context: RuntimeContext,
    ) -> JobHandle:
        handle, manifest = self._prepare_job(requests, context=context)
        from mini_bang.execution.progress import report_progress
        from dataclasses import asdict
        report_progress('backend_job', job_id=handle.job_id, result_uri=handle.result_uri,
                        run_id=handle.run_id, batch_size=handle.metadata.get('batch_size'),
                        backend_kind='k8s_handle', handle={'kind': 'job', 'value': asdict(handle)})
        observation_wait.create_or_rejoin(self, manifest, handle)
        return handle

    def _prepare_job(
        self,
        requests: list[SimulationRequest],
        *,
        context: RuntimeContext,
        run_id: str | None = None,
        request_indices: list[int] | None = None,
    ) -> tuple[JobHandle, dict[str, Any]]:
        request_list = list(requests)
        effective_context = context
        run_id = run_id or effective_context.run_id or f"k8s-run-{uuid4().hex[:12]}"
        job_id = f"k8s-{uuid4().hex[:20]}"
        output_uri = self.result_store.job_result_uri(run_id, job_id)
        resource_spec = self._resource_spec(request_list)
        input_uri = self.result_store.put_job_input(
            run_id,
            job_id,
            self._job_input_manifest(
                job_id=job_id,
                run_id=run_id,
                requests=request_list,
                context=effective_context,
                result_uri=output_uri,
                resource_spec=resource_spec,
            ),
        )
        body = self._job_manifest(job_id, input_uri, resource_spec, effective_context)
        handle = JobHandle(
            backend="k8s",
            job_id=job_id,
            run_id=run_id,
            task_id=effective_context.task_id,
            instance_id=effective_context.instance_id,
            result_uri=output_uri,
            metadata={
                "input_uri": input_uri,
                "batch_size": len(request_list),
                "namespace": self.config.namespace,
                "job_name": job_id,
                "request_indices": list(request_indices or range(len(request_list))),
                "resource_spec": resource_spec.to_dict(),
            },
        )
        return handle, body

    def poll(self, handle: JobHandle) -> JobStatus:
        from .persistent_pool import is_pool_handle
        if is_pool_handle(handle):
            return self._pool.poll(handle)
        if self._client is None:
            self._client = KubernetesJobClient()  # Recovery of retained historical Jobs only.
        if isinstance(handle, CompositeJobHandle):
            return k8s_composite.poll(self, handle)
        return self._poll_single(handle)

    def _poll_single(self, handle: JobHandle) -> JobStatus:
        return observation_recovery.poll(self, handle)

    def _poll_single_raw(self, handle: JobHandle) -> JobStatus:
        if handle.result_uri and self.result_store.stat(handle.result_uri):
            envelope = self.result_store.get_job_result(handle.result_uri)
            status = str(envelope.get("status", "unknown"))
            return JobStatus(state=status, detail=envelope.get("error"))
        status = self._client.read_job_status(self.config.namespace, handle.job_id)
        if status.get("succeeded"):
            return JobStatus(state="succeeded")
        if status.get("failed"):
            return JobStatus(state="failed", detail=status.get("message"))
        if status.get("active"):
            return JobStatus(state="running")
        return JobStatus(state="pending")

    def fetch(self, handle: JobHandle) -> SimulationResult | list[SimulationResult]:
        from .persistent_pool import is_pool_handle
        if is_pool_handle(handle):
            return self._pool.fetch(handle)
        if isinstance(handle, CompositeJobHandle):
            return observation_wait.fetch_composite(self, handle)
        original = handle
        handle = observation_recovery.current_handle(self, handle)
        if not handle.result_uri:
            raise ValueError("K8s JobHandle has no result_uri")
        envelope = observation_wait.result_envelope(self, handle, original)
        if envelope.get("status") != "succeeded":
            observation_wait.raise_failure(envelope.get('error'), original)
        results = [result_from_transport(item) for item in envelope.get("results", [])]
        if int(handle.metadata.get("batch_size", len(results))) == 1:
            return results[0]
        return results

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "k8s",
            "namespace": self.config.namespace,
            "worker_image": self.config.worker_image,
            "result_store": self.result_store.describe(),
            "batch_chunk_size": self.config.batch_chunk_size,
            "max_parallel_jobs": self.config.max_parallel_jobs,
            "request_runs_policy": self.config.request_runs_policy,
            "execution_mode": "persistent_pool" if self.config.worker_mode == "pool" else "legacy_jobs",
            **({"pool_url": self.config.pool_url} if self.config.worker_mode == "pool" else {}),
        }

    def _wait_and_fetch(self, handle: JobHandle) -> SimulationResult | list[SimulationResult]:
        return observation_wait.wait_and_fetch(self, handle)

    def _validate_request_runs(self, requests: list[SimulationRequest]) -> None:
        policy = self.config.request_runs_policy.strip().lower()
        if policy in {"allow", "single_job", "single-job"}:
            return
        if policy not in {"reject", "error", "fail"}:
            raise ValueError(
                "Unsupported MINI_BANG_K8S_REQUEST_RUNS_POLICY "
                f"'{self.config.request_runs_policy}'"
            )
        offenders = [request for request in requests if request.runs != 1]
        if offenders:
            raise ValueError(
                "K8sBackend scaling mode requires SimulationRequest.runs == 1. "
                "Use generate_samples_batch/run_batch with one run per request so "
                "the backend can distribute work across Kubernetes Jobs."
            )

    def _request_chunks(
        self,
        requests: list[SimulationRequest],
    ) -> Iterable[tuple[list[int], list[SimulationRequest]]]:
        chunk_size = max(1, self.config.batch_chunk_size)
        for start in range(0, len(requests), chunk_size):
            end = min(len(requests), start + chunk_size)
            yield list(range(start, end)), requests[start:end]

    def _max_parallel_jobs(self, requests: list[SimulationRequest]) -> int:
        configured = self.config.max_parallel_jobs
        resource_limits = [
            spec.max_concurrent_jobs
            for spec in (self._resource_spec([request]) for request in requests)
            if spec.max_concurrent_jobs is not None
        ]
        values = [value for value in [configured, *resource_limits] if value is not None]
        if not values:
            return 1
        return max(1, min(int(value) for value in values))

    def _resource_spec(self, requests: list[SimulationRequest]) -> ResourceSpec:
        spec = self.config.default_resources
        for request in requests:
            spec = spec.merge(get_simulator_entry(request.simulator_id).resource_spec)
            override = request.sample_params.get("resource_spec") or request.sample_params.get("resources")
            spec = spec.merge(override)
        return spec

    def _job_input_manifest(
        self,
        *,
        job_id: str,
        run_id: str,
        requests: list[SimulationRequest],
        context: RuntimeContext,
        result_uri: str,
        resource_spec: ResourceSpec,
    ) -> dict[str, Any]:
        return {
            "kind": "mini_bang.simulation_job",
            "job_id": job_id,
            "run_id": run_id,
            "task_id": context.task_id,
            "instance_id": context.instance_id,
            "requests": [request_to_transport(request) for request in requests],
            "runtime_context": context.to_dict(),
            "result_uri": result_uri,
            "resource_spec": resource_spec.to_dict(),
            "timeout_seconds": resource_spec.timeout_seconds or self.config.job_timeout_seconds,
            "metadata": {"backend": "k8s", "launch_tracking_version": LAUNCH_TRACKING_VERSION},
            "watchdog_policy": observation_recovery.policy(self.config),
            "runtime_capacity": self.config.runtime_capacity,
        }

    def _job_manifest(
        self,
        job_id: str,
        input_uri: str,
        resource_spec: ResourceSpec,
        context: RuntimeContext,
    ) -> dict[str, Any]:
        env = [
            {"name": "MINI_BANG_K8S_JOB_INPUT_URI", "value": input_uri},
            {
                "name": "MINI_BANG_K8S_RESULT_STORE_CONFIG",
                "value": json.dumps(self.config.worker_result_store_config or self.result_store.describe()),
            },
        ]
        if resource_spec.internal_parallelism is not None:
            env.append(
                {
                    "name": "MINI_BANG_INTERNAL_PARALLELISM",
                    "value": str(resource_spec.internal_parallelism),
                }
            )
        if resource_spec.pids_limit is not None:
            env.append(
                {
                    "name": "MINI_BANG_PIDS_LIMIT",
                    "value": str(resource_spec.pids_limit),
                }
            )
        env.extend(_storage_env(self.config.worker_result_store_config or {}))
        env.extend(_runtime_storage_env(context))
        env.extend({"name": key, "value": str(value)} for key, value in context.env.items())
        gpu = bool(resource_spec.gpu and float(resource_spec.gpu) > 0)
        pod_options = {}
        if gpu and self.config.gpu_runtime_class:
            pod_options["runtimeClassName"] = self.config.gpu_runtime_class
        if gpu and self.config.gpu_node_selector:
            pod_options["nodeSelector"] = self.config.gpu_node_selector
        body = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_id, "labels": {"app": "mini-bang-worker"}},
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": resource_spec.timeout_seconds or self.config.job_timeout_seconds,
                **_ttl_spec(self.config.ttl_seconds_after_finished),
                "template": {
                    "metadata": {"labels": {"app": "mini-bang-worker", "mini-bang-job-id": job_id}},
                    "spec": {
                        **pod_options,
                        "restartPolicy": "Never",
                        "terminationGracePeriodSeconds": OBSERVATION_TERMINATION_GRACE_SECONDS,
                        "automountServiceAccountToken": False,
                        "serviceAccountName": self.config.service_account,
                        "containers": [
                            {
                                "name": "worker",
                                "image": (self.config.gpu_worker_image or self.config.worker_image) if gpu else self.config.worker_image,
                                "imagePullPolicy": self.config.image_pull_policy,
                                "command": ["python", "-m", "mini_bang.worker.k8s_worker"],
                                "env": env,
                                "resources": _k8s_resources(resource_spec),
                            }
                        ],
                    },
                },
            },
        }
        return observation_recovery.configure_job(body, self.config)


def _k8s_resources(spec: ResourceSpec) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    if spec.cpu is not None:
        limits["cpu"] = str(spec.cpu)
    if spec.memory:
        limits["memory"] = spec.memory
    if spec.gpu is not None:
        limits["nvidia.com/gpu"] = str(spec.gpu)
    if spec.ephemeral_storage:
        limits["ephemeral-storage"] = spec.ephemeral_storage
    return {"limits": limits, "requests": {k: v for k, v in limits.items() if k != "nvidia.com/gpu"}}


def _env_resources(timeout: int) -> dict[str, Any]:
    return {
        "cpu": os.getenv("MINI_BANG_K8S_DEFAULT_CPU", "1"),
        "memory": os.getenv("MINI_BANG_K8S_DEFAULT_MEMORY", "1Gi"),
        "gpu": _optional_int(os.getenv("MINI_BANG_K8S_DEFAULT_GPU")),
        "timeout_seconds": timeout,
        "ephemeral_storage": os.getenv("MINI_BANG_K8S_DEFAULT_EPHEMERAL_STORAGE"),
    }


def _optional_int(value: Any) -> int | None:
    return int(value) if value not in {None, ""} else None


def _ttl_from_env() -> int | None:
    if os.getenv("MINI_BANG_KEEP_K8S_OBJECTS") == "1":
        return None
    return _optional_int(os.getenv("MINI_BANG_K8S_TTL_SECONDS_AFTER_FINISHED") or "300")


def _ttl_spec(value: int | None) -> dict[str, int]:
    return {"ttlSecondsAfterFinished": int(value)} if value is not None else {}


def _add_worker_s3_credentials(config: dict[str, Any]) -> None:
    mapping = {
        "access_key_id": ("MINI_BANG_RESULT_S3_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
        "secret_access_key": ("MINI_BANG_RESULT_S3_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
        "session_token": ("MINI_BANG_RESULT_S3_SESSION_TOKEN", "AWS_SESSION_TOKEN"),
    }
    for key, env_names in mapping.items():
        value = next((os.getenv(name) for name in env_names if os.getenv(name)), None)
        if value:
            config[key] = value


def _storage_env(config: dict[str, Any]) -> list[dict[str, str]]:
    env = []
    if config.get("access_key_id"):
        env.append({"name": "AWS_ACCESS_KEY_ID", "value": str(config["access_key_id"])})
    if config.get("secret_access_key"):
        env.append({"name": "AWS_SECRET_ACCESS_KEY", "value": str(config["secret_access_key"])})
    if config.get("session_token"):
        env.append({"name": "AWS_SESSION_TOKEN", "value": str(config["session_token"])})
    return env


def _runtime_storage_env(context: RuntimeContext) -> list[dict[str, str]]:
    payload = dict(context.payload or {})
    env: list[dict[str, str]] = []
    endpoint = payload.get("endpoint_url") or payload.get("s3_endpoint_url")
    if endpoint:
        env.append({"name": "MINI_BANG_PAYLOAD_S3_ENDPOINT_URL", "value": str(endpoint)})
    region = payload.get("region_name") or payload.get("region")
    if region:
        env.append({"name": "MINI_BANG_PAYLOAD_S3_REGION", "value": str(region)})
    return env
