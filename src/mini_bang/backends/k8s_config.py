"""Kubernetes transport and startup-only persistent-pool configuration."""
from dataclasses import dataclass
from typing import Any

from mini_bang.execution.resources import ResourceSpec


@dataclass(frozen=True)
class K8sBackendConfig:
    namespace: str
    worker_image: str
    service_account: str = "mini-bang-executor"
    executable_image: str | None = None
    executable_service_account: str = "mini-bang-executable"
    job_timeout_seconds: int = 600
    polling_interval_seconds: float = 1.0
    image_pull_policy: str = "IfNotPresent"
    ttl_seconds_after_finished: int | None = 300
    result_store_config: dict[str, Any] | None = None
    worker_result_store_config: dict[str, Any] | None = None
    default_resources: ResourceSpec = ResourceSpec(cpu="1", memory="1Gi")
    max_parallel_jobs: int | None = None
    batch_chunk_size: int = 1
    request_runs_policy: str = "reject"
    gpu_worker_image: str | None = None
    gpu_runtime_class: str | None = None
    gpu_node_selector: dict[str, str] | None = None
    observation_idle_seconds: float = 600
    observation_max_retries: int = 2
    observation_cpu_limit_seconds: float | None = None
    emergency_job_timeout_seconds: int | None = None
    observation_scheduling_idle_seconds: float = 600
    runtime_capacity: dict[str, Any] | None = None
    worker_mode: str = "pool"
    pool_url: str = "http://mini-bang-pool:8770"
    worker_pool_url: str | None = None
