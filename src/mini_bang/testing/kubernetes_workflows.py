from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mini_bang.artifacts import ArtifactRef
from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.client.deployment.k8s import render_k8s_manifests
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.result_store import S3ResultStore
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.simulators import registry as simulator_registry
from mini_bang.testing.acceptance_workload import acceptance_workload

ACCESS_KEY = "mini-bang"
SECRET_KEY = "mini-bang-secret"
TEST_PACKAGES = ",".join(
    [
        "mini_bang.testing.context_marker_simulator",
        "mini_bang.testing.field_payload_simulator",
        "mini_bang.testing.artifact_echo_simulator",
    ]
)


def batch_parallelism_e2e(
    *,
    namespace: str,
    image: str,
    host_endpoint: str,
    worker_endpoint: str,
    bucket: str,
) -> dict[str, Any]:
    _register_testing_simulators()
    store = _host_store(host_endpoint, bucket)
    before = _job_names(namespace, "app=mini-bang-worker")
    base = _backend_config(namespace, image, store, worker_endpoint)
    backend = K8sBackend(
        config=K8sBackendConfig(
            worker_mode="legacy_jobs",
            namespace=base.namespace,
            worker_image=base.worker_image,
            service_account=base.service_account,
            image_pull_policy=base.image_pull_policy,
            polling_interval_seconds=base.polling_interval_seconds,
            job_timeout_seconds=base.job_timeout_seconds,
            result_store_config=base.result_store_config,
            worker_result_store_config=base.worker_result_store_config,
            batch_chunk_size=2,
            max_parallel_jobs=2,
        ),
        result_store=store,
    )
    context = _runtime_context(
        "run-batch",
        "toy/batch",
        "case",
        worker_endpoint,
        bucket=bucket,
        artifact_prefix=f"s3://{bucket}/artifacts/",
        payload_prefix="batch-payloads",
    )
    results = backend.run_batch(
        [SimulationRequest("field_payload_test", index) for index in range(10)],
        context=context,
    )
    created = sorted(_job_names(namespace, "app=mini-bang-worker") - before)
    assert len(created) >= 5
    assert [result.request.saturation for result in results] == list(range(10))
    before_submit = _job_names(namespace, "app=mini-bang-worker")
    submit_handle = backend.submit_batch(
        [SimulationRequest("field_payload_test", 100 + index) for index in range(10)],
        context=context,
    )
    immediate_submit = _job_names(namespace, "app=mini-bang-worker") - before_submit
    assert len(immediate_submit) <= 2
    _wait_submit_batch(backend, submit_handle)
    submit_results = backend.fetch(submit_handle)
    submit_created = sorted(_job_names(namespace, "app=mini-bang-worker") - before_submit)
    assert len(submit_created) >= 5
    assert [result.request.saturation for result in submit_results] == list(range(100, 110))
    return {
        "status": "CORRECT",
        "job_count": len(created),
        "max_parallel_jobs": 2,
        "ordered_results": True,
        "submit_batch_created_multiple_jobs": True,
        "submit_batch_job_count": len(submit_created),
        "submit_batch_max_parallel_jobs_enforced": True,
        "submit_batch_ordered_fetch": True,
    }


def k8s_submit_e2e(
    *,
    namespace: str,
    image: str,
    worker_endpoint: str,
    bucket: str,
    root: Path,
) -> dict[str, Any]:
    manifests = render_k8s_manifests(
        _Namespace(
            namespace=namespace,
            executor_replicas=2,
            worker_image=image,
            payload_store="minio",
            s3_bucket=bucket,
            s3_prefix="workflow",
            s3_endpoint_url=worker_endpoint,
            agent_image=None,
            with_minio=False,
        )
    )
    for name in (
        "serviceaccounts.yaml",
        "configmap.yaml",
        "secret-runner.yaml",
        "secret-storage.yaml",
        "runner.yaml",
        "executor.yaml",
        "networkpolicy.yaml",
    ):
        _kubectl_apply_yaml(namespace, manifests[name])
    _run([
        "kubectl",
        "-n",
        namespace,
        "wait",
        "--for=condition=available",
        "deployment/mini-bang-runner",
        "--timeout=180s",
    ], root=root)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    workload = acceptance_workload()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mini_bang.cli",
            "k8s",
            "submit",
            "--task",
            workload.task_id,
            # The default instance; the smoke agent submits a valid constant answer.
            "--agent-image",
            image,
            "--agent",
            workload.smoke_agent,
            "--namespace",
            namespace,
            "--admin-token",
            "mini-bang-admin",
            "--timeout",
            "180",
            "--wait",
            "--format",
            "json",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    # Exit code 3 reports a scored run below the task's success threshold.
    if proc.returncode not in (0, 3):
        raise RuntimeError(proc.stderr or proc.stdout)
    payload = json.loads(proc.stdout)
    # The constant smoke answer is scored as zero skill: a scored run, not a successful one.
    assert payload["success"] is False and payload["score"] is not None
    run_id = payload["run_id"]
    secret_name = f"mini-bang-runner-session-{_safe_name(run_id)}"
    job_name = f"mini-bang-agent-{_safe_name(run_id)}"
    _wait_absent(namespace, "secret", secret_name, root=root)
    _wait_absent(namespace, "job", job_name, root=root)
    assert Path(payload["run_record_path"]).is_file()
    return {
        "status": "CORRECT",
        "run_id": run_id,
        "scored": True,
        "session_secret_cleaned": True,
        "agent_job_cleaned": True,
        "result_fetchable_after_cleanup": True,
    }


def _backend_config(namespace: str, image: str, store: S3ResultStore, worker_endpoint: str) -> K8sBackendConfig:
    return K8sBackendConfig(
        worker_mode="legacy_jobs",
        namespace=namespace,
        worker_image=image,
        service_account="mini-bang-executor",
        image_pull_policy="Never",
        polling_interval_seconds=1,
        job_timeout_seconds=180,
        result_store_config=store.describe(),
        worker_result_store_config=_worker_store_config(worker_endpoint, store.bucket),
    )


def _runtime_context(
    run_id: str,
    task_id: str,
    instance_id: str,
    endpoint: str,
    *,
    bucket: str,
    artifact_prefix: str,
    payload_prefix: str,
) -> RuntimeContext:
    return RuntimeContext(
        run_id=run_id,
        task_id=task_id,
        instance_id=instance_id,
        artifact={
            "cache_dir": "/tmp/mini-bang-artifact-cache",
            "allowed_schemes": ["s3"],
            "allowed_s3_prefixes": [artifact_prefix],
            "s3_endpoint_url": endpoint,
        },
        payload={"kind": "s3", "bucket": bucket, "prefix": payload_prefix, "endpoint_url": endpoint},
        security={"network_enabled": False, "gdrive_enabled": False},
        env={"MINI_BANG_SIMULATOR_PACKAGES": TEST_PACKAGES},
    )


def _host_store(endpoint: str, bucket: str) -> S3ResultStore:
    return S3ResultStore(bucket=bucket, prefix="results", endpoint_url=endpoint, access_key_id=ACCESS_KEY, secret_access_key=SECRET_KEY)


def _worker_store_config(endpoint: str, bucket: str) -> dict[str, Any]:
    return {"kind": "s3", "bucket": bucket, "prefix": "results", "endpoint_url": endpoint, "access_key_id": ACCESS_KEY, "secret_access_key": SECRET_KEY}


def _register_testing_simulators() -> None:
    specs = [
        ("context_marker_test", "mini_bang.testing.context_marker_simulator.factory:create_macro"),
        ("field_payload_test", "mini_bang.testing.field_payload_simulator.factory:create_macro"),
        ("artifact_echo_test", "mini_bang.testing.artifact_echo_simulator.factory:create_macro"),
    ]
    for simulator_id, factory in specs:
        simulator_registry._REGISTRY._specs.pop(simulator_id, None)
        simulator_registry.register_simulator_entry(
            simulator_registry.SimulatorSpec(simulator_id, "e2e", factory, grouped_sampling_safe=True)
        )


def _kubectl_apply_yaml(namespace: str, manifest: str) -> None:
    proc = subprocess.run(["kubectl", "-n", namespace, "apply", "-f", "-"], input=manifest, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)


def _job_names(namespace: str, selector: str) -> set[str]:
    raw = _run(["kubectl", "-n", namespace, "get", "jobs", "-l", selector, "-o", "json"], capture=True)
    data = json.loads(raw)
    return {item["metadata"]["name"] for item in data.get("items", [])}


def _wait_absent(namespace: str, kind: str, name: str, *, root: Path) -> None:
    for _ in range(30):
        proc = subprocess.run(
            ["kubectl", "-n", namespace, "get", kind, name],
            text=True,
            capture_output=True,
            cwd=root,
        )
        if proc.returncode != 0:
            return
        import time

        time.sleep(1)
    raise RuntimeError(f"stale Kubernetes object after submit: {kind}/{name}")


def _wait_submit_batch(backend: K8sBackend, handle) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        status = backend.poll(handle)
        if status.state == "succeeded":
            return
        if status.state == "failed":
            raise RuntimeError(status.detail or "submit_batch failed")
        time.sleep(1)
    raise TimeoutError("submit_batch did not finish")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value).strip("-._") or "run"


def _run(cmd: list[str], *, root: Path | None = None, capture: bool = False) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"command failed: {cmd}")
    return (proc.stdout or "").strip() if capture else ""


class _Namespace:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
