"""Historical per-Job diagnostic, not an installer for the persistent pool.

Current installed-pool acceptance: python -m mini_bang.testing.persistent_kubernetes.
The retained historical CLI workflow requires its original deployment routing.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import boto3

from mini_bang.artifacts import ArtifactRef
from mini_bang.client.paths import INFRASTRUCTURE_ROOT, KUBERNETES_VALIDATION_STATUS
from mini_bang.client.cli.io import write_json
from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.result_store import S3ResultStore
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.services import ExecutionServices
from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.k8s_executable import K8sExecutableConfig, K8sExecutableRunner
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring_jobs import (
    K8sScoringJobBackend,
    ScoringJob,
    instance_spec_to_manifest,
    task_spec_to_manifest,
)
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission
from mini_bang.simulators import registry as simulator_registry
from mini_bang.testing.kubernetes_workflows import batch_parallelism_e2e, k8s_submit_e2e


ROOT = Path(__file__).resolve().parents[3]
ACCESS_KEY = "mini-bang"
SECRET_KEY = "mini-bang-secret"
BUCKET = "mini-bang-e2e"
TEST_PACKAGES = ",".join(
    [
        "mini_bang.testing.context_marker_simulator",
        "mini_bang.testing.field_payload_simulator",
        "mini_bang.testing.artifact_echo_simulator",
    ]
)


def main() -> int:
    suffix = run_stamp()
    namespace = f"mini-bang-e2e-{suffix}"
    image = f"mini-bang-worker:e2e-{suffix}"
    executable_image = f"mini-bang-executable:e2e-{suffix}"
    host_port = random.randint(19000, 19999)
    keep = os.getenv("MINI_BANG_KEEP_E2E_NAMESPACE") == "1"
    status: dict[str, Any] = {
        "namespace": namespace,
        "worker_image": image,
        "executable_image": executable_image,
        "result_storage_mode": "s3/minio",
        "payload_storage_mode": "s3/minio",
        "adversarial_executable_security_tests": "NOT VERIFIED",
        "cleanup_tests": "NOT VERIFIED",
        "steps": [],
    }
    port_forward: subprocess.Popen[str] | None = None
    try:
        _record_environment(status)
        docker_config = ROOT / INFRASTRUCTURE_ROOT / f"docker-config-{suffix}"
        docker_config.mkdir(parents=True, exist_ok=True)
        os.environ["DOCKER_CONFIG"] = str(docker_config)
        _run(["docker", "build", "-f", "deploy/shared/Dockerfile.worker", "-t", image, "."])
        _run(["docker", "build", "-f", "deploy/shared/Dockerfile.executable", "-t", executable_image, "."])
        status["steps"].append("docker image built")
        if _current_context().startswith("kind-"):
            cluster = _current_context()[len("kind-") :]
            _run(["kind", "load", "docker-image", image, "--name", cluster])
            _run(["kind", "load", "docker-image", executable_image, "--name", cluster])
            status["steps"].append("image loaded into kind")
        _run(["kubectl", "create", "namespace", namespace])
        _run(["kubectl", "-n", namespace, "create", "serviceaccount", "mini-bang-executor"])
        _run(["kubectl", "-n", namespace, "create", "serviceaccount", "mini-bang-executable"])
        _deploy_minio(namespace)
        _run([
            "kubectl",
            "-n",
            namespace,
            "wait",
            "--for=condition=available",
            "deployment/minio",
            "--timeout=180s",
        ])
        port_forward = subprocess.Popen(
            ["kubectl", "-n", namespace, "port-forward", "svc/minio", f"{host_port}:9000"],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        host_endpoint = f"http://127.0.0.1:{host_port}"
        worker_endpoint = f"http://minio.{namespace}.svc.cluster.local:9000"
        s3 = _wait_for_minio(host_endpoint)
        _ensure_bucket(s3)
        status["steps"].append("minio ready")
        status["simulation"] = _simulation_e2e(namespace, image, host_endpoint, worker_endpoint, s3)
        status["batch_parallelism"] = batch_parallelism_e2e(
            namespace=namespace,
            image=image,
            host_endpoint=host_endpoint,
            worker_endpoint=worker_endpoint,
            bucket=BUCKET,
        )
        status["scoring"] = _scoring_e2e(namespace, image, host_endpoint, worker_endpoint, s3)
        _apply_executable_network_policy(namespace)
        status["executable"] = _executable_e2e(namespace, executable_image, host_endpoint, worker_endpoint)
        status["k8s_submit"] = k8s_submit_e2e(
            namespace=namespace,
            image=image,
            worker_endpoint=worker_endpoint,
            bucket=BUCKET,
            root=ROOT,
        )
        status["real_k8s_e2e_result"] = "CORRECT"
        status["cleanup_tests"] = "CORRECT"
        _write_status(status)
        print(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        status["real_k8s_e2e_result"] = "INCORRECT"
        status["error"] = repr(exc)
        _write_status(status)
        print(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True))
        return 1
    finally:
        if port_forward is not None:
            port_forward.terminate()
            try:
                port_forward.wait(timeout=5)
            except subprocess.TimeoutExpired:
                port_forward.kill()
        if not keep:
            subprocess.run(
                ["kubectl", "delete", "namespace", namespace, "--ignore-not-found"],
                text=True,
                capture_output=True,
                check=False,
            )


def _simulation_e2e(namespace: str, image: str, host_endpoint: str, worker_endpoint: str, s3) -> dict[str, Any]:
    _register_testing_simulators()
    s3.put_object(Bucket=BUCKET, Key="artifacts/marker.txt", Body=b"agent-k8s")
    store = _host_store(host_endpoint)
    backend = K8sBackend(
        config=_backend_config(namespace, image, store, worker_endpoint),
        result_store=store,
    )
    context = _runtime_context(
        "run-sim",
        "toy/simulation",
        "case",
        worker_endpoint,
        artifact_prefix=f"s3://{BUCKET}/artifacts/",
        payload_prefix="payloads",
    )
    requests = [
        SimulationRequest(
            "context_marker_test",
            1,
            macro_params={
                "marker_ref": ArtifactRef(uri=f"s3://{BUCKET}/artifacts/marker.txt").to_dict()
            },
        ),
        SimulationRequest("field_payload_test", 1),
    ]
    results = backend.run_batch(requests, context=context)
    _host_payload_env(host_endpoint)
    marker_payload = results[0].materialize_payload()
    field_payload = results[1].materialize_payload()
    assert marker_payload["trajectories"][0]["marker"] == "agent-k8s"
    assert results[0].payload_ref is not None
    assert results[1].payload_ref is not None
    assert field_payload["trajectories"][0]["fields"]["field"]["__mini_bang_payload_ref__"] is True
    return {"status": "CORRECT", "job_count": len(results), "payload_ref": True}


def _scoring_e2e(namespace: str, image: str, host_endpoint: str, worker_endpoint: str, s3) -> dict[str, Any]:
    _register_testing_simulators()
    s3.put_object(Bucket=BUCKET, Key="hidden/secret.bin", Body=b"hidden-body")
    store = _host_store(host_endpoint)
    backend = K8sScoringJobBackend(
        config=_backend_config(namespace, image, store, worker_endpoint),
        result_store=store,
    )
    task_spec = TaskSpec(
        task_id="toy/scoring",
        package="mini_bang.testing",
        api_class_path="mini_bang.testing.scoring_support:ToyAPI",
        scorer_class_path="mini_bang.testing.scoring_support:RuntimeSimulationScorer",
        policy_class_path="mini_bang.testing.scoring_support:ToyPolicy",
        config={"scoring": {"requires_services": True, "distributed": True}},
    )
    instance = InstanceSpec(task_id=task_spec.task_id, instance_id="case", package=task_spec.package)
    context = _runtime_context(
        "run-score",
        task_spec.task_id,
        "case",
        worker_endpoint,
        artifact_prefix=f"s3://{BUCKET}/hidden/",
        payload_prefix="scoring-payloads",
    )
    record = RunRecord.from_parts(
        run_id="run-score",
        task_id=task_spec.task_id,
        instance_id="case",
        submission=TaskSubmission(answer={}),
    )
    job = ScoringJob(
        run_id="run-score",
        task_id=task_spec.task_id,
        instance_id="case",
        provisional_record=record,
        submission=TaskSubmission(answer={}),
        gold={
            "data": {
                "artifact_ref": ArtifactRef(uri=f"s3://{BUCKET}/hidden/secret.bin").to_dict(),
                "expected_size": 11,
            }
        },
        scorer_class_path=task_spec.scorer_class_path or "",
        task_spec=task_spec_to_manifest(task_spec),
        instance_spec=instance_spec_to_manifest(instance),
        scoring_context=context,
    )
    envelope = backend.run(job)
    agent_context = _runtime_context("run-score", task_spec.task_id, "case", worker_endpoint, artifact_prefix=f"s3://{BUCKET}/public/", payload_prefix="x")
    try:
        ExecutionServices().materialize_artifact(
            ArtifactRef(uri=f"s3://{BUCKET}/hidden/secret.bin"),
            context=agent_context,
        )
        raise AssertionError("agent context unexpectedly read hidden artifact")
    except PermissionError:
        pass
    assert envelope["validation"]["success"] is True
    assert envelope["run_record"]["trace"] == []
    assert envelope["scoring_trace"][0]["event"] == "scoring_simulation_started"
    return {"status": "CORRECT", "trace_separate": True}


def _executable_e2e(namespace: str, image: str, host_endpoint: str, worker_endpoint: str) -> dict[str, Any]:
    store = _host_store(host_endpoint)
    runner = K8sExecutableRunner(
        config=K8sExecutableConfig(
            execution_mode="legacy_jobs",
            namespace=namespace,
            image=image,
            result_store_config=store.describe(),
            worker_result_store_config=_worker_store_config(worker_endpoint),
            image_pull_policy="Never",
        ),
        result_store=store,
    )
    code = (
        "import json, os, sys\n"
        "inp, out = sys.argv[1], sys.argv[2]\n"
        "json.dump({'ok': True, 'aws_visible': any(k.startswith('AWS_') for k in os.environ)}, open(out, 'w'))\n"
    )
    result = runner.run({"code": code}, {"run_id": "run-exec"}, ExecutionLimits(timeout_seconds=30))
    assert result.exit_code == 0
    assert result.output == {"ok": True, "aws_visible": False}
    return {"status": "CORRECT", "credentials_hidden_from_code": True}


def _apply_executable_network_policy(namespace: str) -> None:
    policy = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": "mini-bang-executable-default-deny-egress", "namespace": namespace},
        "spec": {
            "podSelector": {
                "matchLabels": {
                    "mini-bang-role": "executable",
                    "mini-bang-network": "disabled",
                }
            },
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [],
            "egress": [],
        },
    }
    _kubectl_apply(namespace, policy)


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
        worker_result_store_config=_worker_store_config(worker_endpoint),
    )


def _runtime_context(run_id: str, task_id: str, instance_id: str, endpoint: str, *, artifact_prefix: str, payload_prefix: str) -> RuntimeContext:
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
        payload={
            "kind": "s3",
            "bucket": BUCKET,
            "prefix": payload_prefix,
            "endpoint_url": endpoint,
        },
        security={"network_enabled": False, "gdrive_enabled": False},
        env={"MINI_BANG_SIMULATOR_PACKAGES": TEST_PACKAGES},
    )


def _host_store(endpoint: str) -> S3ResultStore:
    return S3ResultStore(
        bucket=BUCKET,
        prefix="results",
        endpoint_url=endpoint,
        access_key_id=ACCESS_KEY,
        secret_access_key=SECRET_KEY,
    )


def _worker_store_config(endpoint: str) -> dict[str, Any]:
    return {
        "kind": "s3",
        "bucket": BUCKET,
        "prefix": "results",
        "endpoint_url": endpoint,
        "access_key_id": ACCESS_KEY,
        "secret_access_key": SECRET_KEY,
    }


def _host_payload_env(endpoint: str) -> None:
    os.environ["MINI_BANG_PAYLOAD_S3_ENDPOINT_URL"] = endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY


def _deploy_minio(namespace: str) -> None:
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "minio", "namespace": namespace},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "minio"}},
            "template": {
                "metadata": {"labels": {"app": "minio"}},
                "spec": {
                    "containers": [
                        {
                            "name": "minio",
                            "image": "quay.io/minio/minio:latest",
                            "args": ["server", "/data"],
                            "env": [
                                {"name": "MINIO_ROOT_USER", "value": ACCESS_KEY},
                                {"name": "MINIO_ROOT_PASSWORD", "value": SECRET_KEY},
                            ],
                            "ports": [{"containerPort": 9000}],
                        }
                    ]
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "minio", "namespace": namespace},
        "spec": {"selector": {"app": "minio"}, "ports": [{"port": 9000, "targetPort": 9000}]},
    }
    _kubectl_apply(namespace, manifest)
    _kubectl_apply(namespace, service)


def _kubectl_apply(namespace: str, manifest: dict[str, Any]) -> None:
    proc = subprocess.run(
        ["kubectl", "-n", namespace, "apply", "-f", "-"],
        input=json.dumps(manifest),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)


def _wait_for_minio(endpoint: str):
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            client.list_buckets()
            return client
        except Exception:
            time.sleep(1)
    raise TimeoutError("MinIO port-forward did not become ready")


def _ensure_bucket(client) -> None:
    try:
        client.create_bucket(Bucket=BUCKET)
    except Exception:
        pass


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


def _record_environment(status: dict[str, Any]) -> None:
    for key, cmd in {
        "kubectl_context": ["kubectl", "config", "current-context"],
        "kubectl_nodes": ["kubectl", "get", "nodes", "-o", "wide"],
        "docker_version": ["docker", "version"],
        "python_version": [sys.executable, "--version"],
        "which_kubectl": ["which", "kubectl"],
        "which_docker": ["which", "docker"],
    }.items():
        status[key] = _run(cmd, capture=True)
    _write_status(status)


def _write_status(status: dict[str, Any]) -> None:
    write_json(ROOT / KUBERNETES_VALIDATION_STATUS, status)


def _run(cmd: list[str], *, capture: bool = False) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"command failed: {cmd}")
    return (proc.stdout or "").strip() if capture else ""


def _current_context() -> str:
    return _run(["kubectl", "config", "current-context"], capture=True)


def run_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')


if __name__ == "__main__":
    raise SystemExit(main())
