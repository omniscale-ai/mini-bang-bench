from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.client.paths import INFRASTRUCTURE_ROOT, KUBERNETES_VALIDATION_STATUS, new_run_directory
from mini_bang.client.cli.io import read_json
from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.k8s_executable import K8sExecutableConfig, K8sExecutableRunner
from mini_bang.testing.acceptance_workload import acceptance_workload
from mini_bang.testing.kubernetes_benchmark import ROOT, _current_context, run_stamp, _record_environment, _run, _write_status


def main() -> int:
    suffix = run_stamp()
    namespace = f"mini-bang-exec-sec-{suffix}"
    image = f"mini-bang-executable:e2e-{suffix}"
    keep = os.getenv("MINI_BANG_KEEP_E2E_NAMESPACE") == "1"
    status: dict[str, Any] = {
        "namespace": namespace,
        "worker_image": "NOT SUPPORTED",
        "executable_image": image,
        "result_storage_mode": "filesystem",
        "payload_storage_mode": "NOT SUPPORTED",
        "adversarial_executable_security_tests": "NOT VERIFIED",
        "cleanup_tests": "NOT VERIFIED",
        "steps": [],
    }
    try:
        _record_environment(status)
        docker_config = ROOT / INFRASTRUCTURE_ROOT / f"docker-config-exec-sec-{suffix}"
        docker_config.mkdir(parents=True, exist_ok=True)
        os.environ["DOCKER_CONFIG"] = str(docker_config)
        _run(["docker", "build", "-f", "deploy/shared/Dockerfile.executable", "-t", image, "."])
        if _current_context().startswith("kind-"):
            _run(["kind", "load", "docker-image", image, "--name", _current_context()[len("kind-") :]])
        _run(["kubectl", "create", "namespace", namespace])
        _run(["kubectl", "-n", namespace, "create", "serviceaccount", "mini-bang-executable"])
        _apply(namespace, _network_policy(namespace))
        _deploy_forbidden_http(namespace, image)
        result_root = ROOT / new_run_directory("kubernetes-executable-security")
        runner = _runner(namespace, image, result_root)
        status["adversarial"] = _adversarial_checks(runner)
        status["schema_validation"] = _schema_check(runner)
        status["timeout"] = _timeout_check(runner)
        status["resource_limit"] = _resource_check(runner)
        _wait_no_resources(namespace)
        status["adversarial_executable_security_tests"] = "CORRECT"
        status["cleanup_tests"] = "CORRECT"
        status["real_k8s_e2e_result"] = "CORRECT"
        if not keep:
            _run(["kubectl", "delete", "namespace", namespace, "--wait=true", "--timeout=120s"])
            status["namespace_cleanup"] = "CORRECT"
        status = _merge_previous_status(status)
        _write_status(status)
        print(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        status["real_k8s_e2e_result"] = "INCORRECT"
        status["error"] = repr(exc)
        status = _merge_previous_status(status)
        _write_status(status)
        print(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True))
        return 1
    finally:
        if not keep:
            subprocess.run(
                ["kubectl", "delete", "namespace", namespace, "--ignore-not-found"],
                text=True,
                capture_output=True,
                check=False,
            )


def _runner(namespace: str, image: str, result_root: Path) -> K8sExecutableRunner:
    store = FileSystemResultStore(result_root)
    return K8sExecutableRunner(
        config=K8sExecutableConfig(
            execution_mode="legacy_jobs",
            namespace=namespace,
            image=image,
            service_account="mini-bang-executable",
            result_store_config=store.describe(),
            image_pull_policy="Never",
            polling_interval_seconds=1,
        ),
        result_store=store,
    )


def _adversarial_checks(runner: K8sExecutableRunner) -> dict[str, Any]:
    result = runner.run({"code": _adversarial_code()}, {"run_id": "exec-security"}, ExecutionLimits(timeout_seconds=30))
    assert result.exit_code == 0, result.stderr
    output = result.output or {}
    expected_false = [
        "runner_token_visible",
        "session_token_visible",
        "storage_credentials_visible",
        "serviceaccount_token_readable",
        "proc_environ_has_secret",
        "gold_readable",
        "network_reachable",
    ]
    for key in expected_false:
        assert output.get(key) is False, output
    return {"status": "CORRECT", **{key: False for key in expected_false}}


def _schema_check(runner: K8sExecutableRunner) -> dict[str, Any]:
    result = runner.run(
        {"code": "import sys\nopen(sys.argv[2], 'w').write('not-json')\n"},
        {"run_id": "exec-schema"},
        ExecutionLimits(timeout_seconds=10),
    )
    assert result.exit_code != 0
    return {"status": "CORRECT", "failed_submission": True}


def _timeout_check(runner: K8sExecutableRunner) -> dict[str, Any]:
    result = runner.run({"code": "import time\ntime.sleep(20)\n"}, {"run_id": "exec-timeout"}, ExecutionLimits(timeout_seconds=2))
    assert result.exit_code != 0
    assert result.timed_out is True
    return {"status": "CORRECT", "failed_submission": True}


def _resource_check(runner: K8sExecutableRunner) -> dict[str, Any]:
    code = "buf = bytearray(512 * 1024 * 1024)\nprint(len(buf))\n"
    result = runner.run({"code": code}, {"run_id": "exec-resource"}, ExecutionLimits(timeout_seconds=20, memory_mb=96))
    assert result.exit_code != 0
    return {"status": "CORRECT", "failed_submission": True}


def _adversarial_code() -> str:
    gold = acceptance_workload().gold_path
    gold_paths = ["/workspace/src/mini_bang/" + gold, "/usr/local/lib/python3.13/site-packages/mini_bang/" + gold]
    return ADVERSARIAL_CODE.replace("GOLD_PATHS", json.dumps(gold_paths))


ADVERSARIAL_CODE = r"""
import json
import os
import pathlib
import sys
import urllib.request

env = os.environ
storage = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GDRIVE_CREDENTIALS"]
checks = {
    "runner_token_visible": bool(env.get("MINI_BANG_RUNNER_TOKEN")),
    "session_token_visible": bool(env.get("MINI_BANG_SESSION_TOKEN")),
    "storage_credentials_visible": any(env.get(name) for name in storage),
}
try:
    pathlib.Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text()
    checks["serviceaccount_token_readable"] = True
except Exception:
    checks["serviceaccount_token_readable"] = False
needles = [b"MINI_BANG_RUNNER_TOKEN", b"AWS_SECRET_ACCESS_KEY", b"SESSION_TOKEN"]
body = b""
for name in ["/proc/self/environ", "/proc/1/environ"]:
    try:
        body += pathlib.Path(name).read_bytes()
    except Exception:
        pass
checks["proc_environ_has_secret"] = any(item in body for item in needles)
gold_paths = GOLD_PATHS
checks["gold_readable"] = any(pathlib.Path(path).is_file() for path in gold_paths)
try:
    urllib.request.urlopen("http://forbidden-http:8080", timeout=3).read(1)
    checks["network_reachable"] = True
except Exception:
    checks["network_reachable"] = False
json.dump(checks, open(sys.argv[2], "w"))
"""


def _network_policy(namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": "mini-bang-executable-default-deny-egress", "namespace": namespace},
        "spec": {
            "podSelector": {"matchLabels": {"mini-bang-role": "executable", "mini-bang-network": "disabled"}},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [],
            "egress": [],
        },
    }


def _deploy_forbidden_http(namespace: str, image: str) -> None:
    _apply(
        namespace,
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "forbidden-http", "namespace": namespace},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "forbidden-http"}},
                "template": {
                    "metadata": {"labels": {"app": "forbidden-http"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "http",
                                "image": image,
                                "imagePullPolicy": "Never",
                                "command": ["python", "-m", "http.server", "8080"],
                                "ports": [{"containerPort": 8080}],
                            }
                        ]
                    },
                },
            },
        },
    )
    _apply(
        namespace,
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "forbidden-http", "namespace": namespace},
            "spec": {"selector": {"app": "forbidden-http"}, "ports": [{"port": 8080, "targetPort": 8080}]},
        },
    )
    _run(["kubectl", "-n", namespace, "wait", "--for=condition=available", "deployment/forbidden-http", "--timeout=120s"])


def _wait_no_resources(namespace: str) -> None:
    deadline = time.time() + 30
    last: dict[str, list[str]] = {}
    while time.time() < deadline:
        last = {}
        for kind in ["jobs", "configmaps"]:
            raw = _run(["kubectl", "-n", namespace, "get", kind, "-l", "app=mini-bang-executable", "-o", "json"], capture=True)
            names = [item["metadata"]["name"] for item in json.loads(raw).get("items", [])]
            if names:
                last[kind] = names
        if not last:
            return
        time.sleep(1)
    raise AssertionError(last)


def _apply(namespace: str, manifest: dict[str, Any]) -> None:
    proc = subprocess.run(
        ["kubectl", "-n", namespace, "apply", "-f", "-"],
        input=json.dumps(manifest),
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)


def _merge_previous_status(status: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / KUBERNETES_VALIDATION_STATUS
    try:
        previous = read_json(path)
    except Exception:
        return status
    merged = dict(previous)
    merged.update(status)
    if previous.get("simulation") or previous.get("batch_parallelism"):
        merged["main_k8s_e2e"] = {
            "namespace": previous.get("namespace"),
            "worker_image": previous.get("worker_image"),
            "simulation": previous.get("simulation", {}).get("status", "NOT VERIFIED"),
            "batch_parallelism": previous.get("batch_parallelism", {}).get("status", "NOT VERIFIED"),
            "submit_batch_created_multiple_jobs": previous.get("batch_parallelism", {}).get(
                "submit_batch_created_multiple_jobs",
                "NOT VERIFIED",
            ),
            "submit_batch_max_parallel_jobs_enforced": previous.get("batch_parallelism", {}).get(
                "submit_batch_max_parallel_jobs_enforced",
                "NOT VERIFIED",
            ),
            "submit_batch_ordered_fetch": previous.get("batch_parallelism", {}).get(
                "submit_batch_ordered_fetch",
                "NOT VERIFIED",
            ),
            "scoring": previous.get("scoring", {}).get("status", "NOT VERIFIED"),
            "executable_runner": previous.get("executable", {}).get("status", "NOT VERIFIED"),
            "k8s_submit": previous.get("k8s_submit", {}).get("status", "NOT VERIFIED"),
        }
        if previous.get("namespace") and status.get("namespace") != previous.get("namespace"):
            merged["namespace"] = (
                f"main={previous['namespace']}; executable-security={status.get('namespace')}"
            )
    for key in ("worker_image", "payload_storage_mode"):
        if status.get(key) in {"NOT SUPPORTED", "NOT VERIFIED"} and previous.get(key):
            merged[key] = previous[key]
    if previous.get("result_storage_mode") and status.get("result_storage_mode") == "filesystem":
        merged["result_storage_mode"] = f"{previous['result_storage_mode']}; executable-security filesystem"
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
