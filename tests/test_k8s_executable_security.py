from __future__ import annotations

import argparse
import json

from mini_bang.client.deployment.k8s import render_k8s_manifests
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.k8s_executable import K8sExecutableConfig, K8sExecutableRunner

from k8s_executable_fakes import LocalExecutableK8sClient


def test_k8s_executable_cannot_read_runner_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MINI_BANG_RUNNER_TOKEN", "runner-secret")
    monkeypatch.setenv("MINI_BANG_SESSION_TOKEN", "session-secret")
    result, client = _run(
        tmp_path,
        _probe_env_code(["MINI_BANG_RUNNER_TOKEN", "MINI_BANG_SESSION_TOKEN", "AUTHORIZATION"]),
    )

    assert result.exit_code == 0
    assert result.output == {"found": {}, "proc_self_has_secret": False}
    assert _env_names(client).isdisjoint({"MINI_BANG_RUNNER_TOKEN", "MINI_BANG_SESSION_TOKEN"})


def test_k8s_executable_cannot_read_storage_credentials(monkeypatch, tmp_path) -> None:
    secret_names = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GDRIVE_CREDENTIALS",
        "MINI_BANG_RESULT_S3_ENDPOINT_URL",
        "MINI_BANG_PAYLOAD_S3_ENDPOINT_URL",
    ]
    for name in secret_names:
        monkeypatch.setenv(name, f"secret-{name}")

    result, client = _run(tmp_path, _probe_env_code(secret_names))

    assert result.exit_code == 0
    assert result.output == {"found": {}, "proc_self_has_secret": False}
    assert _env_names(client).isdisjoint(set(secret_names))


def test_k8s_executable_manifest_disables_serviceaccount_token_mount(tmp_path) -> None:
    result, client = _run(
        tmp_path,
        "import json, sys\njson.dump({'ok': True}, open(sys.argv[2], 'w'))\n",
    )

    spec = client.jobs[0]["spec"]["template"]["spec"]
    assert result.output == {"ok": True}
    assert spec["automountServiceAccountToken"] is False
    assert all("serviceaccount" not in json.dumps(v).lower() for v in spec.get("volumes", []))


def test_local_executable_fake_shares_host_filesystem(tmp_path) -> None:
    # A host subprocess cannot certify Pod filesystem isolation. Prove that
    # limitation using only a synthetic file, never this host's credentials.
    sentinel = tmp_path / "synthetic-host-file.txt"
    sentinel.write_text("test-only host filesystem marker", encoding="utf-8")
    result, _client = _run(
        tmp_path,
        "import json, pathlib, sys\n"
        f"path = pathlib.Path({str(sentinel)!r})\n"
        "json.dump({'marker': path.read_text()}, open(sys.argv[2], 'w'))\n",
    )

    assert result.exit_code == 0
    assert result.output == {"marker": "test-only host filesystem marker"}


def test_k8s_executable_cannot_read_proc_environ(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MINI_BANG_RUNNER_TOKEN", "proc-secret-token")
    result, _client = _run(
        tmp_path,
        (
            "import json, pathlib, sys\n"
            "needles = [b'proc-secret-token', b'MINI_BANG_RUNNER_TOKEN']\n"
            "hits = {}\n"
            "for name in ['/proc/self/environ', '/proc/1/environ']:\n"
            "    try:\n"
            "        body = pathlib.Path(name).read_bytes()\n"
            "    except Exception:\n"
            "        body = b''\n"
            "    hits[name] = any(item in body for item in needles)\n"
            "json.dump(hits, open(sys.argv[2], 'w'))\n"
        ),
    )

    assert result.exit_code == 0
    assert result.output == {"/proc/self/environ": False, "/proc/1/environ": False}


def test_k8s_executable_network_denied_by_default(tmp_path) -> None:
    result, client = _run(
        tmp_path,
        (
            "import json, sys, urllib.request\n"
            "denied = False\n"
            "try:\n"
            "    urllib.request.urlopen('http://127.0.0.1:9', timeout=0.2).read()\n"
            "except Exception:\n"
            "    denied = True\n"
            "json.dump({'network_denied': denied}, open(sys.argv[2], 'w'))\n"
        ),
    )
    manifests = render_k8s_manifests(_render_args())
    network = manifests["networkpolicy.yaml"]

    assert result.output == {"network_denied": True}
    assert client.jobs[0]["spec"]["template"]["metadata"]["labels"]["mini-bang-network"] == "disabled"
    assert client.jobs[0]["spec"]["template"]["spec"]["dnsPolicy"] == "None"
    assert client.jobs[0]["spec"]["template"]["spec"]["enableServiceLinks"] is False
    assert "name: mini-bang-executable-default-deny-egress" in network
    assert "mini-bang-role: executable" in network
    assert "mini-bang-network: disabled" in network
    assert "egress: []" in network


def test_k8s_executable_cannot_access_gold(tmp_path) -> None:
    result, _client = _run(
        tmp_path,
        (
            "import json, pathlib, sys\n"
            "paths = [\n"
            "  pathlib.Path('/workspace/src/mini_bang/tasks/raf_mechanism_prob/gold/default.json'),\n"
            "  pathlib.Path('/usr/local/lib/python3.13/site-packages/mini_bang/tasks/raf_mechanism_prob/gold/default.json'),\n"
            "]\n"
            "json.dump({'gold_readable': any(path.exists() and path.is_file() for path in paths)}, open(sys.argv[2], 'w'))\n"
        ),
    )

    assert result.output == {"gold_readable": False}


def test_k8s_executable_output_schema_validation(tmp_path) -> None:
    result, _client = _run(
        tmp_path,
        "import sys\nopen(sys.argv[2], 'w').write('not-json')\n",
        schema=lambda out: None,
    )

    assert result.exit_code == 1
    assert "output_error" in result.stderr


def test_k8s_executable_timeout_is_failed_submission(tmp_path) -> None:
    result, _client = _run(
        tmp_path,
        "import time\ntime.sleep(5)\n",
        limits=ExecutionLimits(timeout_seconds=1),
    )

    assert result.exit_code == 124
    assert result.timed_out is True


def test_k8s_executable_resource_limit_is_failed_submission(tmp_path) -> None:
    result, client = _run(
        tmp_path,
        "raise MemoryError('simulated resource exhaustion')\n",
        limits=ExecutionLimits(timeout_seconds=5, memory_mb=32, pids_limit=16),
    )
    container = client.jobs[0]["spec"]["template"]["spec"]["containers"][0]

    assert result.exit_code != 0
    assert container["resources"]["limits"]["memory"] == "32Mi"
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]


def test_k8s_jobs_have_ttl(tmp_path) -> None:
    _result, client = _run(tmp_path, "import json, sys\njson.dump({'ok': True}, open(sys.argv[2], 'w'))\n")

    assert client.jobs[0]["spec"]["ttlSecondsAfterFinished"] == 300


def _run(
    tmp_path,
    code: str,
    limits: ExecutionLimits | None = None,
    schema=None,
):
    store = FileSystemResultStore(tmp_path / "results")
    client = LocalExecutableK8sClient()
    runner = K8sExecutableRunner(
        config=K8sExecutableConfig(
            execution_mode="legacy_jobs",
            namespace="mini-bang",
            image="python:3.13-slim",
            result_store_config=store.describe(),
            polling_interval_seconds=0.01,
        ),
        result_store=store,
        k8s_client=client,
    )
    result = runner.run({"code": code}, {"run_id": "security-run"}, limits or ExecutionLimits(), schema)
    return result, client


def _probe_env_code(names: list[str]) -> str:
    return (
        "import json, os, pathlib, sys\n"
        f"names = {names!r}\n"
        "found = {name: os.environ[name] for name in names if os.environ.get(name)}\n"
        "try:\n"
        "    proc = pathlib.Path('/proc/self/environ').read_bytes()\n"
        "except Exception:\n"
        "    proc = b''\n"
        "has_secret = any(value.encode() in proc for value in found.values())\n"
        "json.dump({'found': found, 'proc_self_has_secret': has_secret}, open(sys.argv[2], 'w'))\n"
    )


def _env_names(client: LocalExecutableK8sClient) -> set[str]:
    container = client.jobs[0]["spec"]["template"]["spec"]["containers"][0]
    return {item["name"] for item in container.get("env", [])}


def _render_args() -> argparse.Namespace:
    return argparse.Namespace(
        namespace="mini-bang",
        executor_replicas=1,
        worker_image="mini-bang-worker:test",
        payload_store="minio",
        s3_bucket=None,
        s3_prefix=None,
        s3_endpoint_url=None,
        agent_image=None,
        with_minio=True,
    )
