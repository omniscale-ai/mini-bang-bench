"""Operator startup reuses auth identity and cannot return before all Pods Ready."""
import base64
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from mini_bang.client.deployment import pool_install


def manifests():
    return {"namespace.yaml": yaml.safe_dump({"kind": "Namespace", "metadata": {"name": "test"}}),
        "pool-auth.yaml": yaml.safe_dump({"kind": "Secret", "metadata": {"name": "mini-bang-pool-auth"},
                                          "immutable": True, "stringData": {"token": "new-unused-random-token-with-sufficient-length"}}),
        "workers.yaml": yaml.safe_dump_all([
            {"kind": "StatefulSet", "metadata": {"name": "mini-bang-observation"}},
            {"kind": "Deployment", "metadata": {"name": "mini-bang-pool"}},
            {"kind": "Deployment", "metadata": {"name": "mini-bang-runner"}}]),
        "minio-bootstrap.yaml": yaml.safe_dump({"kind": "Pod", "metadata": {"name": "mini-bang-minio-bootstrap"}})}


@pytest.mark.parametrize("existing", [False, True])
def test_install_only_applies_startup_resources_and_preserves_existing_secret(tmp_path, monkeypatch, existing):
    calls, applied = [], []
    def kubectl(arguments, **_):
        calls.append(arguments)
        if "secret" in arguments:
            value = {"immutable": True, "data": {"token": base64.b64encode(b"original-private-startup-token-with-sufficient-length").decode()}}
            return SimpleNamespace(stdout=json.dumps(value) if existing else "")
        if arguments[:2] == ["apply", "-f"]:
            path = Path(arguments[2])
            sources = [path] if path.is_file() else list(path.glob("*.yaml"))
            applied.extend(obj for source in sources for obj in yaml.safe_load_all(source.read_text()))
        return SimpleNamespace(stdout="")
    monkeypatch.setattr(pool_install, "_kubectl", kubectl)
    result = pool_install.install_k8s_manifests(SimpleNamespace(namespace="test", create_namespace=True,
        pool_ready_timeout=60), manifests())
    assert result["pool_ready"] is True and result["pool_token_preserved"] is existing
    secrets = [obj for obj in applied if obj["kind"] == "Secret"]
    assert len(secrets) == (0 if existing else 1)
    assert applied[0]["kind"] == "Namespace"
    assert [args[4] for args in calls if args[2:4] == ["rollout", "status"]] == [
        "statefulset/mini-bang-observation", "deployment/mini-bang-runner", "deployment/mini-bang-pool"]
    ready = next(args for args in calls if "--for=condition=Ready" in args)
    assert ready[ready.index("-l") + 1] == "mini-bang-pool-member=true"
    assert any("--for=jsonpath={.status.phase}=Succeeded" in args for args in calls)
    deletes = [args for args in calls if "delete" in args]
    assert deletes == [["-n", "test", "delete", "rolebinding", "mini-bang-runner-job-controller", "--ignore-not-found"],
                       ["-n", "test", "delete", "role", "mini-bang-job-controller", "--ignore-not-found"]]


def test_invalid_existing_secret_fails_without_applying_replacement(monkeypatch):
    def kubectl(arguments, **_):
        assert "secret" in arguments
        return SimpleNamespace(stdout=json.dumps({"immutable": False, "data": {"token": "invalid"}}))
    monkeypatch.setattr(pool_install, "_kubectl", kubectl)
    with pytest.raises(ValueError, match="immutable valid token"):
        pool_install.install_k8s_manifests(SimpleNamespace(namespace="test", create_namespace=False), manifests())


def test_reinstall_preserves_legacy_mutable_admin_token(monkeypatch):
    applied = []
    def kubectl(arguments, **_):
        if "secret" in arguments:
            if "mini-bang-runner-admin" in arguments:
                value = {"immutable": False, "data": {"MINI_BANG_RUNNER_ADMIN_TOKEN": base64.b64encode(b"legacy-admin").decode()}}
                return SimpleNamespace(stdout=json.dumps(value))
            return SimpleNamespace(stdout="")
        if arguments[:2] == ["apply", "-f"]:
            path = Path(arguments[2])
            applied.extend(obj for source in path.glob("*.yaml") for obj in yaml.safe_load_all(source.read_text()))
        return SimpleNamespace(stdout="")
    configured = manifests()
    configured["secret-runner.yaml"] = yaml.safe_dump({"kind": "Secret", "metadata": {"name": "mini-bang-runner-admin"},
        "immutable": True, "stringData": {"MINI_BANG_RUNNER_ADMIN_TOKEN": "new-token-must-not-replace-existing"}})
    monkeypatch.setattr(pool_install, "_kubectl", kubectl)
    result = pool_install.install_k8s_manifests(SimpleNamespace(namespace="test", create_namespace=False,
        pool_ready_timeout=60), configured)
    assert result["admin_token_preserved"] is True
    assert not any(obj["metadata"]["name"] == "mini-bang-runner-admin" for obj in applied)


def test_pending_capacity_is_reported_as_startup_failure_not_ready(monkeypatch):
    def kubectl(arguments, **_):
        if "rollout" in arguments:
            raise subprocess.CalledProcessError(1, ["kubectl", *arguments])
        return SimpleNamespace(stdout=json.dumps({"items": [{"metadata": {"name": "mini-bang-observation-0"},
            "status": {"conditions": [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}]}}]}))
    monkeypatch.setattr(pool_install, "_kubectl", kubectl)
    with pytest.raises(RuntimeError, match="mini-bang-observation-0.*Unschedulable"):
        pool_install.wait_pool_ready(SimpleNamespace(namespace="test", pool_ready_timeout=2), manifests())


def test_invalid_readiness_configuration_cannot_create_resources(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Startup configuration must be checked before any Kubernetes request")
    monkeypatch.setattr(pool_install, "_kubectl", forbidden)
    with pytest.raises(ValueError, match="positive integer"):
        pool_install.install_k8s_manifests(SimpleNamespace(namespace="test", pool_ready_timeout=0), manifests())
