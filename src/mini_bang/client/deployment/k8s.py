from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import secrets
import time
from typing import Any

from mini_bang.client.deployment.templating import render_template, runtime_setting_entries, yaml_string
from mini_bang.client.deployment.persistent_pool import render_pool_manifests, _resources as pool_resources


def render_k8s_manifests(args: argparse.Namespace) -> dict[str, str]:
    namespace = args.namespace
    payload_store = effective_k8s_payload_store(args)
    values = {
        "namespace": namespace,
        "executor_replicas": args.executor_replicas,
        "worker_image": getattr(args, "worker_image", None) or "mini-bang-worker:latest",
        "minio_image": yaml_string(getattr(args, "minio_image", None) or "mini-bang-storage:2025-09-07"),
        "executable_image": getattr(args, "executable_image", None) or "mini-bang-executable:latest",
        "agent_image": yaml_string(getattr(args, "agent_image", None) or ""),
        "payload_store": payload_store,
        "result_store": "s3" if payload_store in {"s3", "minio"} else "filesystem",
        "s3_bucket": args.s3_bucket or "mini-bang-payloads",
        "minio_bucket": yaml_string(args.s3_bucket or "mini-bang-payloads"),
        "s3_prefix": args.s3_prefix or "runs",
        "s3_endpoint_url": args.s3_endpoint_url
        or ("http://mini-bang-minio:9000" if payload_store == "minio" else ""),
    }
    package_settings = extension_settings(args, values["s3_endpoint_url"])
    values["runtime_settings"] = "\n".join(filter(None, (package_settings, runtime_settings(args, package_settings))))
    minio_resources = pool_resources(args, "minio")
    values.update(minio_cpu_request=minio_resources["requests"]["cpu"],
                  minio_cpu_limit=minio_resources["limits"]["cpu"],
                  minio_memory=minio_resources["limits"]["memory"])

    storage_secret_envfrom = ""
    if payload_store in {"s3", "minio"}:
        storage_secret_envfrom = render_template("k8s/partials/storage-secret-envfrom.yaml")

    payload_volume_mount = ""
    payload_volumes = ""
    if payload_store == "filesystem":
        payload_volume_mount = render_template("k8s/partials/payload-volume-mount.yaml").replace("          volumeMounts:\n", "", 1)
        payload_volumes = render_template("k8s/partials/payload-volumes.yaml").replace("      volumes:\n", "", 1)

    workload_values = {
        **values,
        "storage_secret_envfrom": storage_secret_envfrom,
        "payload_volume_mount": payload_volume_mount,
        "payload_volumes": payload_volumes,
    }

    manifests = {
        "namespace.yaml": render_template("k8s/namespace.yaml", values),
        "serviceaccounts.yaml": render_template("k8s/serviceaccounts.yaml", values),
        "configmap.yaml": render_template("k8s/configmap.yaml", values),
        "secret-runner.yaml": render_template(
            "k8s/secret-runner.yaml",
            {
                "namespace": namespace,
                "runner_admin_token": yaml_string(
                    os.getenv("MINI_BANG_RUNNER_ADMIN_TOKEN") or secrets.token_urlsafe(32)
                ),
            },
        ),
        "runner.yaml": render_template("k8s/runner.yaml", workload_values),
        "networkpolicy.yaml": render_template("k8s/networkpolicy.yaml", values),
    }
    judge = judge_settings(args)
    if judge:
        manifests["secret-judge.yaml"] = render_template(
            "k8s/secret-judge.yaml", {"namespace": namespace, "judge_settings": judge})

    if payload_store in {"s3", "minio"}:
        manifests["secret-storage.yaml"] = render_template(
            "k8s/secret-storage.yaml",
            {
                "namespace": namespace,
                "aws_access_key_id": yaml_string("mini-bang" if payload_store == "minio" else ""),
                "aws_secret_access_key": yaml_string("mini-bang-secret" if payload_store == "minio" else ""),
            },
        )

    if payload_store == "filesystem":
        manifests["payload-pvc.yaml"] = render_template("k8s/payload-pvc.yaml", values)

    if payload_store == "minio":
        manifests["minio.yaml"] = render_template("k8s/minio.yaml", values)
        manifests["minio-bootstrap.yaml"] = render_template("k8s/minio-bootstrap.yaml", values)
    manifests.update(render_pool_manifests(args, values, storage_secret_envfrom))
    import yaml
    state = {"apiVersion": "v1", "kind": "PersistentVolumeClaim",
        "metadata": {"name": "mini-bang-runner-state", "namespace": namespace},
        "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "1Gi"}}}}
    storage_class = getattr(args, "pool_storage_class", None)
    if storage_class:
        state["spec"]["storageClassName"] = storage_class
    manifests["runner-state.yaml"] = yaml.safe_dump(state, sort_keys=False)
    if payload_store == "minio":
        storage = {"apiVersion": "v1", "kind": "PersistentVolumeClaim",
            "metadata": {"name": "mini-bang-minio-data", "namespace": namespace},
            "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {
                "storage": getattr(args, "minio_storage_size", None) or "20Gi"}}}}
        if storage_class:
            storage["spec"]["storageClassName"] = storage_class
        manifests["minio-storage.yaml"] = yaml.safe_dump(storage, sort_keys=False)

    return manifests


def extension_settings(args: argparse.Namespace, s3_endpoint_url: str) -> str:
    """ConfigMap lines from the deployment options that simulator and task packages register."""
    from mini_bang.client.cli.extensions import extension_deployment_settings
    entries = extension_deployment_settings(args, s3_endpoint_url=s3_endpoint_url)
    return "\n".join(f"  {key}: {json.dumps(value)}" for key, value in entries.items())


def runtime_settings(args: argparse.Namespace, other_settings: str = "") -> str:
    """ConfigMap lines from repeated `--runtime-setting NAME=VALUE` options.

    A new simulator's deployment settings (its config.json `runtime_environment`)
    reach the runner and workers this way, without a simulator-specific option.
    """
    entries = runtime_setting_entries(getattr(args, "runtime_setting", None),
                                      render_template("k8s/configmap.yaml", {}) + "\n" + other_settings)
    return "\n".join(f"  {key}: {json.dumps(value)}" for key, value in entries.items())


def judge_settings(args: argparse.Namespace) -> str:
    """Description-judge authority for runner-side scoring only; agents never receive it."""
    from mini_bang.client.cli.scoring_environment import PRIVATE_JUDGE_ENVIRONMENT
    path = getattr(args, "judge_environment", None)
    if not path:
        return ""
    with open(path, encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict) or not set(values) <= set(PRIVATE_JUDGE_ENVIRONMENT):
        raise ValueError("--judge-environment must be a JSON object with keys from: "
                         + ", ".join(PRIVATE_JUDGE_ENVIRONMENT))
    return "\n".join(f"  {key}: {yaml_string(value)}" for key, value in sorted(values.items()))


def effective_k8s_payload_store(args: argparse.Namespace) -> str:
    if getattr(args, "with_minio", False):
        return "minio"
    return args.payload_store


def render_k8s_agent_job(args: argparse.Namespace) -> str:
    run_id = args.run_id or safe_name(f"{args.task}-{int(time.time())}")
    container_args = ""
    container_command = ""
    if args.agent_command:
        container_args = render_template(
            "k8s/partials/agent-args.yaml",
            {"args_json": json.dumps(shlex.split(args.agent_command))},
        )
    elif getattr(args, "agent", None) or getattr(args, "agent_file", None):
        container_command = render_template("k8s/partials/agent-command.yaml")

    return render_template(
        "k8s/agent-job.yaml",
        {
            "run_id": run_id,
            "namespace": args.namespace,
            "agent_image": yaml_string(args.agent_image),
            "session_secret_name": getattr(args, "session_secret_name", None)
            or f"mini-bang-runner-session-{safe_name(run_id)}",
            "task_id": yaml_string(args.task),
            "instance_id": yaml_string(args.instance or "default"),
            "agent_env": _agent_env(args),
            "container_command": container_command,
            "container_args": container_args,
            "resources": _resources(args),
        },
    )


def _agent_env(args: argparse.Namespace) -> str:
    entries: list[str] = []
    if getattr(args, "agent", None):
        entries.append(_agent_env_var("MINI_BANG_AGENT", args.agent))
    if getattr(args, "agent_file", None):
        entries.append(_agent_env_var("MINI_BANG_AGENT_FILE", args.agent_file))
    return "".join(entries)


def _agent_env_var(name: str, value: str) -> str:
    return render_template(
        "k8s/partials/agent-env-var.yaml",
        {"name": name, "value": yaml_string(value)},
    )


def _resources(args: argparse.Namespace) -> str:
    limits: dict[str, Any] = {}
    if args.cpu:
        limits["cpu"] = args.cpu
    if args.memory:
        limits["memory"] = args.memory
    if args.gpu is not None:
        limits["nvidia.com/gpu"] = str(args.gpu)
    if not limits:
        limits = {"cpu": "1", "memory": "1Gi"}

    requests = {key: value for key, value in limits.items() if key != "nvidia.com/gpu"}
    requests_section = ""
    if requests:
        requests_section = render_template(
            "k8s/partials/resource-requests.yaml",
            {"requests": _resource_entries(requests)},
        )
    return render_template(
        "k8s/partials/resources.yaml",
        {
            "limits": _resource_entries(limits),
            "requests_section": requests_section,
        },
    )


def _resource_entries(values: dict[str, Any]) -> str:
    return "".join(
        render_template(
            "k8s/partials/resource-entry.yaml",
            {"key": key, "value": yaml_string(value)},
        )
        for key, value in values.items()
    )


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return cleaned or "unnamed-run"
