from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from mini_bang.agent_runtime import get_json, post_json, with_session_id
from mini_bang.client.deployment.compose import effective_compose_payload_store, render_compose_yaml
from mini_bang.client.deployment.k8s import effective_k8s_payload_store, render_k8s_agent_job, render_k8s_manifests
from mini_bang.client.paths import ARTIFACTS_ROOT, CACHE_ROOT, PAYLOADS_ROOT
from mini_bang.client.cli.config import load_config, nested_get
from mini_bang.client.cli.errors import CliError, EXIT_CONFIG, EXIT_SUCCESS, EXIT_USAGE
from mini_bang.client.cli.io import emit, find_free_port, get_cache_dir, safe_name, utc_now
from mini_bang.client.cli.records import run_output_summary, save_run_outputs
from mini_bang.client.cli.security import (
    check_command_or_raise,
    check_docker_or_raise,
    check_docker_runtime_or_raise,
    command_compatibility,
    docker_compose_cmd,
    validate_or_raise,
)
from mini_bang.framework.records import RunRecord


def cmd_compose_up(args: argparse.Namespace) -> int:
    config = load_config()
    payload_store = effective_compose_payload_store(args)
    validate_or_raise(
        command_compatibility(
            mode="compose-local",
            agent_placement="compose-stack",
            backend="local",
            engine="executor_replicas",
            payload_store=payload_store,
            config=config,
            command="compose up",
        )
    )
    if args.executor_workers < 1:
        raise CliError("--executor-workers must be at least 1", EXIT_USAGE)
    if not os.getenv("MINI_BANG_RUNNER_ADMIN_TOKEN"):
        raise CliError("set MINI_BANG_RUNNER_ADMIN_TOKEN: the compose runner creates sessions only for it", EXIT_CONFIG)
    check_docker_or_raise()
    check_docker_runtime_or_raise()
    cache_dir = get_cache_dir() / "compose"
    cache_dir.mkdir(parents=True, exist_ok=True)
    compose_path = cache_dir / "docker-compose.yml"
    compose_path.write_text(render_compose_yaml(args), encoding="utf-8")
    cmd = [*docker_compose_cmd(), "-f", str(compose_path), "up"]
    if args.detach:
        cmd.append("-d")
    exit_code = subprocess.call(cmd)
    if exit_code != 0:
        raise CliError(f"docker compose up failed with exit code {exit_code}", EXIT_CONFIG)
    payload = {
        "compose_file": str(compose_path),
        "with_minio": args.with_minio,
        "payload_store": payload_store,
        "runner_url": f"http://127.0.0.1:{args.port}",
        "detached": args.detach,
    }
    emit(payload, args.format, lambda: f"Compose stack started using {compose_path}")
    return EXIT_SUCCESS

def cmd_compose_down(args: argparse.Namespace) -> int:
    check_docker_or_raise()
    compose_path = get_cache_dir() / "compose" / "docker-compose.yml"
    if not compose_path.exists():
        raise CliError(f"compose file not found: {compose_path}", EXIT_CONFIG)
    cmd = [*docker_compose_cmd(), "-f", str(compose_path), "down"]
    if args.volumes:
        cmd.append("--volumes")
    exit_code = subprocess.call(cmd)
    if exit_code != 0:
        raise CliError(f"docker compose down failed with exit code {exit_code}", EXIT_CONFIG)
    removed: list[str] = []
    if args.remove_artifacts:
        config = load_config()
        for path in [
            Path(nested_get(config, "payload.filesystem.root", f"./{PAYLOADS_ROOT.as_posix()}")),
            Path(nested_get(config, "artifacts.root", f"./{ARTIFACTS_ROOT.as_posix()}")),
        ]:
            if path.exists():
                shutil.rmtree(path)
                removed.append(str(path))
    payload = {"compose_file": str(compose_path), "removed": removed}
    emit(payload, args.format, lambda: "Compose stack stopped")
    return EXIT_SUCCESS

def cmd_k8s_render(args: argparse.Namespace) -> int:
    config = load_config()
    payload_store = effective_k8s_payload_store(args)
    validate_or_raise(
        command_compatibility(
            mode="k8s-official",
            agent_placement="k8s-stack",
            backend="cluster",
            engine="executor_replicas",
            payload_store=("filesystem-pvc" if payload_store == "filesystem" else payload_store),
            config=config,
            command="k8s render",
        )
    )
    manifests = render_k8s_manifests(args)
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, body in manifests.items():
            path = output / name
            path.write_text(body, encoding="utf-8")
            paths.append(str(path))
        emit(
            {"output": str(output), "files": paths},
            args.format,
            lambda: "\n".join(paths),
        )
    else:
        joined = "\n---\n".join(manifests.values())
        emit({"manifests": manifests}, args.format, lambda: joined)
    return EXIT_SUCCESS

def cmd_k8s_install(args: argparse.Namespace) -> int:
    config = load_config()
    payload_store = effective_k8s_payload_store(args)
    validate_or_raise(
        command_compatibility(
            mode="k8s-official",
            agent_placement="k8s-stack",
            backend="cluster",
            engine="executor_replicas",
            payload_store=("filesystem-pvc" if payload_store == "filesystem" else payload_store),
            config=config,
            command="k8s install",
            allow_insecure=args.i_know_this_is_insecure,
        )
    )
    check_command_or_raise("kubectl")
    from mini_bang.client.deployment.pool_install import install_k8s_manifests
    install_k8s_manifests(args)
    emit(
        {"namespace": args.namespace, "executor_replicas": args.executor_replicas, "payload_store": payload_store},
        args.format,
        lambda: f"Installed mini-bang stack in namespace {args.namespace}",
    )
    return EXIT_SUCCESS

def _historical_k8s_submit(args: argparse.Namespace) -> int:
    if args.agent_command and (args.agent or args.agent_file):
        raise CliError("--agent-command cannot be combined with --agent/--agent-file", EXIT_USAGE)
    config = load_config()
    validate_or_raise(
        command_compatibility(
            mode="k8s-official",
            agent_placement="k8s-job",
            backend="cluster",
            engine="executor_replicas",
            payload_store=str(nested_get(config, "payload.mode", "s3")),
            config=config,
            command="k8s submit",
            allow_insecure=args.i_know_this_is_insecure,
        )
    )
    check_command_or_raise("kubectl")
    runner_url, port_forward = _runner_url(args)
    secret_name: str | None = None
    job_name: str | None = None
    try:
        created = post_json(
            f"{runner_url}/v1/runs",
            {"task_id": args.task, "instance_id": args.instance},
            token=args.admin_token or os.getenv("MINI_BANG_RUNNER_ADMIN_TOKEN", "mini-bang-admin"),
        )
        session_id = str(created["session_id"])
        session_token = str(created["session_token"])
        secret_name = f"mini-bang-runner-session-{safe_name(session_id)}"
        _apply_session_secret(args.namespace, secret_name, session_token)
        job_args = argparse.Namespace(**vars(args))
        job_args.run_id = session_id
        job_args.session_secret_name = secret_name
        job_yaml = render_k8s_agent_job(job_args)
        process = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=job_yaml,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            raise CliError(process.stderr.strip() or "kubectl apply failed", EXIT_CONFIG)
        job_name = f"mini-bang-agent-{safe_name(session_id)}"
        _bind_secret_to_job(args.namespace, secret_name, job_name)
        if args.wait:
            _wait_for_agent_job(args.namespace, job_name, args.timeout or 600)
            if args.logs:
                subprocess.call(["kubectl", "-n", args.namespace, "logs", f"job/{job_name}"])
            result = _wait_for_runner_result(
                runner_url=runner_url,
                session_id=session_id,
                session_token=args.admin_token or os.getenv("MINI_BANG_RUNNER_ADMIN_TOKEN", "mini-bang-admin"),
                timeout_seconds=args.timeout or 600,
            )
            record = dict(result["run_record"])
            record.setdefault("backend_metadata", {}).setdefault("cli", {})
            record["backend_metadata"]["cli"].update(
                {
                    "mode": "k8s-official",
                    "agent_image": args.agent_image,
                    "agent": args.agent or args.agent_file or args.agent_command,
                    "created_at": utc_now(),
                    "security": {
                        "agent_gets_runner_token": True,
                        "agent_gets_storage_credentials": False,
                        "agent_gets_gold_refs": False,
                        "agent_gets_mcp_endpoint": False,
                    },
                }
            )
            run_dir = save_run_outputs(
                RunRecord(**record),
                args.out,
                mode="k8s-official",
                allow_pickle_payloads=args.i_know_this_is_insecure,
            )
            output = run_output_summary(record, run_dir)
            emit(output, args.format, lambda: f"Saved Kubernetes run {session_id} to {run_dir}")
            return EXIT_SUCCESS if record.get("score_summary", {}).get("success") else EXIT_CONFIG
        emit(
            {
                "session_id": session_id,
                "job_name": job_name,
                "kubectl": process.stdout.strip(),
                "manifest": job_yaml,
            },
            args.format,
            lambda: process.stdout.strip(),
        )
    finally:
        if getattr(args, "wait", False) and os.getenv("MINI_BANG_KEEP_K8S_OBJECTS") != "1":
            if job_name:
                _delete_k8s_job(args.namespace, job_name)
            if secret_name:
                _delete_k8s_secret(args.namespace, secret_name)
        if port_forward is not None:
            port_forward.terminate()
            try:
                port_forward.wait(timeout=5)
            except subprocess.TimeoutExpired:
                port_forward.kill()
    return EXIT_SUCCESS


def _runner_url(args: argparse.Namespace) -> tuple[str, subprocess.Popen | None]:
    if args.runner_url:
        return args.runner_url.rstrip("/"), None
    port = find_free_port()
    process = subprocess.Popen(
        [
            "kubectl",
            "-n",
            args.namespace,
            "port-forward",
            "svc/mini-bang-runner",
            f"{port}:8765",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise CliError(stderr.strip() or "kubectl port-forward exited", EXIT_CONFIG)
        try:
            get_json(f"{url}/health")
            return url, process
        except Exception:
            time.sleep(0.25)
    process.terminate()
    raise CliError("runner service did not become reachable through kubectl port-forward", EXIT_CONFIG)


def _apply_session_secret(namespace: str, name: str, token: str) -> None:
    manifest = (
        "apiVersion: v1\n"
        "kind: Secret\n"
        "metadata:\n"
        f"  name: {name}\n"
        f"  namespace: {namespace}\n"
        "  labels:\n"
        "    app: mini-bang-agent\n"
        "    mini-bang-role: session-token\n"
        "type: Opaque\n"
        "stringData:\n"
        f"  token: {token}\n"
    )
    process = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise CliError(process.stderr.strip() or "failed to create runner session secret", EXIT_CONFIG)


def _bind_secret_to_job(namespace: str, secret_name: str, job_name: str) -> None:
    """Make the agent Job the garbage-collection parent of its session secret."""
    uid_proc = subprocess.run(
        ["kubectl", "-n", namespace, "get", "job", job_name, "-o", "jsonpath={.metadata.uid}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if uid_proc.returncode != 0 or not uid_proc.stdout.strip():
        raise CliError(uid_proc.stderr.strip() or "failed to read agent job uid", EXIT_CONFIG)
    patch = {
        "metadata": {
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": job_name,
                    "uid": uid_proc.stdout.strip(),
                    "blockOwnerDeletion": False,
                }
            ]
        }
    }
    proc = subprocess.run(
        ["kubectl", "-n", namespace, "patch", "secret", secret_name, "--type=merge", "-p", json.dumps(patch)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CliError(proc.stderr.strip() or "failed to bind session secret to the agent job", EXIT_CONFIG)


def _delete_k8s_job(namespace: str, name: str) -> None:
    subprocess.run(
        ["kubectl", "-n", namespace, "delete", "job", name, "--ignore-not-found", "--wait=false"],
        text=True,
        capture_output=True,
        check=False,
    )


def _delete_k8s_secret(namespace: str, name: str) -> None:
    subprocess.run(
        ["kubectl", "-n", namespace, "delete", "secret", name, "--ignore-not-found", "--wait=false"],
        text=True,
        capture_output=True,
        check=False,
    )


def _wait_for_agent_job(namespace: str, job_name: str, timeout_seconds: int) -> None:
    process = subprocess.run(
        [
            "kubectl",
            "-n",
            namespace,
            "wait",
            "--for=condition=complete",
            f"job/{job_name}",
            f"--timeout={timeout_seconds}s",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise CliError(process.stderr.strip() or process.stdout.strip() or "agent job failed", EXIT_CONFIG)


def _wait_for_runner_result(
    *,
    runner_url: str,
    session_id: str,
    session_token: str,
    timeout_seconds: int,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = get_json(
            with_session_id(f"{runner_url}/v1/result", session_id),
            token=session_token,
        )
        status = response.get("status")
        if status == "succeeded":
            return response
        if status == "failed":
            raise CliError(str(response.get("error") or "scoring failed"), EXIT_CONFIG)
        time.sleep(1)
    raise CliError(f"runner result for session {session_id} timed out", EXIT_CONFIG)
