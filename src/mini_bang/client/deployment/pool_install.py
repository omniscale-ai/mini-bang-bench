"""Operator-only startup apply/readiness; scientific requests never call kubectl."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time

import yaml


def _kubectl(arguments, *, timeout=60):
    return subprocess.run(["kubectl", *arguments], check=True, capture_output=True,
                          text=True, timeout=timeout)


def _existing_token(namespace):
    reply = _kubectl(["-n", namespace, "get", "secret", "mini-bang-pool-auth",
                      "--ignore-not-found", "-o", "json"])
    if not reply.stdout.strip():
        return False
    try:
        value = json.loads(reply.stdout)
        token = base64.b64decode(value["data"]["token"], validate=True).decode()
        if value.get("immutable") is not True or len(token) < 24:
            raise ValueError("invalid immutable token")
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise ValueError("Existing pool auth Secret must have an immutable valid token") from error
    return True


def _existing_admin(namespace):
    reply = _kubectl(["-n", namespace, "get", "secret", "mini-bang-runner-admin",
                      "--ignore-not-found", "-o", "json"])
    if not reply.stdout.strip():
        return False
    try:
        value = json.loads(reply.stdout)
        token = base64.b64decode(value["data"]["MINI_BANG_RUNNER_ADMIN_TOKEN"], validate=True).decode()
        if not token:
            raise ValueError("empty token")
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise ValueError("Existing runner admin Secret has no usable token") from error
    # A legacy mutable token is retained too; reinstall never rotates authority.
    return True


def _pending_summary(namespace):
    try:
        reply = _kubectl(["-n", namespace, "get", "pods", "-l", "mini-bang-pool-member=true", "-o", "json"])
        rows = json.loads(reply.stdout).get("items", [])
        summary = []
        for row in rows:
            state = row.get("status", {})
            ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in state.get("conditions", []))
            if ready:
                continue
            reasons = [c.get("reason") for c in state.get("conditions", []) if c.get("status") == "False"]
            reasons += [c.get("state", {}).get("waiting", {}).get("reason") for c in state.get("containerStatuses", [])]
            summary.append(row["metadata"]["name"] + ": " + ", ".join(str(x) for x in reasons if x))
        return "; ".join(summary) or "fixed Pods have not become Ready"
    except Exception:
        return "could not read fixed Pod startup diagnostics"


def _readiness_seconds(args):
    seconds = getattr(args, "pool_ready_timeout", 600)
    if type(seconds) is not int or seconds <= 0:
        raise ValueError("Pool readiness timeout must be a positive integer")
    return seconds


def wait_pool_ready(args, manifests):
    seconds = _readiness_seconds(args)
    deadline = time.monotonic() + seconds
    namespace = args.namespace
    objects = [obj for body in manifests.values() for obj in yaml.safe_load_all(body) if obj]
    # Rollout completion first proves that every desired replica actually exists;
    # a selector wait alone can succeed while only a subset has been created.
    workloads = [(obj["kind"], obj["metadata"]["name"]) for obj in objects
                 if obj["kind"] in {"Deployment", "StatefulSet"}]
    workloads.sort(key=lambda item: item[1] == "mini-bang-pool")
    try:
        for kind, name in workloads:
            remaining = max(1, int(deadline - time.monotonic()))
            _kubectl(["-n", namespace, "rollout", "status", f"{kind.lower()}/{name}",
                      f"--timeout={remaining}s"], timeout=remaining + 5)
        remaining = max(1, int(deadline - time.monotonic()))
        _kubectl(["-n", namespace, "wait", "--for=condition=Ready", "pods", "-l",
                  "mini-bang-pool-member=true", f"--timeout={remaining}s"], timeout=remaining + 5)
        if "minio-bootstrap.yaml" in manifests:
            remaining = max(1, int(deadline - time.monotonic()))
            _kubectl(["-n", namespace, "wait", "--for=jsonpath={.status.phase}=Succeeded",
                      "pod/mini-bang-minio-bootstrap", f"--timeout={remaining}s"], timeout=remaining + 5)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Persistent pool is not ready: " + _pending_summary(namespace)) from error


def install_k8s_manifests(args, manifests=None):
    """Apply startup resources, retain the existing auth identity, await all slots."""
    _readiness_seconds(args)
    if manifests is None:
        from .k8s import render_k8s_manifests
        manifests = render_k8s_manifests(args)
    with tempfile.TemporaryDirectory(prefix="mini-bang-pool-install-") as folder:
        root = Path(folder)
        root.chmod(0o700)
        if getattr(args, "create_namespace", False) and "namespace.yaml" in manifests:
            path = root / "namespace.yaml"
            path.write_text(manifests["namespace.yaml"])
            _kubectl(["apply", "-f", str(path)])
            path.unlink()
        preserve_token = _existing_token(args.namespace)
        preserve_admin = "secret-runner.yaml" in manifests and _existing_admin(args.namespace)
        for name, body in manifests.items():
            if (name == "namespace.yaml" or name == "pool-auth.yaml" and preserve_token
                    or name == "secret-runner.yaml" and preserve_admin):
                continue
            path = root / name
            path.write_text(body)
            path.chmod(0o600)
        _kubectl(["apply", "-f", str(root)])
    # Remove only obsolete broad grants from the previous per-request Job mode.
    for kind, name in (("rolebinding", "mini-bang-runner-job-controller"),
                       ("role", "mini-bang-job-controller")):
        _kubectl(["-n", args.namespace, "delete", kind, name, "--ignore-not-found"])
    wait_pool_ready(args, manifests)
    return {"namespace": args.namespace, "pool_ready": True,
            "pool_token_preserved": preserve_token, "admin_token_preserved": preserve_admin}
