from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from mini_bang.client.cli.config import nested_get
from mini_bang.client.cli.errors import CliError, EXIT_CONFIG, EXIT_SECURITY
from mini_bang.config_validation import (
    CompatibilityConfig,
    CompatibilityFinding,
    has_errors,
    validate_compatibility,
)


def project_config_to_compatibility(config: dict[str, Any]) -> CompatibilityConfig:
    return CompatibilityConfig(
        mode=str(config.get("mode", "dev-local")),
        agent_placement=str(nested_get(config, "agent.placement", "in-process")),
        backend=str(nested_get(config, "backend.kind", "local")),
        engine=(
            str(nested_get(config, "backend.engine"))
            if nested_get(config, "backend.engine") is not None
            else None
        ),
        payload_store=str(nested_get(config, "payload.mode", "inline")),
        allow_pickle_payloads=bool(nested_get(config, "security.allow_pickle_payloads", False)),
        allow_raw_file_refs=bool(nested_get(config, "security.allow_raw_file_refs", False)),
        mcp_endpoint_exposed_to_agent=bool(
            nested_get(config, "security.mcp_endpoint_exposed_to_agent", False)
        ),
        trusted_dev_mode=bool(nested_get(config, "security.trusted_dev_mode", False)),
        command="config",
    )

def command_compatibility(
    *,
    mode: str,
    agent_placement: str,
    payload_store: str,
    config: dict[str, Any],
    backend: str = "local",
    engine: str | None = "sequential",
    command: str,
    allow_insecure: bool = False,
) -> CompatibilityConfig:
    return CompatibilityConfig(
        mode=mode,
        agent_placement=agent_placement,
        backend=backend,
        engine=engine,
        payload_store=payload_store,
        allow_pickle_payloads=(
            False
            if allow_insecure
            else bool(nested_get(config, "security.allow_pickle_payloads", False))
        ),
        allow_raw_file_refs=(
            False
            if allow_insecure
            else bool(nested_get(config, "security.allow_raw_file_refs", False))
        ),
        mcp_endpoint_exposed_to_agent=(
            False
            if allow_insecure
            else bool(nested_get(config, "security.mcp_endpoint_exposed_to_agent", False))
        ),
        trusted_dev_mode=bool(nested_get(config, "security.trusted_dev_mode", False)),
        command=command,
    )

def validate_or_raise(compatibility: CompatibilityConfig) -> list[CompatibilityFinding]:
    findings = validate_compatibility(compatibility)
    if has_errors(findings):
        errors = [item for item in findings if item.level == "error"]
        raise CliError(render_compatibility_errors(errors), EXIT_SECURITY)
    return findings

def render_compatibility_errors(findings: list[CompatibilityFinding]) -> str:
    return "configuration is not compatible: " + "; ".join(
        f"{item.code}: {item.message}" for item in findings
    )

def add_compatibility_findings(add: Any, findings: list[CompatibilityFinding]) -> None:
    for finding in findings:
        status = "ERR" if finding.level == "error" else "WARN"
        add(status, f"{finding.code}: {finding.message}", **finding.details)

def check_docker(add: Any) -> None:
    if shutil.which("docker") is None:
        add("ERR", "Docker is not installed or not on PATH")
        return
    try:
        cmd = docker_compose_cmd()
        add("OK", "Docker Compose available")
        add("OK", f"Docker Compose command: {' '.join(cmd)}")
    except CliError as exc:
        add("ERR", str(exc))
    try:
        check_docker_runtime_or_raise()
        add("OK", "Docker daemon available")
    except CliError as exc:
        add("ERR", str(exc))

def check_docker_or_raise() -> None:
    if shutil.which("docker") is None:
        raise CliError("Docker is not installed or not on PATH", EXIT_CONFIG)
    docker_compose_cmd()

def docker_compose_cmd() -> list[str]:
    process = subprocess.run(
        ["docker", "compose", "version"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode == 0:
        return ["docker", "compose"]
    if shutil.which("docker-compose") is not None:
        legacy = subprocess.run(
            ["docker-compose", "version"],
            text=True,
            capture_output=True,
            check=False,
        )
        if legacy.returncode == 0:
            return ["docker-compose"]
    raise CliError("Docker Compose is unavailable", EXIT_CONFIG)

def check_docker_runtime_or_raise() -> None:
    if shutil.which("docker") is None:
        raise CliError("Docker is not installed or not on PATH", EXIT_CONFIG)
    process = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise CliError(
            "Docker daemon is not reachable. Start Docker Desktop, Colima, or your Docker service"
            f"{suffix}",
            EXIT_CONFIG,
        )

def check_command(add: Any, command: str) -> None:
    if shutil.which(command):
        add("OK", f"{command} available")
    else:
        add("ERR", f"{command} is not installed or not on PATH")

def check_command_or_raise(command: str) -> None:
    if shutil.which(command) is None:
        raise CliError(f"{command} is not installed or not on PATH", EXIT_CONFIG)

def has_s3_credential_hint() -> bool:
    return any(
        os.getenv(name)
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_PROFILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        )
    )
