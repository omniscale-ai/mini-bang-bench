from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

from mini_bang.client.cli.errors import CliError, EXIT_SECURITY


@dataclass
class DockerNetwork:
    """Dedicated Docker network for one local official agent run."""

    run_id: str
    name: str | None = None

    def __enter__(self) -> str:
        suffix = "".join(ch if ch.isalnum() else "-" for ch in self.run_id.lower())[:24]
        self.name = f"mini-bang-official-{suffix}"
        try:
            subprocess.run(
                ["docker", "network", "create", "--driver", "bridge", self.name],
                text=True,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            message = f"failed to create dedicated Docker network {self.name}"
            if detail:
                message = f"{message}: {detail}"
            raise CliError(message, EXIT_SECURITY) from exc
        return self.name

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.name:
            subprocess.run(
                ["docker", "network", "rm", self.name],
                text=True,
                capture_output=True,
                check=False,
            )


def require_insecure_host_gateway(args: Any) -> bool:
    allow = bool(
        getattr(args, "allow_host_gateway", False)
        or getattr(args, "insecure_host_network", False)
    )
    if not allow:
        raise CliError(
            "local Docker submit no longer exposes the host by default. Use compose/k8s "
            "official mode, or pass --allow-host-gateway with --i-know-this-is-insecure "
            "for trusted local debugging.",
            EXIT_SECURITY,
        )
    if not bool(getattr(args, "i_know_this_is_insecure", False)):
        raise CliError(
            "--allow-host-gateway requires --i-know-this-is-insecure; host gateway "
            "runs are local debug runs and are not official-scoreable.",
            EXIT_SECURITY,
        )
    return allow


def build_agent_docker_command(
    *,
    args: Any,
    agent_image: str,
    network_name: str,
    runner_url: str,
    runner_token: str,
    task_id: str,
    instance_id: str,
    run_id: str,
) -> list[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        network_name,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=256m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--cpus",
        "2",
        "--memory",
        "1g",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-e",
        f"MINI_BANG_RUNNER_URL={runner_url}",
        "-e",
        f"MINI_BANG_RUNNER_TOKEN={runner_token}",
        "-e",
        f"MINI_BANG_TASK_ID={task_id}",
        "-e",
        f"MINI_BANG_INSTANCE_ID={instance_id}",
        "-e",
        f"MINI_BANG_RUN_ID={run_id}",
    ]
    if getattr(args, "agent", None):
        cmd.extend(["-e", f"MINI_BANG_AGENT={args.agent}"])
    if getattr(args, "agent_file", None):
        from mini_bang.client.cli.runtime import container_agent_file

        cmd.extend(["-e", f"MINI_BANG_AGENT_FILE={container_agent_file(args.agent_file)}"])
    cmd.append(agent_image)
    if getattr(args, "agent_command", None):
        cmd.extend(shlex.split(args.agent_command))
    elif getattr(args, "agent", None) or getattr(args, "agent_file", None):
        cmd.extend(["python", "-m", "mini_bang.agent_runtime"])
    return cmd
