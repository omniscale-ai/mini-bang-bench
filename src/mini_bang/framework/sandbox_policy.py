"""Single source of the submitted-code sandbox requirements.

Every path that runs participant code reads its requirements here: one-shot
Kubernetes slots, legacy Kubernetes Jobs and the local Docker runner. Official
evaluation refuses to run code without a sandbox runtime (gVisor ``runsc`` by
default, Kata allowed) unless the explicit, logged dev override is set.

The child resource limits are computed here from ``ExecutionLimits`` and sent
to the pod-side supervisor, which applies them before exec and refuses to
start when they are missing (``mini_bang.pool.offline_sandbox``).
"""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

SANDBOX_POLICY_VERSION = "executable-sandbox"
# Kubernetes RuntimeClass created by the operator; its handler must be one of
# SANDBOX_HANDLERS.
DEFAULT_RUNTIME_CLASS = "gvisor"
DEFAULT_DOCKER_RUNTIME = "runsc"
SANDBOX_HANDLERS = frozenset({"runsc", "kata", "kata-qemu", "kata-clh", "kata-fc", "kata-dragonball"})
RUNTIME_CLASS_ENV = "MINI_BANG_SANDBOX_RUNTIME_CLASS"
DEV_OVERRIDE_ENV = "MINI_BANG_ALLOW_UNSANDBOXED_EXECUTABLE"

MIB = 1024 * 1024
OPEN_FILES = 1024
FILE_SIZE_MB = 512          # largest single file submitted code may write
ADDRESS_SPACE_FACTOR = 2    # RLIMIT_AS per process = factor x memory limit
FALLBACK_ADDRESS_SPACE_MB = 8192
MIN_PROCESSES = 16
OUTPUT_LIMIT_BYTES = 4 * MIB  # output.json bound, below the 8 MiB control message

_LOG = logging.getLogger("mini_bang.sandbox")


class SandboxUnavailable(RuntimeError):
    """Submitted code was refused because the sandbox requirements are not met."""


def dev_override() -> bool:
    return os.getenv(DEV_OVERRIDE_ENV) == "1"


def runtime_class_name(value: Any = None) -> str | None:
    """Configured value wins (an empty string disables), then env, then default."""
    if value is None:
        value = os.getenv(RUNTIME_CLASS_ENV, DEFAULT_RUNTIME_CLASS)
    value = str(value).strip()
    return value or None


def require_sandbox_runtime(runtime: str | None, *, component: str, official: bool = True,
                            handler: str | None = None) -> dict[str, Any]:
    """Return an audit record or refuse; never silently run without the sandbox."""
    verified = bool(runtime) and (handler is None or handler in SANDBOX_HANDLERS)
    record = {"policy_version": SANDBOX_POLICY_VERSION, "component": component,
              "runtime": runtime, "runtime_verified": verified, "dev_override": False}
    if verified or not official:
        return record
    if dev_override():
        _LOG.warning("%s runs submitted code WITHOUT a sandbox runtime because %s=1; "
                     "results are not valid for official evaluation", component, DEV_OVERRIDE_ENV)
        return {**record, "dev_override": True}
    raise SandboxUnavailable(f"{component}: submitted code requires a sandbox runtime "
                             f"(RuntimeClass/handler in {sorted(SANDBOX_HANDLERS)}); "
                             f"set {DEV_OVERRIDE_ENV}=1 only for local development")


def participant_rlimits(limits: Any) -> dict[str, int]:
    """Child rlimits applied by the trusted supervisor before exec."""
    values = limits if isinstance(limits, dict) else dataclasses.asdict(limits)
    processes = values.get("pids_limit")
    if type(processes) is not int or processes < MIN_PROCESSES:
        raise SandboxUnavailable("Submitted code requires an explicit process cap (pids_limit >= 16)")
    memory = values.get("memory_mb")
    address_space = (int(memory) * ADDRESS_SPACE_FACTOR if memory else FALLBACK_ADDRESS_SPACE_MB) * MIB
    return {"RLIMIT_NPROC": processes, "RLIMIT_NOFILE": OPEN_FILES,
            "RLIMIT_FSIZE": FILE_SIZE_MB * MIB, "RLIMIT_CORE": 0, "RLIMIT_AS": address_space}


def docker_sandbox_args(limits: Any, runtime: str | None) -> list[str]:
    """Docker flags equivalent to the pod-side child hardening."""
    rlimits = participant_rlimits(limits)
    args = ["--ulimit", f"nproc={rlimits['RLIMIT_NPROC']}:{rlimits['RLIMIT_NPROC']}",
            "--ulimit", f"nofile={OPEN_FILES}:{OPEN_FILES}",
            "--ulimit", f"fsize={rlimits['RLIMIT_FSIZE']}:{rlimits['RLIMIT_FSIZE']}",
            "--ulimit", "core=0:0", "--user", "10001:10001", "--env", "HOME=/tmp"]
    if runtime:
        args[:0] = ["--runtime", runtime]
    return args


def sandbox_profile(*, isolated_kernel: bool, one_shot: bool, result_channel: str,
                    runtime: str | None) -> dict[str, Any]:
    trusted = result_channel in {"root_supervisor", "docker_daemon"}
    return {"policy_version": SANDBOX_POLICY_VERSION, "isolated_kernel": bool(isolated_kernel),
            "one_shot": bool(one_shot), "result_channel": result_channel,
            "trusted_result_channel": trusted, "runtime": runtime,
            "hardened": bool(isolated_kernel and one_shot and trusted)}


def kubernetes_profile(execution_mode: str, runtime: str | None) -> dict[str, Any]:
    """Pool slots verify the RuntimeClass per Pod and stamp every result; in legacy
    Jobs the participant's own UID writes the result to the container log."""
    if execution_mode == "pool":
        return sandbox_profile(isolated_kernel=True, one_shot=True, result_channel="root_supervisor",
                               runtime="verified per one-shot Pod and checked on every result")
    return sandbox_profile(isolated_kernel=bool(runtime), one_shot=True,
                           result_channel="participant_uid_container_log", runtime=runtime)


def require_hardened_runner(runner: Any) -> dict[str, Any]:
    """Official scoring gate: isolated kernel, one-shot sandbox, unforgeable result."""
    describe = getattr(runner, "sandbox_profile", None)
    profile = describe() if callable(describe) else None
    if profile and profile.get("hardened"):
        return profile
    name = type(runner).__name__
    if dev_override():
        _LOG.warning("official scoring uses unhardened runner %s because %s=1", name, DEV_OVERRIDE_ENV)
        return {**(profile or {}), "dev_override": True}
    raise SandboxUnavailable(f"official scoring refuses {name}: it is not a one-shot sandbox with an isolated "
                             f"kernel and a supervisor-owned result channel ({profile})")
